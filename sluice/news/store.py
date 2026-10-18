import hashlib
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Self
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sluice.market_calendar import session_of
from sluice.news.models import AddResult, NewsItem, StoredNews
from sluice.news.timeutil import format_utc, parse_utc

# Also bumped when stored values change meaning, so that a stale database is rebuilt.
SCHEMA_VERSION = 3

# Same normalized title within this long of the first copy counts as a reprint.
# Measured on Finnhub data: real reprints fall within hours, while identically
# titled daily columns come 24 h apart.
REPRINT_WINDOW = timedelta(hours=12)

# No migrations: the database can always be rebuilt from the raw JSONL archive.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id             INTEGER PRIMARY KEY,
    url            TEXT,
    url_norm       TEXT NOT NULL UNIQUE,
    title          TEXT NOT NULL,
    summary        TEXT,
    publisher      TEXT,
    source         TEXT NOT NULL,
    published_at   TEXT,
    time_precision TEXT NOT NULL CHECK (time_precision IN ('exact', 'day', 'missing')),
    fetched_at     TEXT NOT NULL,
    cluster_id     INTEGER,  -- id of the earliest article of its reprint cluster; own id if alone
    CHECK ((published_at IS NULL) = (time_precision = 'missing'))
);
CREATE TABLE IF NOT EXISTS article_tickers (
    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    ticker     TEXT NOT NULL,
    PRIMARY KEY (article_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_article_tickers_ticker ON article_tickers(ticker);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_cluster_id ON articles(cluster_id);
"""


def normalize_url(url: str) -> str:
    """Canonical form used as the dedup key: same article, same string."""
    parts = urlsplit(url.strip())
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
        )
    )
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, "")
    )


def normalize_title(title: str) -> str:
    """Lowercase, with punctuation and whitespace removed."""
    return "".join(ch for ch in title.lower() if ch.isalnum())


def _dedup_key(item: NewsItem) -> str:
    if item.url and item.url.strip():
        return normalize_url(item.url)
    published = format_utc(item.published_at) if item.published_at else ""
    raw = "\n".join((item.source, normalize_title(item.title), published))
    return "nourl:" + hashlib.sha256(raw.encode()).hexdigest()


class NewsStore:
    """SQLite news store. Readers go through `query`, which is point-in-time."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A pipeline run may read from a worker thread; each run has its own store.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise RuntimeError(
                f"{self.path} has schema version {version}, expected {SCHEMA_VERSION}; "
                "rebuild it from the raw JSONL archive"
            )
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(self, items: Iterable[NewsItem]) -> AddResult:
        """Store items in one transaction. A known article only gains its new tickers."""
        result = AddResult()
        with self._conn:
            for item in items:
                key = _dedup_key(item)
                cursor = self._conn.execute(
                    "INSERT INTO articles (url, url_norm, title, summary, publisher, source,"
                    " published_at, time_precision, fetched_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(url_norm) DO NOTHING",
                    (
                        item.url,
                        key,
                        item.title,
                        item.summary,
                        item.publisher,
                        item.source,
                        format_utc(item.published_at) if item.published_at else None,
                        item.time_precision,
                        format_utc(item.fetched_at),
                    ),
                )
                article_id = self._conn.execute(
                    "SELECT id FROM articles WHERE url_norm = ?", (key,)
                ).fetchone()[0]
                if cursor.rowcount:
                    result.inserted += 1
                    # Alone until `rebuild_clusters` says otherwise.
                    self._conn.execute(
                        "UPDATE articles SET cluster_id = id WHERE id = ?", (article_id,)
                    )
                else:
                    result.skipped += 1
                    # A later copy may know what the first did not. It fills gaps, never overwrites.
                    self._conn.execute(
                        "UPDATE articles SET publisher = COALESCE(publisher, ?),"
                        " summary = COALESCE(summary, ?) WHERE id = ?",
                        (item.publisher, item.summary, article_id),
                    )
                self._conn.executemany(
                    "INSERT OR IGNORE INTO article_tickers (article_id, ticker) VALUES (?, ?)",
                    [(article_id, ticker) for ticker in item.tickers],
                )
        return result

    def rebuild_clusters(self, window: timedelta = REPRINT_WINDOW) -> int:
        """Group reprints: same normalized title, published within `window` of the first copy.

        Everything is recomputed, so the result does not depend on the order in
        which articles arrived. Call it after adding articles. Returns the number
        of articles that are reprints of an earlier one.
        """
        rows = self._conn.execute(
            "SELECT id, title, published_at FROM articles"
            " WHERE published_at IS NOT NULL ORDER BY published_at, id"
        ).fetchall()
        anchors: dict[str, tuple[int, datetime]] = {}
        assignments = []
        reprints = 0
        for row in rows:
            key = normalize_title(row["title"])
            published = parse_utc(row["published_at"])
            anchor = anchors.get(key)
            if key and anchor is not None and published - anchor[1] <= window:
                assignments.append((anchor[0], row["id"]))
                reprints += 1
            else:
                anchors[key] = (row["id"], published)
                assignments.append((row["id"], row["id"]))
        with self._conn:
            self._conn.execute("UPDATE articles SET cluster_id = id WHERE published_at IS NULL")
            self._conn.executemany("UPDATE articles SET cluster_id = ? WHERE id = ?", assignments)
        return reprints

    def query(
        self,
        ticker: str,
        before: datetime,
        since: datetime | None = None,
        include_invalid: bool = False,
        limit: int | None = None,
        dedup: bool = True,
    ) -> list[StoredNews]:
        """Articles for `ticker` published strictly before `before`, newest first.

        `since` is an inclusive lower bound. Items without a publish time are
        returned only with `include_invalid`, and then regardless of the window.

        With `dedup`, a story and its reprints come back once, as the earliest
        copy. Only copies published before `before` are counted and contribute
        tickers, so nothing here depends on what was published later.
        `reprint_count` is the largest number of copies any single data vendor
        carried, minus one: the same article collected from two vendors is not
        a reprint. When the earliest copy names no publisher, the publisher of
        another visible copy is used.
        """
        params: dict[str, object] = {
            "ticker": ticker.strip().upper(),
            "before": format_utc(before),
        }
        window = "a.published_at < :before"
        if since is not None:
            window += " AND a.published_at >= :since"
            params["since"] = format_utc(since)
        if include_invalid:
            window = f"({window}) OR a.time_precision = 'missing'"
        if dedup:
            visible = "m.cluster_id = a.id AND (m.id = a.id OR m.published_at < :before)"
            sql = (
                "SELECT a.*,"
                " COALESCE(a.publisher, (SELECT m.publisher FROM articles m"
                f"  WHERE {visible} AND m.publisher IS NOT NULL"
                "  ORDER BY m.published_at, m.id LIMIT 1)) AS known_publisher,"
                " (SELECT group_concat(DISTINCT t.ticker) FROM articles m"
                f"  JOIN article_tickers t ON t.article_id = m.id WHERE {visible}) AS tickers,"
                " (SELECT max(copies) FROM (SELECT count(*) AS copies FROM articles m"
                f"  WHERE {visible} GROUP BY m.source)) - 1 AS reprint_count"
                " FROM articles a"
                f" WHERE a.cluster_id = a.id AND ({window})"
                " AND EXISTS (SELECT 1 FROM articles m"
                "  JOIN article_tickers t ON t.article_id = m.id"
                f"  WHERE {visible} AND t.ticker = :ticker)"
            )
        else:
            sql = (
                "SELECT a.*, a.publisher AS known_publisher, 0 AS reprint_count,"
                " (SELECT group_concat(ticker)"
                " FROM article_tickers WHERE article_id = a.id) AS tickers"
                " FROM articles a JOIN article_tickers t ON t.article_id = a.id"
                f" WHERE t.ticker = :ticker AND ({window})"
            )
        sql += " ORDER BY a.published_at DESC, a.id DESC"
        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = limit
        return [_to_stored(row) for row in self._conn.execute(sql, params)]

    def get(self, news_id: int) -> StoredNews | None:
        """One article by id, with its full ticker list. No point-in-time windowing."""
        row = self._conn.execute(
            "SELECT a.*, a.publisher AS known_publisher, 0 AS reprint_count,"
            " (SELECT group_concat(ticker) FROM article_tickers WHERE article_id = a.id) AS tickers"
            " FROM articles a WHERE a.id = ?",
            (news_id,),
        ).fetchone()
        return _to_stored(row) if row else None

    def missing_publisher_urls(self, source: str) -> dict[int, str]:
        """Id -> url for every article from `source` with no publisher yet and a url to try."""
        rows = self._conn.execute(
            "SELECT id, url FROM articles WHERE source = ? AND publisher IS NULL AND url IS NOT NULL",
            (source,),
        )
        return {row["id"]: row["url"] for row in rows}

    def fill_publisher(self, article_id: int, publisher: str) -> bool:
        """Set a still-missing publisher. Returns whether it was empty and got set."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE articles SET publisher = ? WHERE id = ? AND publisher IS NULL",
                (publisher, article_id),
            )
        return cursor.rowcount > 0


def _to_stored(row: sqlite3.Row) -> StoredNews:
    published_at = parse_utc(row["published_at"]) if row["published_at"] else None
    return StoredNews(
        id=row["id"],
        url=row["url"],
        title=row["title"],
        summary=row["summary"],
        publisher=row["known_publisher"],
        source=row["source"],
        published_at=published_at,
        time_precision=row["time_precision"],
        fetched_at=parse_utc(row["fetched_at"]),
        tickers=sorted(row["tickers"].split(",")),
        reprint_count=row["reprint_count"],
        session=session_of(published_at) if row["time_precision"] == "exact" else None,
    )
