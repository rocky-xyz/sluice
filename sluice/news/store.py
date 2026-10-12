import hashlib
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Self
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sluice.news.models import AddResult, NewsItem, StoredNews
from sluice.news.timeutil import format_utc, parse_utc

SCHEMA_VERSION = 1

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
    CHECK ((published_at IS NULL) = (time_precision = 'missing'))
);
CREATE TABLE IF NOT EXISTS article_tickers (
    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    ticker     TEXT NOT NULL,
    PRIMARY KEY (article_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_article_tickers_ticker ON article_tickers(ticker);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
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
        self._conn = sqlite3.connect(self.path)
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
                if cursor.rowcount:
                    result.inserted += 1
                else:
                    result.skipped += 1
                article_id = self._conn.execute(
                    "SELECT id FROM articles WHERE url_norm = ?", (key,)
                ).fetchone()[0]
                self._conn.executemany(
                    "INSERT OR IGNORE INTO article_tickers (article_id, ticker) VALUES (?, ?)",
                    [(article_id, ticker) for ticker in item.tickers],
                )
        return result

    def query(
        self,
        ticker: str,
        before: datetime,
        since: datetime | None = None,
        include_invalid: bool = False,
        limit: int | None = None,
    ) -> list[StoredNews]:
        """Articles for `ticker` published strictly before `before`, newest first.

        `since` is an inclusive lower bound. Items without a publish time are
        returned only with `include_invalid`, and then regardless of the window.
        """
        window = "a.published_at < ?"
        params: list[object] = [ticker.strip().upper(), format_utc(before)]
        if since is not None:
            window += " AND a.published_at >= ?"
            params.append(format_utc(since))
        if include_invalid:
            window = f"({window}) OR a.time_precision = 'missing'"
        sql = (
            "SELECT a.*, (SELECT group_concat(ticker) FROM article_tickers"
            " WHERE article_id = a.id) AS tickers"
            " FROM articles a JOIN article_tickers t ON t.article_id = a.id"
            f" WHERE t.ticker = ? AND ({window})"
            " ORDER BY a.published_at DESC, a.id DESC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_to_stored(row) for row in self._conn.execute(sql, params)]


def _to_stored(row: sqlite3.Row) -> StoredNews:
    return StoredNews(
        id=row["id"],
        url=row["url"],
        title=row["title"],
        summary=row["summary"],
        publisher=row["publisher"],
        source=row["source"],
        published_at=parse_utc(row["published_at"]) if row["published_at"] else None,
        time_precision=row["time_precision"],
        fetched_at=parse_utc(row["fetched_at"]),
        tickers=sorted(row["tickers"].split(",")),
    )
