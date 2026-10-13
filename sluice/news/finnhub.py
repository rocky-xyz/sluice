import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests
from pydantic import BaseModel

from sluice.news.archive import (
    ImportResult,
    TickerResult,
    append_records,
    archive_files,
    import_files,
)
from sluice.news.models import NewsItem
from sluice.news.publishers import publisher_for
from sluice.news.store import NewsStore
from sluice.news.timeutil import ET
from sluice.universe import canonical

SOURCE = "finnhub"
URL = "https://finnhub.io/api/v1/company-news"
MIN_INTERVAL = 1.1  # seconds between requests; the free plan allows 60 a minute
RATE_LIMIT_WAITS = (30.0, 60.0, 60.0)  # seconds; HTTP 429 only
# One request returns at most about 245 items and keeps the newest. A day at or
# above this count may have lost its earliest items.
CAP = 240
MAX_CONSECUTIVE_FAILURES = 5
# "Yahoo" is where Finnhub found the article, not who published it: Yahoo carries
# dozens of outlets. Storing it would lump them into one publisher.
AGGREGATORS = frozenset({"Yahoo"})
# Sources whose timestamps are US Eastern wall-clock time stored as if it were UTC.
# Measured on a year of data (2026-10-13). MarketWatch and DowJones: the same
# headline is stamped exactly 4 h (EDT) or 5 h (EST) before its Yahoo copy. Benzinga,
# SeekingAlpha and CNBC: fixed columns land at the right hour only when read as
# Eastern ("Wall Street Breakfast" at 06-07, "moves midday" at 12, after-market
# recaps after 16:00) and keep the same stamped hour in winter and summer.
# Yahoo and ChartMill are true UTC. Finnhub and Fintel are too few to tell.
EASTERN_STAMPED_SOURCES = frozenset(
    {"Benzinga", "CNBC", "DowJones", "MarketWatch", "SeekingAlpha"}
)

Fetch = Callable[[str, date], list[dict]]


class FinnhubError(RuntimeError):
    pass


class Client:
    """Company news for one ticker and one UTC day. The key travels in a header, never in a URL."""

    def __init__(
        self,
        token: str,
        get: Callable[..., requests.Response] = requests.get,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._token = token
        self._get = get
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = MIN_INTERVAL - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def company_news(self, ticker: str, day: date) -> list[dict]:
        params = {"symbol": ticker, "from": day.isoformat(), "to": day.isoformat()}
        for wait in (*RATE_LIMIT_WAITS, None):
            self._throttle()
            response = self._get(
                URL, params=params, headers={"X-Finnhub-Token": self._token}, timeout=30
            )
            if response.status_code != 429:
                break
            if wait is not None:
                self._sleep(wait)
        if response.status_code != 200:
            raise FinnhubError(f"HTTP {response.status_code}")
        body = response.json()
        if not isinstance(body, list):
            raise FinnhubError(f"unexpected answer: {str(body)[:200]}")
        return body


def _stamped_at(raw: dict) -> datetime | None:
    """The timestamp exactly as Finnhub gives it."""
    try:
        return datetime.fromtimestamp(raw["datetime"], UTC) if raw["datetime"] else None
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return None


def _published_at(raw: dict) -> datetime | None:
    """Publish time in UTC, with Finnhub's time zone error for some sources undone.

    For these sources Finnhub stores the US Eastern wall-clock time as if it
    were UTC, which dates an article four or five hours before it came out.
    """
    stamped = _stamped_at(raw)
    if stamped is not None and raw.get("source") in EASTERN_STAMPED_SOURCES:
        return stamped.replace(tzinfo=ET).astimezone(UTC)
    return stamped


def parse_item(
    raw: dict, ticker: str, fetched_at: datetime, location: str | None = None
) -> NewsItem | None:
    """Turn one raw item into a NewsItem, or None when it has no headline.

    Finnhub's "source" is the publisher, except for aggregators. For those the
    publisher is read from `location`, the address the article's link leads to,
    when it is known and is the publisher's own site. "url" stays the Finnhub
    redirect: it identifies the article whether or not its link was followed.
    """
    title = (raw.get("headline") or "").strip()
    if not title:
        return None
    published_at = _published_at(raw)
    return NewsItem(
        title=title,
        url=raw.get("url") or None,
        summary=raw.get("summary") or None,
        publisher=(
            publisher_for(location)
            if raw.get("source") in AGGREGATORS
            else raw.get("source") or None
        ),
        source=SOURCE,
        tickers=[canonical(ticker)],
        published_at=published_at,
        time_precision="exact" if published_at else "missing",
        fetched_at=fetched_at,
    )


def archive(raw_dir: Path, ticker: str, fetched_at: datetime, items: Iterable[dict]) -> int:
    """Archive raw items by publish date, so a year of backfill is spread over a year of files."""
    by_day: dict[str, list[dict]] = defaultdict(list)
    for raw in items:
        published_at = _stamped_at(raw)  # file names must not move when the fix changes
        by_day[f"{published_at:%Y-%m-%d}" if published_at else "undated"].append(raw)
    return sum(
        append_records(raw_dir / SOURCE / f"{name}.jsonl", ticker, fetched_at, day_items)
        for name, day_items in by_day.items()
    )


def import_archive(raw_dir: Path, store: NewsStore) -> ImportResult:
    from sluice.news.finnhub_links import load_links

    links = load_links(raw_dir)

    def parse(raw: dict, ticker: str, fetched_at: datetime) -> NewsItem | None:
        return parse_item(raw, ticker, fetched_at, links.get(raw.get("id")))

    return import_files(archive_files(raw_dir, SOURCE), parse, store)


def collect(
    tickers: Iterable[str],
    store: NewsStore,
    fetch: Fetch,
    raw_dir: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[TickerResult]:
    """Fetch, archive and store today's and yesterday's news (UTC) for each ticker.

    Yesterday is asked again so that nothing published between two runs around
    midnight is missed. An empty answer is not an error here: a quiet ticker can
    have a day without news, and Finnhub reports its failures as HTTP errors.
    """
    results = []
    for symbol in dict.fromkeys(canonical(t) for t in tickers):
        result = TickerResult(ticker=symbol)
        results.append(result)
        fetched_at = now()
        try:
            raw_items = [
                raw
                for day in (fetched_at.date() - timedelta(days=1), fetched_at.date())
                for raw in fetch(symbol, day)
            ]
        except (FinnhubError, requests.RequestException) as exc:
            result.error = f"{type(exc).__name__}: {exc}".rstrip(": ")
            continue
        result.fetched = len(raw_items)
        if raw_dir is not None:
            archive(raw_dir, symbol, fetched_at, raw_items)
        items = [parse_item(raw, symbol, fetched_at) for raw in raw_items]
        kept = [item for item in items if item is not None]
        result.dropped = len(items) - len(kept)
        added = store.add(kept)
        result.inserted, result.skipped = added.inserted, added.skipped
    return results


class BackfillResult(BaseModel):
    requested: int = 0  # (ticker, day) pairs fetched in this run
    already_done: int = 0
    items: int = 0
    archived: int = 0  # new lines written; the rest were archived before
    capped: int = 0  # days that hit the per-request cap and may be incomplete
    failed: int = 0
    aborted: str | None = None


def _done(progress: Path) -> set[tuple[str, str]]:
    if not progress.exists():
        return set()
    with progress.open() as lines:
        return {(r["ticker"], r["day"]) for r in map(json.loads, lines)}


def backfill(
    tickers: Iterable[str],
    start: date,
    end: date,
    fetch: Fetch,
    raw_dir: Path,
    progress: Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    on_step: Callable[[int, int], None] = lambda done, total: None,
) -> BackfillResult:
    """Archive company news for every ticker and day from `start` to `end`, oldest day first.

    One request covers one ticker and one day: wider windows return fewer items
    per day. Each finished pair is recorded in `progress`, so an interrupted run
    resumes where it stopped. Oldest first, because the free plan's one-year
    window moves forward every day.
    """
    symbols = list(dict.fromkeys(canonical(t) for t in tickers))
    days = [start + timedelta(days=n) for n in range((end - start).days + 1)]
    pairs = [(symbol, day) for day in days for symbol in symbols]
    done = _done(progress)
    progress.parent.mkdir(parents=True, exist_ok=True)

    result = BackfillResult()
    failures_in_a_row = 0
    for step, (symbol, day) in enumerate(pairs, start=1):
        on_step(step, len(pairs))
        if (symbol, day.isoformat()) in done:
            result.already_done += 1
            continue
        try:
            items = fetch(symbol, day)
        except (FinnhubError, requests.RequestException) as exc:
            result.failed += 1
            failures_in_a_row += 1
            if failures_in_a_row >= MAX_CONSECUTIVE_FAILURES:
                result.aborted = f"{failures_in_a_row} requests failed in a row; last: {exc}"
                break
            continue
        failures_in_a_row = 0
        fetched_at = now()
        capped = len(items) >= CAP
        result.requested += 1
        result.items += len(items)
        result.capped += capped
        result.archived += archive(raw_dir, symbol, fetched_at, items)
        record = {
            "ticker": symbol,
            "day": day.isoformat(),
            "items": len(items),
            "capped": capped,
            "fetched_at": fetched_at.isoformat(),
        }
        with progress.open("a") as out:
            out.write(json.dumps(record) + "\n")
    return result
