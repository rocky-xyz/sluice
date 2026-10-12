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
    append_records,
    archive_files,
    import_files,
)
from sluice.news.models import NewsItem
from sluice.news.store import NewsStore
from sluice.universe import canonical

SOURCE = "finnhub"
URL = "https://finnhub.io/api/v1/company-news"
MIN_INTERVAL = 1.1  # seconds between requests; the free plan allows 60 a minute
RATE_LIMIT_WAITS = (30.0, 60.0, 60.0)  # seconds; HTTP 429 only
# One request returns at most about 245 items and keeps the newest. A day at or
# above this count may have lost its earliest items.
CAP = 240
MAX_CONSECUTIVE_FAILURES = 5

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


def _published_at(raw: dict) -> datetime | None:
    try:
        return datetime.fromtimestamp(raw["datetime"], UTC) if raw["datetime"] else None
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return None


def parse_item(raw: dict, ticker: str, fetched_at: datetime) -> NewsItem | None:
    """Turn one raw item into a NewsItem, or None when it has no headline.

    Finnhub's "source" is an aggregator (Yahoo, Benzinga, ...), not always the
    original publisher, and "url" is a Finnhub redirect to the article.
    """
    title = (raw.get("headline") or "").strip()
    if not title:
        return None
    published_at = _published_at(raw)
    return NewsItem(
        title=title,
        url=raw.get("url") or None,
        summary=raw.get("summary") or None,
        publisher=raw.get("source") or None,
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
        published_at = _published_at(raw)
        by_day[f"{published_at:%Y-%m-%d}" if published_at else "undated"].append(raw)
    return sum(
        append_records(raw_dir / SOURCE / f"{name}.jsonl", ticker, fetched_at, day_items)
        for name, day_items in by_day.items()
    )


def import_archive(raw_dir: Path, store: NewsStore) -> ImportResult:
    return import_files(archive_files(raw_dir, SOURCE), parse_item, store)


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
