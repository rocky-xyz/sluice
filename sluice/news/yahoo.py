import json
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import yfinance as yf
from pydantic import BaseModel
from yfinance.exceptions import YFRateLimitError

from sluice.news.models import NewsItem
from sluice.news.store import NewsStore
from sluice.news.timeutil import to_utc
from sluice.universe import canonical

SOURCE = "yfinance"
RETRY_DELAYS = (2.0, 4.0, 8.0)  # seconds; rate limits only

# yfinance turns some failed requests into an empty result; make them raise instead.
yf.config.debug.hide_exceptions = False


class TickerResult(BaseModel):
    ticker: str
    fetched: int = 0
    inserted: int = 0
    skipped: int = 0  # already stored
    dropped: int = 0  # no title, or not tagged with any collected ticker
    error: str | None = None


def _retry(call: Callable[[], list], sleep: Callable[[float], None] = time.sleep) -> list:
    for delay in RETRY_DELAYS:
        try:
            return call()
        except YFRateLimitError:
            sleep(delay)
    return call()


def fetch_raw(ticker: str, count: int = 50) -> list[dict]:
    """Raw news items for one ticker: the quote feed, or Yahoo search when the feed is empty."""
    feed = _retry(lambda: yf.Ticker(ticker).get_news(count=count))
    if feed:
        return feed
    return _retry(lambda: yf.Search(ticker, news_count=count, max_results=0).news) or []


def _raw_id(raw: dict) -> str | None:
    return raw.get("id") or raw.get("uuid")


def _parse_time(raw: dict, content: dict | None) -> datetime | None:
    try:
        if content is not None:
            return to_utc(datetime.fromisoformat(content["pubDate"]))
        return datetime.fromtimestamp(raw["providerPublishTime"], UTC)
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return None


def parse_item(
    raw: dict, ticker: str, fetched_at: datetime, allowed: set[str]
) -> NewsItem | None:
    """Turn one raw item into a NewsItem, or None when it should not be stored.

    Yahoo has two shapes: feed items nest everything under "content", search
    items are flat. Feed items belong to the queried ticker; search items are
    only loosely matched, so they are linked through Yahoo's own ticker tags.
    """
    content = raw.get("content") if isinstance(raw.get("content"), dict) else None
    if content is not None:
        link = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        fields = {
            "title": content.get("title"),
            "url": link.get("url"),
            "summary": content.get("summary"),
            "publisher": (content.get("provider") or {}).get("displayName"),
        }
        finance = content.get("finance") or {}
        tagged = [t.get("symbol") for t in finance.get("stockTickers") or []] + [ticker]
    else:
        fields = {
            "title": raw.get("title"),
            "url": raw.get("link"),
            "summary": None,
            "publisher": raw.get("publisher"),
        }
        tagged = raw.get("relatedTickers") or []

    title = (fields["title"] or "").strip()
    tickers = {canonical(t) for t in tagged if isinstance(t, str)} & allowed
    if not title or not tickers:
        return None
    published_at = _parse_time(raw, content)
    return NewsItem(
        title=title,
        url=fields["url"] or None,
        summary=fields["summary"] or None,
        publisher=fields["publisher"] or None,
        source=SOURCE,
        tickers=sorted(tickers),
        published_at=published_at,
        time_precision="exact" if published_at else "missing",
        fetched_at=fetched_at,
    )


def _archive(raw_dir: Path, ticker: str, fetched_at: datetime, items: list[dict]) -> None:
    """Append raw items to today's JSONL file, once per (ticker, item id) per day."""
    path = raw_dir / SOURCE / f"{fetched_at:%Y-%m-%d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        with path.open() as lines:
            for line in lines:
                record = json.loads(line)
                seen.add((record["ticker"], _raw_id(record["raw"])))
    with path.open("a") as out:
        for raw in items:
            key = (ticker, _raw_id(raw))
            if key[1] is not None and key in seen:
                continue
            seen.add(key)
            record = {"fetched_at": fetched_at.isoformat(), "ticker": ticker, "raw": raw}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect(
    tickers: Iterable[str],
    store: NewsStore,
    raw_dir: Path | None = None,
    fetch: Callable[[str], list[dict]] = fetch_raw,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[TickerResult]:
    """Fetch, archive and store news for each ticker. One failing ticker does not stop the rest."""
    symbols = list(dict.fromkeys(canonical(t) for t in tickers))
    allowed = set(symbols)
    results = []
    for symbol in symbols:
        result = TickerResult(ticker=symbol)
        results.append(result)
        try:
            raw_items = fetch(symbol)
        except Exception as exc:  # noqa: BLE001 - any vendor failure is reported, not raised
            result.error = f"{type(exc).__name__}: {exc}".rstrip(": ")
            continue
        result.fetched = len(raw_items)
        if not raw_items:
            # An empty answer is far more likely an outage than a day without news.
            result.error = "no items returned"
            continue
        fetched_at = now()
        if raw_dir is not None:
            _archive(raw_dir, symbol, fetched_at, raw_items)
        items = [parse_item(raw, symbol, fetched_at, allowed) for raw in raw_items]
        kept = [item for item in items if item is not None]
        result.dropped = len(items) - len(kept)
        added = store.add(kept)
        result.inserted, result.skipped = added.inserted, added.skipped
    return results
