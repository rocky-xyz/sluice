import json
from datetime import UTC, datetime

import pytest
import yfinance
from yfinance.exceptions import YFRateLimitError

from sluice.news.archive import TickerResult, write_run_log
from sluice.news.store import NewsStore
from sluice.news.yahoo import (
    RETRY_DELAYS,
    _retry,
    collect,
    fetch_raw,
    import_archive,
    parse_item,
)

FETCHED = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
LATER = datetime(2027, 1, 1, tzinfo=UTC)
POOL = {"NVDA", "MSFT", "GOOGL"}


def search_item(n: int = 1, **overrides) -> dict:
    """A Yahoo search item, as captured on 2026-10-12 (thumbnail removed)."""
    raw = {
        "uuid": f"0e75cf79-e597-36d3-b10c-39791f20d02{n}",
        "title": f"Quantum Computing Stocks: Darpa Spreads Its Bets {n}",
        "publisher": "Investor's Business Daily",
        "link": f"https://finance.yahoo.com/m/0e75cf79/quantum-{n}.html",
        "providerPublishTime": 1791816456,
        "type": "STORY",
        "relatedTickers": ["IBM", "MSFT", "IONQ", "GOOG", "NVDA"],
    }
    raw.update(overrides)
    return raw


def feed_item(**content) -> dict:
    """A quote-feed item. Built from the documented shape: the feed was down when this was written."""
    body = {
        "title": "Nvidia unveils new chip",
        "summary": "A short summary.",
        "pubDate": "2026-10-12T14:20:00Z",
        "provider": {"displayName": "Reuters"},
        "canonicalUrl": {"url": "https://example.com/nvidia-chip"},
    }
    body.update(content)
    return {"id": "feed-1", "content": body}


@pytest.fixture
def store(tmp_path):
    with NewsStore(tmp_path / "news.db") as s:
        yield s


def test_search_item_is_linked_to_tagged_tickers_in_the_pool():
    item = parse_item(search_item(), "NVDA", FETCHED, POOL)
    assert item.tickers == ["GOOGL", "MSFT", "NVDA"]  # GOOG is an alias; IBM, IONQ are outside
    assert item.published_at == datetime(2026, 10, 12, 14, 47, 36, tzinfo=UTC)
    assert item.time_precision == "exact"
    assert item.publisher == "Investor's Business Daily"
    assert item.source == "yfinance"
    assert item.summary is None
    assert item.fetched_at == FETCHED


def test_search_item_not_tagged_with_the_queried_ticker_is_not_linked_to_it():
    item = parse_item(search_item(relatedTickers=["MSFT"]), "NVDA", FETCHED, POOL)
    assert item.tickers == ["MSFT"]


@pytest.mark.parametrize(
    "overrides",
    [{"relatedTickers": ["IBM"]}, {"relatedTickers": None}, {"title": "  "}, {"title": None}],
)
def test_items_without_pool_ticker_or_title_are_dropped(overrides):
    assert parse_item(search_item(**overrides), "NVDA", FETCHED, POOL) is None


@pytest.mark.parametrize("value", [None, "not a number"])
def test_missing_or_bad_publish_time_is_kept_and_flagged(value):
    item = parse_item(search_item(providerPublishTime=value), "NVDA", FETCHED, POOL)
    assert item.published_at is None
    assert item.time_precision == "missing"


def test_feed_item_belongs_to_the_queried_ticker():
    item = parse_item(feed_item(), "NVDA", FETCHED, POOL)
    assert item.tickers == ["NVDA"]
    assert item.summary == "A short summary."
    assert item.publisher == "Reuters"
    assert item.url == "https://example.com/nvidia-chip"
    assert item.published_at == datetime(2026, 10, 12, 14, 20, tzinfo=UTC)


def test_feed_item_with_null_fields_and_naive_time():
    raw = feed_item(provider=None, canonicalUrl=None, summary=None, pubDate="2026-10-12T14:20:00")
    item = parse_item(raw, "NVDA", FETCHED, POOL)
    assert (item.publisher, item.url, item.summary) == (None, None, None)
    assert item.time_precision == "missing"  # a time without a zone is never guessed


def test_collect_stores_items_and_reports_counts(store):
    batches = {
        "NVDA": [search_item(1), search_item(2, relatedTickers=["IBM"])],
        "MSFT": [search_item(1)],
    }
    results = collect(["nvda", "MSFT"], store, fetch=batches.__getitem__, now=lambda: FETCHED)

    nvda, msft = results
    assert (nvda.fetched, nvda.inserted, nvda.skipped, nvda.dropped) == (2, 1, 0, 1)
    assert (msft.fetched, msft.inserted, msft.skipped, msft.dropped) == (1, 0, 1, 0)
    assert nvda.error is None and msft.error is None
    rows = store.query("MSFT", before=LATER)
    assert len(rows) == 1
    # GOOG is tagged, but GOOGL is not being collected in this run
    assert rows[0].tickers == ["MSFT", "NVDA"]


def test_one_failing_ticker_does_not_stop_the_rest(store):
    def fetch(ticker: str) -> list[dict]:
        if ticker == "NVDA":
            raise ConnectionError("boom")
        return [] if ticker == "KO" else [search_item(1)]

    nvda, ko, msft = collect(["NVDA", "KO", "MSFT"], store, fetch=fetch, now=lambda: FETCHED)
    assert nvda.error == "ConnectionError: boom"
    assert ko.error == "no items returned"
    assert (msft.inserted, msft.error) == (1, None)


def test_raw_items_are_archived_once_per_day_even_when_dropped(store, tmp_path):
    raw_dir = tmp_path / "raw"
    batch = [search_item(1), search_item(2, relatedTickers=["IBM"])]
    for _ in range(2):
        collect(["NVDA"], store, raw_dir=raw_dir, fetch=lambda _t: batch, now=lambda: FETCHED)

    lines = (raw_dir / "yfinance" / "2026-10-13.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["raw"] for r in records] == batch
    assert {r["ticker"] for r in records} == {"NVDA"}
    assert records[0]["fetched_at"] == FETCHED.isoformat()


def test_archive_can_rebuild_the_store(store, tmp_path):
    raw_dir = tmp_path / "raw"
    collect(["NVDA"], store, raw_dir=raw_dir, fetch=lambda _t: [search_item(1)], now=lambda: FETCHED)
    record = json.loads((raw_dir / "yfinance" / "2026-10-13.jsonl").read_text())
    rebuilt = parse_item(
        record["raw"], record["ticker"], datetime.fromisoformat(record["fetched_at"]), {"NVDA"}
    )
    with NewsStore(tmp_path / "rebuilt.db") as other:
        assert other.add([rebuilt]).inserted == 1
        assert other.query("NVDA", before=LATER) == store.query("NVDA", before=LATER)


def test_rate_limit_is_retried_with_backoff():
    slept = []
    calls = iter([YFRateLimitError(), YFRateLimitError(), ["ok"]])

    def call() -> list:
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    assert _retry(call, sleep=slept.append) == ["ok"]
    assert slept == [2.0, 4.0]


def test_rate_limit_that_outlasts_retries_is_raised():
    slept = []

    def call() -> list:
        raise YFRateLimitError()

    with pytest.raises(YFRateLimitError):
        _retry(call, sleep=slept.append)
    assert slept == list(RETRY_DELAYS)


def test_other_errors_are_not_retried():
    slept = []

    def call() -> list:
        raise ConnectionError

    with pytest.raises(ConnectionError):
        _retry(call, sleep=slept.append)
    assert slept == []


class _Feed:
    def __init__(self, items: list):
        self.items = items

    def get_news(self, count: int) -> list:
        return self.items


def _patch_yahoo(monkeypatch, feed: list, search: dict) -> list:
    """Fake Yahoo: `search` maps (news_count, max_results) to the news it answers with."""
    asked = []

    class Search:
        def __init__(self, query: str, news_count: int, max_results: int):
            asked.append((news_count, max_results))
            self.news = search.get((news_count, max_results), [])

    monkeypatch.setattr(yfinance, "Ticker", lambda ticker: _Feed(feed))
    monkeypatch.setattr(yfinance, "Search", Search)
    return asked


def test_feed_is_used_when_it_answers(monkeypatch):
    asked = _patch_yahoo(monkeypatch, feed=["feed item"], search={(50, 0): ["search item"]})
    assert fetch_raw("NVDA") == ["feed item"]
    assert asked == []


def test_empty_search_is_asked_again_in_a_different_shape(monkeypatch):
    asked = _patch_yahoo(monkeypatch, feed=[], search={(40, 0): ["found"]})
    assert fetch_raw("NVDA") == ["found"]
    assert asked == [(50, 0), (50, 1), (40, 0)]


def test_search_that_stays_empty_returns_nothing(monkeypatch):
    asked = _patch_yahoo(monkeypatch, feed=[], search={})
    assert fetch_raw("NVDA") == []
    assert asked == [(50, 0), (50, 1), (40, 0), (25, 0)]


def test_run_log_appends_one_line_per_run(tmp_path):
    results = [TickerResult(ticker="NVDA", fetched=2, inserted=2), TickerResult(ticker="KO", error="x")]
    path = write_run_log(tmp_path / "runs", FETCHED, "yfinance", results)
    write_run_log(tmp_path / "runs", FETCHED, "yfinance", results)

    assert path == tmp_path / "runs" / "2026-10-13.jsonl"
    first, second = (json.loads(line) for line in path.read_text().splitlines())
    assert first == second
    assert first["started_at"] == FETCHED.isoformat()
    assert [r["ticker"] for r in first["results"]] == ["NVDA", "KO"]
    assert first["results"][1]["error"] == "x"


def test_import_rebuilds_the_store_and_is_repeatable(store, tmp_path):
    raw_dir = tmp_path / "raw"
    batches = {"NVDA": [search_item(1), search_item(2, relatedTickers=["IBM"])], "MSFT": [search_item(1)]}
    collect(["NVDA", "MSFT"], store, raw_dir=raw_dir, fetch=batches.__getitem__, now=lambda: FETCHED)

    with NewsStore(tmp_path / "rebuilt.db") as rebuilt:
        first = import_archive(raw_dir, rebuilt)
        assert (first.records, first.inserted, first.skipped, first.dropped) == (3, 1, 1, 1)
        again = import_archive(raw_dir, rebuilt)
        assert (again.inserted, again.skipped) == (0, 2)
        for ticker in ("NVDA", "MSFT"):
            assert rebuilt.query(ticker, before=LATER) == store.query(ticker, before=LATER)


def test_import_links_tickers_that_were_queried_in_any_run(store, tmp_path):
    raw_dir = tmp_path / "raw"
    item = search_item(1, relatedTickers=["NVDA", "MSFT"])
    collect(["NVDA"], store, raw_dir=raw_dir, fetch=lambda _t: [item], now=lambda: FETCHED)
    assert store.query("MSFT", before=LATER) == []  # MSFT was not being collected yet

    other = search_item(9, relatedTickers=["IBM"])
    collect(["MSFT"], store, raw_dir=raw_dir, fetch=lambda _t: [other], now=lambda: FETCHED)
    with NewsStore(tmp_path / "rebuilt.db") as rebuilt:
        import_archive(raw_dir, rebuilt)
        assert [r.title for r in rebuilt.query("MSFT", before=LATER)] == [item["title"]]


def test_import_of_a_missing_archive_is_empty(store, tmp_path):
    assert import_archive(tmp_path / "nothing", store).records == 0


@pytest.mark.integration
def test_yahoo_returns_parsable_news():
    raw_items = fetch_raw("NVDA")
    assert raw_items
    parsed = [parse_item(raw, "NVDA", FETCHED, {"NVDA"}) for raw in raw_items]
    assert any(item is not None and item.published_at is not None for item in parsed)
