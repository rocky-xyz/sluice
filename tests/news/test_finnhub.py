import json
from datetime import UTC, date, datetime

import pytest
import requests

from sluice.news.finnhub import (
    CAP,
    MAX_CONSECUTIVE_FAILURES,
    MIN_INTERVAL,
    RATE_LIMIT_WAITS,
    Client,
    FinnhubError,
    archive,
    backfill,
    collect,
    import_archive,
    parse_item,
)
from sluice.news.store import NewsStore

FETCHED = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
LATER = datetime(2027, 1, 1, tzinfo=UTC)
JAN_11 = datetime(2026, 1, 11, 23, 20, 23, tzinfo=UTC)


def item(n: int = 1, **overrides) -> dict:
    """A company-news item, as captured on 2026-10-13 (image removed)."""
    raw = {
        "category": "company",
        "datetime": 1768173623,
        "headline": f"X Plans ‘Smart Cashtags’ to Link Crypto and Stock Tickers {n}",
        "id": 138068418 + n,
        "related": "NVDA",
        "source": "Yahoo",
        "summary": "X is developing a feature that turns ticker symbols in posts into links.",
        "url": f"https://finnhub.io/api/news?id=8818bb79309899e6bac878717ad9e46cb062f6b{n}",
    }
    raw.update(overrides)
    return raw


@pytest.mark.parametrize(
    ("stamp", "actual"),
    [
        # 2025-10-17 05:30 "UTC" is really 05:30 EDT; 2025-12-08 05:30 is really 05:30 EST
        (datetime(2025, 10, 17, 5, 30, tzinfo=UTC), datetime(2025, 10, 17, 9, 30, tzinfo=UTC)),
        (datetime(2025, 12, 8, 5, 30, tzinfo=UTC), datetime(2025, 12, 8, 10, 30, tzinfo=UTC)),
    ],
)
@pytest.mark.parametrize("source", ["DowJones", "MarketWatch"])
def test_eastern_stamped_sources_are_moved_to_real_utc(source, stamp, actual):
    raw = item(source=source, datetime=int(stamp.timestamp()))
    assert parse_item(raw, "NVDA", FETCHED).published_at == actual
    assert parse_item(item(datetime=raw["datetime"]), "NVDA", FETCHED).published_at == stamp


def test_archive_file_name_ignores_the_time_zone_fix(tmp_path):
    late = int(datetime(2025, 10, 17, 22, 0, tzinfo=UTC).timestamp())  # 02:00 next day after the fix
    archive(tmp_path, "NVDA", FETCHED, [item(source="DowJones", datetime=late)])
    assert [p.name for p in (tmp_path / "finnhub").iterdir()] == ["2025-10-17.jsonl"]


class Response:
    def __init__(self, status_code: int = 200, body: object = None):
        self.status_code = status_code
        self._body = [] if body is None else body

    def json(self) -> object:
        return self._body


class Http:
    """Fake `requests.get` with a clock that only moves when the client sleeps."""

    def __init__(self, *responses: Response):
        self.responses = list(responses)
        self.calls = []
        self.slept = []
        self.time = 0.0

    def get(self, url: str, **kwargs) -> Response:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.time += seconds

    def client(self) -> Client:
        return Client("secret-token", get=self.get, sleep=self.sleep, clock=lambda: self.time)


@pytest.fixture
def store(tmp_path):
    with NewsStore(tmp_path / "news.db") as s:
        yield s


def test_item_is_parsed_and_linked_to_the_queried_ticker():
    parsed = parse_item(item(), "nvda", FETCHED)
    assert parsed.tickers == ["NVDA"]
    assert parsed.published_at == JAN_11
    assert parsed.time_precision == "exact"
    assert parsed.source == "finnhub"
    assert parsed.publisher == "Yahoo"
    assert parsed.summary.startswith("X is developing")
    assert parsed.url.startswith("https://finnhub.io/api/news?id=")


def test_other_share_class_is_stored_under_the_tracked_ticker():
    assert parse_item(item(), "GOOG", FETCHED).tickers == ["GOOGL"]


@pytest.mark.parametrize("headline", [None, "", "   "])
def test_item_without_headline_is_dropped(headline):
    assert parse_item(item(headline=headline), "NVDA", FETCHED) is None


@pytest.mark.parametrize("value", [None, 0, "soon"])
def test_missing_or_bad_time_is_kept_and_flagged(value):
    parsed = parse_item(item(datetime=value), "NVDA", FETCHED)
    assert parsed.published_at is None
    assert parsed.time_precision == "missing"


def test_client_sends_the_key_in_a_header_and_asks_for_one_day():
    http = Http(Response(body=[item()]))
    assert http.client().company_news("NVDA", date(2026, 1, 11)) == [item()]

    (url, kwargs), = http.calls
    assert "secret-token" not in url
    assert kwargs["headers"] == {"X-Finnhub-Token": "secret-token"}
    assert kwargs["params"] == {"symbol": "NVDA", "from": "2026-01-11", "to": "2026-01-11"}


def test_client_spaces_requests_out():
    http = Http(Response(), Response(), Response())
    client = http.client()
    client.company_news("NVDA", date(2026, 1, 11))
    client.company_news("NVDA", date(2026, 1, 12))
    http.time += 5  # a slow caller needs no extra wait
    client.company_news("NVDA", date(2026, 1, 13))
    assert http.slept == [MIN_INTERVAL]


def test_client_waits_and_retries_on_rate_limit():
    http = Http(Response(429), Response(429), Response(body=[item()]))
    assert http.client().company_news("NVDA", date(2026, 1, 11)) == [item()]
    assert http.slept[:2] == list(RATE_LIMIT_WAITS[:2])


def test_client_gives_up_after_repeated_rate_limits():
    http = Http(*[Response(429)] * (len(RATE_LIMIT_WAITS) + 1))
    with pytest.raises(FinnhubError, match="HTTP 429"):
        http.client().company_news("NVDA", date(2026, 1, 11))
    assert not http.responses


@pytest.mark.parametrize(
    "response", [Response(403, {"error": "no access"}), Response(200, {"error": "bad symbol"})]
)
def test_client_raises_on_errors_without_leaking_the_key(response):
    with pytest.raises(FinnhubError) as caught:
        Http(response).client().company_news("NVDA", date(2026, 1, 11))
    assert "secret-token" not in str(caught.value)


def test_archive_files_are_named_by_publish_date(tmp_path):
    items = [item(1), item(2, datetime=1768173623 + 86400), item(3, datetime=None)]
    assert archive(tmp_path, "NVDA", FETCHED, items) == 3
    assert archive(tmp_path, "NVDA", FETCHED, items) == 0  # nothing is written twice
    names = sorted(p.name for p in (tmp_path / "finnhub").iterdir())
    assert names == ["2026-01-11.jsonl", "2026-01-12.jsonl", "undated.jsonl"]


def test_same_article_under_two_tickers_is_stored_once(store, tmp_path):
    archive(tmp_path, "NVDA", FETCHED, [item(1)])
    archive(tmp_path, "AMD", FETCHED, [item(1), item(2, headline=" ")])

    first = import_archive(tmp_path, store)
    assert (first.records, first.inserted, first.skipped, first.dropped) == (3, 1, 1, 1)
    assert import_archive(tmp_path, store).inserted == 0
    (row,) = store.query("AMD", before=LATER)
    assert row.tickers == ["AMD", "NVDA"]
    assert row.published_at == JAN_11


def run_backfill(tmp_path, fetch, tickers=("NVDA", "KO"), start=date(2026, 1, 10), **kwargs):
    return backfill(
        tickers,
        start,
        date(2026, 1, 11),
        fetch,
        tmp_path / "raw",
        tmp_path / "backfill" / "finnhub.jsonl",
        now=lambda: FETCHED,
        **kwargs,
    )


def test_backfill_asks_for_each_ticker_and_day_oldest_first(tmp_path):
    asked = []

    def fetch(ticker: str, day: date) -> list[dict]:
        asked.append((ticker, day.isoformat()))
        return [item(len(asked))]

    steps = []
    result = run_backfill(tmp_path, fetch, on_step=lambda done, total: steps.append((done, total)))

    assert asked == [
        ("NVDA", "2026-01-10"),
        ("KO", "2026-01-10"),
        ("NVDA", "2026-01-11"),
        ("KO", "2026-01-11"),
    ]
    assert (result.requested, result.items, result.archived, result.failed) == (4, 4, 4, 0)
    assert steps == [(1, 4), (2, 4), (3, 4), (4, 4)]
    lines = (tmp_path / "backfill" / "finnhub.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == {
        "ticker": "NVDA",
        "day": "2026-01-10",
        "items": 1,
        "capped": False,
        "fetched_at": FETCHED.isoformat(),
    }


def test_backfill_resumes_without_repeating_finished_pairs(tmp_path):
    asked = []

    def flaky(ticker: str, day: date) -> list[dict]:
        if (ticker, day.day) == ("KO", 10):
            raise FinnhubError("HTTP 500")
        asked.append((ticker, day.day))
        return []

    first = run_backfill(tmp_path, flaky)
    assert (first.requested, first.failed, first.aborted) == (3, 1, None)

    asked.clear()
    second = run_backfill(tmp_path, lambda ticker, day: asked.append((ticker, day.day)) or [])
    assert asked == [("KO", 10)]
    assert (second.requested, second.already_done) == (1, 3)


def test_backfill_stops_after_failures_in_a_row(tmp_path):
    def down(ticker: str, day: date) -> list[dict]:
        raise requests.ConnectionError("offline")

    result = run_backfill(tmp_path, down, start=date(2026, 1, 1))
    assert result.failed == MAX_CONSECUTIVE_FAILURES
    assert "offline" in result.aborted
    assert result.requested == 0


def test_backfill_flags_days_that_hit_the_cap(tmp_path):
    def fetch(ticker: str, day: date) -> list[dict]:
        return [item(n) for n in range(CAP if ticker == "NVDA" else CAP - 1)]

    result = run_backfill(tmp_path, fetch, start=date(2026, 1, 11))
    assert result.capped == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "backfill" / "finnhub.jsonl").read_text().splitlines()
    ]
    assert [(r["ticker"], r["capped"]) for r in records] == [("NVDA", True), ("KO", False)]


def test_collect_asks_for_yesterday_and_today_and_stores_them(store, tmp_path):
    asked = []

    def fetch(ticker: str, day: date) -> list[dict]:
        asked.append((ticker, day.isoformat()))
        return [item(day.day), item(99, headline="")] if ticker == "NVDA" else []

    nvda, ko = collect(["nvda", "KO"], store, fetch, tmp_path / "raw", now=lambda: FETCHED)

    assert asked == [
        ("NVDA", "2026-10-12"),
        ("NVDA", "2026-10-13"),
        ("KO", "2026-10-12"),
        ("KO", "2026-10-13"),
    ]
    assert (nvda.fetched, nvda.inserted, nvda.dropped, nvda.error) == (4, 2, 2, None)
    assert (ko.fetched, ko.error) == (0, None)  # a quiet ticker is not a failure
    assert len(store.query("NVDA", before=LATER)) == 2
    assert (tmp_path / "raw" / "finnhub" / "2026-01-11.jsonl").exists()


def test_collect_reports_a_failing_ticker_and_carries_on(store):
    def fetch(ticker: str, day: date) -> list[dict]:
        if ticker == "NVDA":
            raise FinnhubError("HTTP 500")
        return [item()]

    nvda, ko = collect(["NVDA", "KO"], store, fetch, now=lambda: FETCHED)
    assert nvda.error == "FinnhubError: HTTP 500"
    assert (ko.inserted, ko.error) == (1, None)
