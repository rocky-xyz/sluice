import sqlite3
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from sluice.news.models import NewsItem
from sluice.news.store import NewsStore, normalize_url
from sluice.news.timeutil import ET, end_of_day_et

FETCHED = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
T = datetime(2026, 3, 2, 15, 0, tzinfo=UTC)
NAIVE = T.replace(tzinfo=None)


def make(n: int = 1, **overrides) -> NewsItem:
    fields = {
        "title": f"Headline {n}",
        "url": f"https://example.com/a/{n}",
        "source": "yfinance",
        "tickers": ["NVDA"],
        "published_at": T - timedelta(hours=1),
        "fetched_at": FETCHED,
    }
    fields.update(overrides)
    return NewsItem(**fields)


@pytest.fixture
def store(tmp_path):
    with NewsStore(tmp_path / "news.db") as s:
        yield s


def titles(rows) -> list[str]:
    return [r.title for r in rows]


def test_query_is_strictly_before_decision_time(store):
    store.add(
        [
            make(1, title="at T", published_at=T),
            make(2, title="before T", published_at=T - timedelta(seconds=1)),
            make(3, title="after T", published_at=T + timedelta(seconds=1)),
        ]
    )
    assert titles(store.query("NVDA", before=T)) == ["before T"]


def test_subsecond_item_does_not_leak(store):
    store.add([make(1, published_at=T + timedelta(milliseconds=500))])
    assert store.query("NVDA", before=T + timedelta(milliseconds=900)) == []
    assert len(store.query("NVDA", before=T + timedelta(seconds=1))) == 1


def test_missing_timestamp_hidden_by_default(store):
    store.add([make(1, published_at=None, time_precision="missing")])
    assert store.query("NVDA", before=T) == []
    rows = store.query("NVDA", before=T, include_invalid=True)
    assert len(rows) == 1
    assert rows[0].published_at is None
    assert rows[0].time_precision == "missing"


def test_day_precision_item_visible_only_from_next_day(store):
    day = date(2026, 3, 2)
    store.add([make(1, published_at=end_of_day_et(day), time_precision="day")])
    assert store.query("NVDA", before=datetime(2026, 3, 2, 16, 0, tzinfo=ET)) == []
    rows = store.query("NVDA", before=datetime(2026, 3, 3, 0, 0, tzinfo=ET))
    assert len(rows) == 1
    assert rows[0].time_precision == "day"


def test_same_url_under_two_tickers_is_one_article(store, tmp_path):
    assert store.add([make(1, tickers=["NVDA"])]).inserted == 1
    second = store.add([make(1, tickers=["amd"])])
    assert (second.inserted, second.skipped) == (0, 1)

    conn = sqlite3.connect(tmp_path / "news.db")
    assert conn.execute("SELECT count(*) FROM articles").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM article_tickers").fetchone()[0] == 2
    conn.close()

    for ticker in ("NVDA", "AMD"):
        rows = store.query(ticker, before=T)
        assert len(rows) == 1
        assert rows[0].tickers == ["AMD", "NVDA"]


def test_add_is_idempotent(store):
    batch = [make(1), make(2)]
    assert store.add(batch).inserted == 2
    again = store.add(batch)
    assert (again.inserted, again.skipped) == (0, 2)
    assert len(store.query("NVDA", before=T)) == 2


def test_naive_datetimes_are_rejected(store):
    with pytest.raises(ValidationError):
        make(1, published_at=NAIVE)
    with pytest.raises(ValidationError):
        make(1, fetched_at=NAIVE)
    with pytest.raises(ValueError):
        store.query("NVDA", before=NAIVE)
    with pytest.raises(ValueError):
        store.query("NVDA", before=T, since=NAIVE)


def test_precision_must_match_timestamp():
    with pytest.raises(ValidationError):
        make(1, published_at=None)
    with pytest.raises(ValidationError):
        make(1, time_precision="missing")


@pytest.mark.parametrize(
    ("local", "expected_utc"),
    [
        (datetime(2026, 1, 15, 9, 30, tzinfo=ET), datetime(2026, 1, 15, 14, 30, tzinfo=UTC)),
        (datetime(2026, 7, 15, 9, 30, tzinfo=ET), datetime(2026, 7, 15, 13, 30, tzinfo=UTC)),
    ],
)
def test_eastern_time_is_stored_as_utc(store, local, expected_utc):
    store.add([make(1, published_at=local)])
    (row,) = store.query("NVDA", before=expected_utc + timedelta(seconds=1))
    assert row.published_at == expected_utc
    assert row.published_at.utcoffset() == timedelta(0)


def test_end_of_day_et_follows_daylight_saving():
    assert end_of_day_et(date(2026, 1, 15)) == datetime(2026, 1, 16, 4, 59, 59, tzinfo=UTC)
    assert end_of_day_et(date(2026, 7, 15)) == datetime(2026, 7, 16, 3, 59, 59, tzinfo=UTC)


def test_url_normalization():
    canonical = normalize_url("https://example.com/a/1?id=7")
    variants = [
        "https://EXAMPLE.com/a/1?id=7",
        "HTTPS://example.com/a/1/?id=7",
        "https://example.com/a/1?utm_source=x&id=7&utm_medium=y",
        "https://example.com/a/1?id=7#comments",
    ]
    assert [normalize_url(v) for v in variants] == [canonical] * len(variants)
    assert normalize_url("https://example.com/a/2?id=7") != canonical
    assert normalize_url("https://example.com/a/1?id=8") != canonical


def test_url_variants_are_stored_once(store):
    store.add([make(1, url="https://example.com/a/1")])
    result = store.add([make(1, url="https://example.com/a/1/?utm_source=feed#top")])
    assert result.inserted == 0


def test_items_without_url_are_keyed_by_source_title_and_time(store):
    assert store.add([make(1, url=None)]).inserted == 1
    assert store.add([make(1, url=None, title="headline  1!")]).inserted == 0
    assert store.add([make(1, url=None, published_at=T - timedelta(hours=2))]).inserted == 1
    assert store.add([make(2, url=None)]).inserted == 1


def test_since_is_an_inclusive_lower_bound(store):
    since = T - timedelta(hours=2)
    store.add(
        [
            make(1, title="too old", published_at=since - timedelta(seconds=1)),
            make(2, title="at since", published_at=since),
            make(3, title="inside", published_at=T - timedelta(hours=1)),
        ]
    )
    assert titles(store.query("NVDA", before=T, since=since)) == ["inside", "at since"]


def test_query_orders_newest_first_and_honours_limit(store):
    store.add([make(n, published_at=T - timedelta(hours=n)) for n in (3, 1, 2)])
    assert titles(store.query("NVDA", before=T)) == ["Headline 1", "Headline 2", "Headline 3"]
    assert titles(store.query("nvda", before=T, limit=2)) == ["Headline 1", "Headline 2"]


def test_other_tickers_are_not_returned(store):
    store.add([make(1, tickers=["AMD"])])
    assert store.query("NVDA", before=T) == []


def test_reopening_keeps_data_and_rejects_unknown_schema(tmp_path):
    path = tmp_path / "sub" / "news.db"
    with NewsStore(path) as s:
        s.add([make(1)])
    with NewsStore(path) as s:
        assert len(s.query("NVDA", before=T)) == 1

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError):
        NewsStore(path)
