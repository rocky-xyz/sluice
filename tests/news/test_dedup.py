from datetime import UTC, datetime, timedelta

import pytest

from sluice.news.models import NewsItem
from sluice.news.store import REPRINT_WINDOW, NewsStore

FETCHED = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
T0 = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
LATER = T0 + timedelta(days=30)
_urls = iter(range(10_000))


def story(hours: float = 0, title: str = "Nvidia unveils new chip", **overrides) -> NewsItem:
    """A copy of one story, published `hours` after T0, at its own URL."""
    fields = {
        "title": title,
        "url": f"https://example.com/{next(_urls)}",
        "source": "yfinance",
        "publisher": "Reuters",
        "tickers": ["NVDA"],
        "published_at": T0 + timedelta(hours=hours),
        "fetched_at": FETCHED,
    }
    fields.update(overrides)
    return NewsItem(**fields)


@pytest.fixture
def store(tmp_path):
    with NewsStore(tmp_path / "news.db") as s:
        yield s


def load(store: NewsStore, *items: NewsItem) -> int:
    store.add(items)
    return store.rebuild_clusters()


def test_reprint_is_returned_once_as_the_earliest_copy(store):
    reprints = load(
        store, story(2, publisher="Yahoo Finance"), story(0), story(5, title="NVIDIA unveils: new chip!")
    )
    assert reprints == 2
    (row,) = store.query("NVDA", before=LATER)
    assert row.published_at == T0
    assert row.publisher == "Reuters"
    assert row.reprint_count == 2


def test_window_is_measured_from_the_first_copy_and_is_inclusive(store):
    edge = REPRINT_WINDOW.total_seconds() / 3600
    load(store, story(0), story(edge), story(edge + 1 / 3600))
    rows = store.query("NVDA", before=LATER)
    assert [r.reprint_count for r in rows] == [0, 1]  # newest first: a new story, then the pair


def test_a_chain_of_copies_does_not_stretch_the_window(store):
    load(store, story(0), story(8), story(16))
    rows = store.query("NVDA", before=LATER)
    assert [(r.published_at - T0, r.reprint_count) for r in rows] == [
        (timedelta(hours=16), 0),
        (timedelta(hours=0), 1),
    ]


def test_identically_titled_daily_columns_stay_separate(store):
    title = "10 Information Technology Stocks Whale Activity In Today's Session"
    assert load(store, *(story(24 * day, title=title) for day in range(3))) == 0
    assert len(store.query("NVDA", before=LATER)) == 3


def test_same_article_from_two_vendors_is_not_a_reprint(store):
    load(store, story(0), story(0, source="finnhub", publisher="Yahoo"))
    (row,) = store.query("NVDA", before=LATER)
    assert row.reprint_count == 0

    load(store, story(1, publisher="Zacks"))
    (row,) = store.query("NVDA", before=LATER)
    assert row.reprint_count == 1  # two copies within yfinance, one within finnhub


def test_reprints_published_after_the_decision_time_are_not_counted(store):
    load(store, story(0), story(2), story(6))
    assert store.query("NVDA", before=T0) == []
    assert [r.reprint_count for r in store.query("NVDA", before=T0 + timedelta(hours=1))] == [0]
    assert [r.reprint_count for r in store.query("NVDA", before=T0 + timedelta(hours=3))] == [1]
    assert [r.reprint_count for r in store.query("NVDA", before=LATER)] == [2]


def test_tickers_of_reprints_join_the_story_only_once_they_are_published(store):
    load(store, story(0), story(4, tickers=["AMD"]))

    assert store.query("AMD", before=T0 + timedelta(hours=2)) == []
    (early,) = store.query("NVDA", before=T0 + timedelta(hours=2))
    assert early.tickers == ["NVDA"]

    (late,) = store.query("AMD", before=LATER)
    assert late.published_at == T0
    assert late.tickers == ["AMD", "NVDA"]


def test_an_older_copy_arriving_later_becomes_the_story(store):
    load(store, story(3, publisher="Zacks"))
    load(store, story(0))  # a backfill brings the original in afterwards
    (row,) = store.query("NVDA", before=LATER)
    assert (row.publisher, row.reprint_count) == ("Reuters", 1)
    assert store.rebuild_clusters() == 1  # repeating changes nothing


def test_since_applies_to_the_first_copy(store):
    load(store, story(0), story(4))
    assert store.query("NVDA", before=LATER, since=T0 + timedelta(hours=1)) == []


def test_articles_without_a_time_are_never_grouped(store):
    missing = {"published_at": None, "time_precision": "missing"}
    assert load(store, story(**missing), story(**missing)) == 0
    assert len(store.query("NVDA", before=LATER, include_invalid=True)) == 2


def test_dedup_can_be_turned_off(store):
    load(store, story(0), story(2))
    rows = store.query("NVDA", before=LATER, dedup=False)
    assert [r.reprint_count for r in rows] == [0, 0]


def test_new_articles_are_visible_before_clusters_are_rebuilt(store):
    store.add([story(0), story(2)])
    assert len(store.query("NVDA", before=LATER)) == 2


def test_stories_carry_their_trading_session(store):
    # T0 is Monday 2026-03-02 09:00 UTC, which is 04:00 in New York.
    load(
        store,
        story(0, title="pre"),
        story(6, title="regular"),
        story(13, title="post"),
        story(20, title="closed"),
        story(title="day only", time_precision="day"),
    )
    rows = store.query("NVDA", before=LATER)
    sessions = {r.title: r.session for r in rows}
    assert sessions == {
        "pre": "pre",
        "regular": "regular",
        "post": "post",
        "closed": "closed",
        "day only": None,
    }


def test_missing_publisher_is_filled_from_a_copy_published_before_the_decision_time(store):
    load(store, story(0, source="finnhub", publisher=None), story(2, publisher="Motley Fool"))

    (early,) = store.query("NVDA", before=T0 + timedelta(hours=1))
    assert early.publisher is None
    (late,) = store.query("NVDA", before=LATER)
    assert (late.source, late.publisher) == ("finnhub", "Motley Fool")


def test_a_known_publisher_is_never_replaced(store):
    load(store, story(0), story(2, publisher="Yahoo Finance"))
    (row,) = store.query("NVDA", before=LATER)
    assert row.publisher == "Reuters"
    assert {r.publisher for r in store.query("NVDA", before=LATER, dedup=False)} == {
        "Reuters",
        "Yahoo Finance",
    }
