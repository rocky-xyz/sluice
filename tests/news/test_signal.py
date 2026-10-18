from datetime import UTC, date, datetime

import pytest

from sluice.news.models import NewsItem
from sluice.news.signal import (
    SignalItem,
    group_by_ticker_day,
    item_signal,
    record,
    ticker_day_signal,
    ticker_day_signals,
)
from sluice.news.store import NewsStore
from sluice.news.timeutil import ET
from sluice.news.weights import WeightEstimate, WeightTable
from sluice.scoring.store import ItemScores, ScoreStore


def _table(source: dict[str, float], event_type: dict[str, float]) -> WeightTable:
    return WeightTable(
        lambda0=1.0,
        source={s: WeightEstimate(weight=w, low=w, high=w, n=100) for s, w in source.items()},
        event_type={t: WeightEstimate(weight=w, low=w, high=w, n=100) for t, w in event_type.items()},
    )


def test_item_signal_multiplies_raw_by_the_combined_weight():
    table = _table({"Reuters": 0.02}, {"earnings": 0.01})
    assert item_signal(0.5, "Reuters", "earnings", table) == pytest.approx(0.5 * (0.02 + 0.01))


def test_item_signal_treats_an_unseen_group_as_zero_weight():
    table = _table({"Reuters": 0.02}, {"earnings": 0.01})
    assert item_signal(0.5, "Unknown Blog", "earnings", table) == pytest.approx(0.5 * 0.01)
    assert item_signal(0.5, "Reuters", "unseen_type", table) == pytest.approx(0.5 * 0.02)


def test_ticker_day_signal_is_relevance_weighted():
    table = _table({"A": 1.0, "B": 1.0}, {"t": 0.0})
    items = [
        SignalItem(raw=1.0, source="A", event_type="t", relevance=0.9),  # ŷ=1.0
        SignalItem(raw=-1.0, source="B", event_type="t", relevance=0.1),  # ŷ=-1.0
    ]
    # (0.9*1.0 + 0.1*-1.0) / (0.9+0.1) = 0.8
    assert ticker_day_signal(items, table) == pytest.approx(0.8)


def test_ticker_day_signal_is_none_when_every_relevance_is_zero():
    table = _table({"A": 1.0}, {"t": 0.0})
    items = [SignalItem(raw=1.0, source="A", event_type="t", relevance=0.0)]
    assert ticker_day_signal(items, table) is None


def _score(**overrides) -> ItemScores:
    fields = {
        "relevance": 0.8,
        "stance": "bullish",
        "stance_confidence": 0.8,
        "stance_probabilities": {"bullish": 0.7, "bearish": 0.1, "neutral": 0.2},
        "novelty": 0.5,
        "rumour": 0.1,
        "event_type": "earnings",
        "event_type_confidence": 0.8,
        "event_type_probabilities": {"earnings": 0.8},
    }
    fields.update(overrides)
    return ItemScores(**fields)


def test_group_by_ticker_day_buckets_by_entry_day_not_calendar_date(tmp_path):
    # Friday 17:00 ET is after the close; its "next open" lands on Monday, the
    # same trading day pre-market Monday news about the same stock would get.
    friday_after_close = datetime(2026, 1, 2, 17, 0, tzinfo=ET)
    monday_premarket = datetime(2026, 1, 5, 7, 0, tzinfo=ET)
    cutoff = datetime(2026, 1, 6, tzinfo=UTC)

    with NewsStore(tmp_path / "news.db") as news, ScoreStore(tmp_path / "scores.db") as scores:
        news.add(
            [
                NewsItem(
                    title="Friday close",
                    source="yfinance",
                    tickers=["NVDA"],
                    published_at=friday_after_close,
                    fetched_at=friday_after_close,
                ),
                NewsItem(
                    title="Monday premarket",
                    source="yfinance",
                    tickers=["NVDA"],
                    published_at=monday_premarket,
                    fetched_at=monday_premarket,
                ),
            ]
        )
        for item in news.query("NVDA", before=cutoff):
            scores.put(item.id, "v1", "digest", _score())

        groups = group_by_ticker_day(news, scores, ["NVDA"], as_of=cutoff)

    assert list(groups.keys()) == [("NVDA", date(2026, 1, 5))]
    assert len(groups[("NVDA", date(2026, 1, 5))]) == 2


def test_group_by_ticker_day_duplicates_a_multi_ticker_item_per_ticker(tmp_path):
    published = datetime(2026, 1, 5, 7, 0, tzinfo=ET)  # pre-market: entry day is the same day
    cutoff = datetime(2026, 1, 6, tzinfo=UTC)

    with NewsStore(tmp_path / "news.db") as news, ScoreStore(tmp_path / "scores.db") as scores:
        news.add(
            [
                NewsItem(
                    title="Chips rally",
                    source="yfinance",
                    tickers=["NVDA", "AMD"],
                    published_at=published,
                    fetched_at=published,
                )
            ]
        )
        for item in news.query("NVDA", before=cutoff):
            scores.put(item.id, "v1", "digest", _score())

        groups = group_by_ticker_day(news, scores, ["NVDA", "AMD"], as_of=cutoff)

    assert set(groups.keys()) == {("NVDA", date(2026, 1, 5)), ("AMD", date(2026, 1, 5))}
    # Same underlying item, contributing independently to each ticker's own bucket.
    assert groups[("NVDA", date(2026, 1, 5))][0] == groups[("AMD", date(2026, 1, 5))][0]


def test_ticker_day_signals_combines_grouping_and_aggregation(tmp_path):
    published = datetime(2026, 1, 5, 7, 0, tzinfo=ET)  # pre-market: entry day is the same day
    cutoff = datetime(2026, 1, 6, tzinfo=UTC)
    table = _table({"yfinance": 0.0}, {"earnings": 0.5})

    with NewsStore(tmp_path / "news.db") as news, ScoreStore(tmp_path / "scores.db") as scores:
        news.add(
            [
                NewsItem(
                    title="Earnings beat",
                    source="yfinance",
                    tickers=["NVDA"],
                    published_at=published,
                    fetched_at=published,
                )
            ]
        )
        for item in news.query("NVDA", before=cutoff):
            scores.put(item.id, "v1", "digest", _score(stance_probabilities={"bullish": 0.9, "bearish": 0.0, "neutral": 0.1}))

        signals = ticker_day_signals(news, scores, ["NVDA"], table, as_of=cutoff)

    # raw = 0.9 - 0.0 = 0.9; combined weight = 0.0 + 0.5; only one item, so the
    # relevance-weighted average is just its own ŷ.
    assert signals == {("NVDA", date(2026, 1, 5)): pytest.approx(0.9 * 0.5)}


def test_record_is_confirmed_or_reversed_only_when_the_whole_interval_is_on_one_side():
    assert record(WeightEstimate(weight=0.01, low=0.002, high=0.02, n=50)) == "confirmed"
    assert record(WeightEstimate(weight=-0.01, low=-0.02, high=-0.002, n=50)) == "reversed"
    assert record(WeightEstimate(weight=0.01, low=-0.001, high=0.02, n=50)) == "no reliable record"
    assert record(WeightEstimate(weight=0.0, low=0.0, high=0.0, n=50)) == "no reliable record"


def test_a_source_the_table_never_saw_has_no_record():
    assert record(None) == "no record"
