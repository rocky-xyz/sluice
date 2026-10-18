from datetime import UTC, date, datetime

import pytest

from scripts.news_label_correlation import _rank, collect_rows, correlations, stance_score
from sluice.news.labels import item_label
from sluice.news.models import NewsItem
from sluice.news.store import NewsStore
from sluice.news.timeutil import ET
from sluice.prices.store import Bar, PriceStore
from sluice.scoring.store import ItemScores, ScoreStore

AFTER_EXIT_CLOSE = datetime(2026, 10, 1, 21, 0, tzinfo=UTC)


def score(**overrides) -> ItemScores:
    fields = {
        "relevance": 0.9,
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


def test_rank_averages_tied_positions():
    assert _rank([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_stance_score_is_bullish_minus_bearish():
    assert stance_score({"bullish": 0.7, "bearish": 0.2, "neutral": 0.1}) == pytest.approx(0.5)


def test_correlations_needs_at_least_two_points():
    rows = [{"relevance": 0.5, "label_3d": 0.1}]
    assert correlations(rows, "relevance", 3) == (1, None, None)


def test_collect_rows_averages_across_a_multi_ticker_item(tmp_path):
    published = datetime(2026, 9, 28, 11, 0, tzinfo=ET)  # Monday, mid-session
    entry_day, exit_day = date(2026, 9, 29), date(2026, 10, 1)

    with (
        NewsStore(tmp_path / "news.db") as news,
        ScoreStore(tmp_path / "scores.db") as scores,
        PriceStore(tmp_path / "prices.db") as prices,
    ):
        news.add(
            [
                NewsItem(
                    title="Chips rally on trade news",
                    source="yfinance",
                    tickers=["NVDA", "AMD"],
                    published_at=published,
                    fetched_at=published,
                )
            ]
        )
        (news_id,) = (item.id for item in news.query("NVDA", before=AFTER_EXIT_CLOSE))
        scores.put(news_id, "v1", "digest", score())

        # NVDA: +10% by the exit day; AMD: +30%; SPY flat. Both legs must resolve to average them.
        for ticker, entry_open, exit_close in [
            ("NVDA", 100.0, 110.0),
            ("AMD", 50.0, 65.0),
            ("SPY", 400.0, 400.0),
        ]:
            prices.replace(
                ticker,
                [
                    # `close` on the entry day only feeds the (untested) 1-day label; any
                    # nonzero value keeps that horizon's log-return from a domain error.
                    Bar(day=entry_day, open=entry_open, close=entry_open, raw_close=entry_open, volume=1.0),
                    Bar(
                        day=exit_day,
                        open=exit_close,
                        close=exit_close,
                        raw_close=exit_close,
                        volume=1.0,
                    ),
                ],
            )

        rows = collect_rows(news, scores, prices, now=AFTER_EXIT_CLOSE)
        # The averaging math itself is covered by tests/news/test_labels.py; here we only
        # check that collect_rows passes the item's own tickers through to it correctly.
        expected = item_label(["AMD", "NVDA"], published, 3, prices, now=AFTER_EXIT_CLOSE)

    assert len(rows) == 1
    row = rows[0]
    assert row["tickers"] == "AMD+NVDA"
    assert row["label_3d"] == pytest.approx(expected)
