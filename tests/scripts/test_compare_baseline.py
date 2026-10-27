import math
from datetime import date, timedelta

import pytest

from scripts.compare_baseline import (
    LOOKBACK,
    Row,
    agreement,
    metrics,
    past_excess,
    wilson,
)
from sluice.prices.store import Bar, PriceStore

DAY = date(2026, 3, 31)


def row(ticker="NVDA", day=DAY, rating="overweight", label=0.01, past=0.02, repeat=0) -> Row:
    return Row(ticker, day, repeat, rating, label, past)


def test_wilson_matches_the_textbook_case_and_stays_in_bounds():
    low, high = wilson(5, 10)
    assert (low, high) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))
    low, high = wilson(0, 3)
    assert low == pytest.approx(0.0, abs=1e-12) and 0 < high < 1
    assert wilson(0, 0) is None


def test_hit_rate_leaves_out_hold_failures_and_unknown_labels():
    m = metrics(
        [
            row(rating="overweight", label=0.02),  # hit
            row(rating="underweight", label=0.03),  # miss
            row(rating="hold", label=-0.05),  # no direction
            row(rating=None, label=0.01),  # failed run
            row(rating="sell", label=None),  # label not known yet
        ]
    )
    assert (m.runs, m.failed, m.directional, m.hits, m.hit_rate) == (5, 1, 2, 1, 0.5)
    assert (m.bullish, m.hold, m.bearish) == (0.25, 0.25, 0.5)


def test_daily_spearman_is_taken_within_each_day_then_averaged():
    d1, d2 = DAY, DAY + timedelta(days=14)
    rows = [
        row("A", d1, "underweight", -0.02),
        row("B", d1, "hold", 0.0),
        row("C", d1, "overweight", 0.03),  # day 1: perfectly ordered, +1
        row("A", d2, "overweight", -0.02),
        row("B", d2, "hold", 0.0),
        row("C", d2, "underweight", 0.03),  # day 2: perfectly reversed, -1
        row("D", d2, "hold", None),  # no label: left out
    ]
    m = metrics(rows)
    assert m.days == 2
    assert m.daily_spearman == pytest.approx(0.0)


def test_past_spearman_shows_ratings_that_follow_the_recent_move():
    rows = [
        row("A", rating="underweight", past=-0.06),
        row("B", rating="hold", past=0.01),
        row("C", rating="overweight", past=0.05),
    ]
    assert metrics(rows).past_spearman == pytest.approx(1.0)


def test_agreement_counts_cells_where_both_runs_succeeded():
    ta = [row("A", rating="hold"), row("B", rating="overweight"), row("C", rating=None)]
    v1 = [row("A", rating="hold"), row("B", rating="hold"), row("C", rating="hold")]
    assert agreement(ta, v1) == (2, 0.5)
    assert agreement([], v1) == (0, None)


def _bars(closes: list[float], end: date) -> list[Bar]:
    days = [end - timedelta(days=len(closes) - 1 - i) for i in range(len(closes))]
    return [Bar(day=d, open=c, close=c, raw_close=c, volume=1) for d, c in zip(days, closes)]


def test_past_excess_is_the_stock_log_return_over_spy_to_the_decision_close(tmp_path):
    with PriceStore(tmp_path / "p.db") as prices:
        prices.replace("NVDA", _bars([100] * LOOKBACK + [110], DAY))
        prices.replace("SPY", _bars([200] * LOOKBACK + [202], DAY))
        assert past_excess(prices, "NVDA", DAY) == pytest.approx(math.log(1.1) - math.log(1.01))
        # Too little history, or no bar on the decision day itself: no value.
        assert past_excess(prices, "NVDA", DAY - timedelta(days=LOOKBACK)) is None
        assert past_excess(prices, "NVDA", DAY + timedelta(days=1)) is None
