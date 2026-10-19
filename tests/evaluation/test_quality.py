import math
from datetime import date

import pytest

from sluice.evaluation.decisions import Decision
from sluice.evaluation.quality import (
    Labelled,
    confidence_calibration,
    consistency,
    rating_calibration,
    summary,
)

DAY = date(2026, 3, 3)


def d(ticker="A", rating="hold", confidence=0.6, repeat=0, day=DAY) -> Decision:
    return Decision(
        ticker=ticker, day=day, repeat=repeat, config="v", run_id=None,
        rating=rating, confidence=None if rating is None else confidence,
    )  # fmt: skip


def row(rating, label, confidence=0.6, ticker="A") -> Labelled:
    return Labelled(d(ticker, rating, confidence), label)


def test_summary_by_hand():
    rows = [
        row("overweight", 0.02),  # hit
        row("underweight", 0.01),  # miss
        row("buy", -0.03),  # miss
        row("hold", 0.05),  # no direction
        row(None, None),  # failed run
        row("sell", None),  # label not known yet
    ]
    m = summary(rows)
    assert (m.runs, m.failed, m.labelled, m.directional) == (6, 1, 4, 3)
    assert m.hit_rate == pytest.approx(1 / 3)
    assert (m.bullish, m.bearish) == (pytest.approx(2 / 5), pytest.approx(2 / 5))


def test_spearman_needs_spread_on_both_sides():
    assert summary([row("hold", 0.01), row("hold", 0.02), row("hold", 0.03)]).spearman is None
    perfect = summary([row("underweight", -0.01), row("hold", 0.0), row("overweight", 0.02)])
    assert perfect.spearman == pytest.approx(1.0)


def test_confidence_buckets_count_only_directional_labelled_runs():
    rows = [
        row("overweight", 0.01, 0.45),  # < 0.5, hit
        row("overweight", -0.01, 0.55),  # 0.5-0.6, miss
        row("underweight", -0.02, 0.58),  # 0.5-0.6, hit
        row("underweight", -0.02, 0.6),  # 0.6-0.7 (edge belongs above), hit
        row("hold", 0.03, 0.9),  # no direction: left out
        row("buy", 0.03, 0.8),  # >= 0.7, hit
    ]
    buckets = {b.name: b for b in confidence_calibration(rows)}
    assert [(b.n, b.hit_rate) for b in buckets.values()] == [(1, 1.0), (2, 0.5), (1, 1.0), (1, 1.0)]
    assert buckets["0.5–0.6"].mean_confidence == pytest.approx(0.565)


def test_rating_calibration_lists_every_level_in_order():
    levels = rating_calibration([row("hold", 0.01), row("hold", 0.03), row("sell", -0.02)])
    assert levels == [
        ("buy", 0, None), ("overweight", 0, None), ("hold", 2, pytest.approx(0.02)),
        ("underweight", 0, None), ("sell", 1, pytest.approx(-0.02)),
    ]  # fmt: skip


def test_consistency_over_cells_whose_repeats_all_succeeded():
    decisions = [
        d("A", "hold", repeat=0), d("A", "hold", repeat=1), d("A", "hold", repeat=2),  # agree, std 0
        d("B", "buy", repeat=0), d("B", "hold", repeat=1), d("B", "hold", repeat=2),  # scores 2,0,0
        d("C", "hold", repeat=0), d("C", None, repeat=1),  # a failed repeat: left out
        d("E", "sell", repeat=0),  # one repeat only: left out
    ]  # fmt: skip
    c = consistency(decisions)
    assert (c.cells, c.agree) == (2, 0.5)
    assert c.mean_std == pytest.approx((0 + math.sqrt(8 / 9)) / 2)  # population sd of 2, 0, 0


def test_a_single_repeat_has_no_consistency():
    assert consistency([d("A"), d("B")]).cells == 0
