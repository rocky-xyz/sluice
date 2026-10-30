import json
import math
import statistics
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from scripts.compare_baseline import past_excess
from scripts.momentum_baseline import (
    MAIN,
    RESAMPLES,
    WINDOWS,
    Cell,
    Counts,
    _draws,
    _resampled,
    analyse,
    hits,
    label,
    load_cells,
    matched_tiers,
    partial,
    past_of,
    rank_correlation,
    score,
    unit_mean,
    value,
)
from sluice.evaluation import quality
from sluice.market_calendar import close_of
from sluice.prices.store import Bar, PriceStore

D1, D2 = date(2026, 3, 31), date(2026, 4, 14)
COUNTS = Counts(runs=0, failed=0, unlabelled=0, no_past=0)


def cell(ticker, s, y, p, day=D1, repeat=0) -> Cell:
    return Cell(ticker, day, repeat, s, y, dict.fromkeys(WINDOWS, p))


def test_correlations_are_taken_within_each_unit_then_averaged():
    cells = [
        cell("A", -1, -0.02, -0.05),
        cell("B", 0, 0.00, 0.01),
        cell("C", 1, 0.03, 0.04),  # unit 1: rating, label and past all in order, +1
        cell("A", 1, -0.02, -0.05, D2),
        cell("B", 0, 0.00, 0.01, D2),
        cell("C", -1, 0.03, 0.04, D2),  # unit 2: rating reversed against both, -1
    ]
    forward = unit_mean(cells, rank_correlation(score, label))
    assert forward == {D1: (1.0, 1), D2: (-1.0, 1)}
    assert value(forward) == pytest.approx(0.0)
    restated = unit_mean(cells, rank_correlation(score, past_of(MAIN)))
    assert value(restated) == pytest.approx(0.0)
    assert value(unit_mean(cells, rank_correlation(past_of(MAIN), label))) == pytest.approx(1.0)


def test_a_unit_without_spread_is_left_out_not_counted_as_zero():
    cells = [
        cell("A", -1, -0.02, -0.05),
        cell("B", 0, 0.00, 0.01),
        cell("C", 1, 0.03, 0.04),
        cell("A", 0, -0.02, -0.05, D2),
        cell("B", 0, 0.00, 0.01, D2),
        cell("C", 0, 0.03, 0.04, D2),  # all hold: no rank correlation
        cell("A", 1, 0.01, 0.02, D2, repeat=1),
        cell("B", -1, 0.02, 0.03, D2, repeat=1),  # two stocks: too few
    ]
    parts = unit_mean(cells, rank_correlation(score, label))
    assert parts == {D1: (1.0, 1), D2: (0.0, 0)}
    assert value(parts) == 1.0


def test_the_partial_correlation_matches_a_hand_computed_case():
    # Ranks: past 1 2 3 4, rating 1 3 2 4, label 2 1 4 3. Residuals on past:
    # rating (-0.3, 0.9, -0.9, 0.3), label (0.4, -1.2, 1.2, -0.4): correlation -1.
    group = [
        cell("A", -2, 0.00, 0.01),
        cell("B", 0, -0.01, 0.02),
        cell("C", -1, 0.02, 0.03),
        cell("D", 2, 0.01, 0.04),
    ]
    assert partial(MAIN)(group) == pytest.approx(-1.0)


def test_the_partial_correlation_is_the_textbook_one_on_ranks():
    scores = [2, -1, 0, 1, -2, 0]
    labels = [0.03, -0.01, 0.02, -0.04, 0.01, 0.05]
    pasts = [0.05, -0.02, 0.01, 0.03, -0.06, 0.00]
    group = [cell(t, s, y, p) for t, s, y, p in zip("ABCDEF", scores, labels, pasts, strict=True)]
    rs, ry, rp = (quality._rank(v) for v in (scores, labels, pasts))
    r_sy, r_sp, r_yp = (statistics.correlation(a, b) for a, b in ((rs, ry), (rs, rp), (ry, rp)))
    textbook = (r_sy - r_sp * r_yp) / math.sqrt((1 - r_sp**2) * (1 - r_yp**2))
    assert partial(MAIN)(group) == pytest.approx(textbook)


def test_nothing_is_left_when_the_past_explains_the_rating():
    group = [cell("A", -1, 0.01, -0.03), cell("B", 0, 0.02, 0.0), cell("C", 1, -0.01, 0.03)]
    assert partial(MAIN)(group) is None
    flat = [cell("A", -1, 0.01, 0.0), cell("B", 0, 0.02, 0.0), cell("C", 1, -0.01, 0.0)]
    assert partial(MAIN)(flat) is None  # the past does not vary


def test_the_baseline_takes_the_ratings_shares_from_the_ends_of_the_past():
    cells = [
        cell("A", 1, 0.02, -0.04),
        cell("B", 0, 0.01, 0.05),
        cell("C", -1, -0.03, 0.00),
        cell("D", -1, 0.01, 0.02),
        cell("E", 0, -0.02, 0.02),  # ties D on the past: D, the earlier ticker, ranks higher
    ]
    tiers = matched_tiers(cells, MAIN)
    assert [tiers[c] for c in cells] == [-1, 1, -1, 0, 0]
    assert sum(t > 0 for t in tiers.values()) == sum(c.score > 0 for c in cells)
    assert sum(t < 0 for t in tiers.values()) == sum(c.score < 0 for c in cells)
    # Baseline: B bullish on +0.01 (hit); A bearish on +0.02 (miss); C bearish on -0.03 (hit).
    assert value(hits(cells, tiers.__getitem__)) == pytest.approx(2 / 3)
    # Ratings: A bullish on +0.02 (hit); C bearish (hit); D bearish on +0.01 (miss).
    assert value(hits(cells, lambda c: c.score)) == pytest.approx(2 / 3)


def test_ties_in_the_past_are_settled_the_same_whatever_the_input_order():
    cells = [cell(t, s, 0.01, 0.0) for t, s in zip("ABCD", (1, 0, -1, 0), strict=True)]
    assert matched_tiers(cells, MAIN) == matched_tiers(list(reversed(cells)), MAIN)
    assert [matched_tiers(cells, MAIN)[c] for c in cells] == [1, 0, 0, -1]


def test_a_resample_takes_whole_days():
    parts = {D1: (2.0, 2), D2: (0.0, 1)}
    picks = np.array([[0, 0], [0, 1], [1, 1]])
    assert _resampled(parts, [D1, D2], picks) == pytest.approx([1.0, 2 / 3, 0.0])
    assert _draws([D1, D2], "x").shape == (RESAMPLES, 2)


def _grid() -> list[Cell]:
    rng = np.random.default_rng(0)
    cells = []
    for k in range(6):
        day = D1 + timedelta(days=7 * k)
        for repeat in range(2):
            for t in "ABCDE":
                p = float(rng.normal())
                s = int(np.clip(round(p + rng.normal()), -2, 2))
                cells.append(
                    Cell(t, day, repeat, s, float(rng.normal()), dict.fromkeys(WINDOWS, p))
                )
    return cells


def test_the_two_sides_of_a_difference_share_their_draws():
    cells = _grid()
    days = sorted({c.day for c in cells})
    r = analyse("cfg", cells, COUNTS)
    f = r.forward[MAIN]
    picks = _draws(days, f"cfg|forward|{MAIN}")
    llm = _resampled(unit_mean(cells, rank_correlation(score, label)), days, picks)
    base = _resampled(unit_mean(cells, rank_correlation(past_of(MAIN), label)), days, picks)
    gaps = llm - base
    assert (f.difference.low, f.difference.high) == pytest.approx(
        (np.percentile(gaps, 2.5), np.percentile(gaps, 97.5))
    )
    assert f.difference.value == pytest.approx(f.llm.value - f.baseline.value)
    assert f.llm.low <= f.llm.value <= f.llm.high


def test_the_same_input_gives_the_same_result():
    assert analyse("cfg", _grid(), COUNTS) == analyse("cfg", _grid(), COUNTS)
    point = analyse("cfg", _grid(), COUNTS, intervals=False)
    assert point.restatement.low is None and point.forward[MAIN].difference.high is None


def _sessions(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)
            if close_of(first + timedelta(days=i)) is not None]  # fmt: skip


def _store(tmp_path, through: date) -> PriceStore:
    prices = PriceStore(tmp_path / "p.db")
    days = _sessions(date(2026, 2, 1), through)
    for ticker, step in (("NVDA", 1.002), ("AMD", 0.999), ("SPY", 1.001)):
        closes = [100 * step**i for i in range(len(days))]
        bars = [Bar(day=d, open=c, close=c, raw_close=c, volume=1) for d, c in zip(days, closes)]
        prices.replace(ticker, bars)
    return prices


def _state(folder, ticker: str, day: date, repeat: int, rating: str | None) -> None:
    path = folder / ticker / f"{day}_r{repeat}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = None if rating is None else {"rating": rating, "confidence": 0.6}
    path.write_text(json.dumps({"decision": decision, "meta": {"run_id": "x"}}))


def test_cells_are_read_from_validation_runs_with_label_and_past(tmp_path):
    folder = tmp_path / "v1-test"
    late = date(2026, 5, 20)  # its label needs prices the store does not have
    _state(folder, "NVDA", D1, 0, "overweight")
    _state(folder, "NVDA", D1, 1, None)  # failed
    _state(folder, "AMD", D1, 0, "underweight")  # no past (below)
    _state(folder, "NVDA", late, 0, "hold")
    sealed = folder / "NVDA" / "2026-09-15_r0.json"
    sealed.write_text("{never opened")  # a test_1 day: refused before reading
    with _store(tmp_path, date(2026, 5, 29)) as prices:

        def past(ticker, day, n):
            return None if ticker == "AMD" else past_excess(prices, ticker, day, n)

        cells, counts = load_cells(folder, prices, datetime(2027, 1, 1, tzinfo=UTC), past)
    assert counts == Counts(runs=4, failed=1, unlabelled=1, no_past=1)
    [c] = cells
    assert (c.ticker, c.day, c.repeat, c.score) == ("NVDA", D1, 0, 1)
    # Open of the 1st trading day to the close of the 10th: 9 daily steps here.
    assert c.label == pytest.approx(9 * (math.log(1.002) - math.log(1.001)))
    assert c.past == pytest.approx({n: n * (math.log(1.002) - math.log(1.001)) for n in WINDOWS})


def test_the_past_does_not_change_when_later_bars_are_added(tmp_path):
    with _store(tmp_path / "a", D1) as short, _store(tmp_path / "b", date(2026, 5, 29)) as long:
        for n in WINDOWS:
            assert past_excess(short, "NVDA", D1, n) == past_excess(long, "NVDA", D1, n)
