import math
from pathlib import Path

import pytest

from sluice.sizing.config import Coefficients, load_sizing_config
from sluice.sizing.formula import Inputs, cap, combined, target_weights, zscores

CONFIG = load_sizing_config("v1", Path(__file__).parents[2] / "configs")
EVEN = Coefficients(rating=1.0, reports=1.0, news=1.0)
VOLS = {"A": 0.2, "B": 0.4, "C": 0.2, "D": 0.4}


def inputs(rating=None, reports=None, news=None) -> Inputs:
    blank = dict.fromkeys(VOLS)
    return Inputs(rating=rating or blank, reports=reports or blank, news=news or blank)


def test_zscores_by_hand_with_missing_values_at_zero():
    z = zscores({"A": 1.0, "B": 3.0, "C": None})
    assert z == pytest.approx({"A": -1.0, "B": 1.0, "C": 0.0})  # mean 2, population sd 1


def test_no_spread_or_too_few_values_means_all_zero():
    assert zscores({"A": 0.0, "B": 0.0}) == {"A": 0.0, "B": 0.0}
    assert zscores({"A": 5.0, "B": None}) == {"A": 0.0, "B": 0.0}


def test_a_common_offset_does_not_matter():
    # the news signal leans negative for every stock (task 5.5); only differences count
    assert zscores({"A": -0.004, "B": -0.002}) == pytest.approx(zscores({"A": 0.001, "B": 0.003}))


def test_combined_is_the_coefficient_weighted_mean_of_z():
    x = inputs(rating={"A": 2, "B": 0, "C": 0, "D": -2}, news={"A": 1, "B": 1, "C": 1, "D": 1})
    s = combined(x, Coefficients(rating=2.0, reports=1.0, news=1.0))
    z = zscores(x.rating)
    assert s == pytest.approx({t: 2 * z[t] / 4 for t in VOLS})


def test_without_any_signal_the_result_is_vol_scaled_equal_weight_times_exposure():
    w = target_weights(inputs(), VOLS, 0.8, CONFIG.k, EVEN, max_weight=1.0)
    inverse = {t: 1 / v for t, v in VOLS.items()}
    total = sum(inverse.values())
    assert w == pytest.approx({t: 0.8 * inverse[t] / total for t in VOLS})
    assert sum(w.values()) == pytest.approx(0.8)  # the rest is cash


def test_the_tilt_follows_one_plus_k_s_and_never_goes_short():
    x = inputs(rating={"A": 2, "B": 2, "C": -2, "D": -2})  # z: +1, +1, -1, -1
    flat, only_rating = dict.fromkeys(VOLS, 0.3), Coefficients(rating=1, reports=0, news=0)
    w = target_weights(x, flat, 1.0, 0.5, only_rating, 1.0)
    assert w["A"] / w["C"] == pytest.approx(1.5 / 0.5)
    strong = target_weights(x, flat, 1.0, 2.0, only_rating, 1.0)
    assert strong["C"] == 0.0 and strong["D"] == 0.0  # 1 + 2·(-1) < 0 → no position, not short


def test_cap_clips_and_shares_the_excess_in_proportion():
    capped = cap({"A": 0.5, "B": 0.3, "C": 0.2}, 0.4)
    assert capped == pytest.approx({"A": 0.4, "B": 0.3 + 0.1 * 0.6, "C": 0.2 + 0.1 * 0.4})


def test_cap_repeats_until_nothing_is_over():
    capped = cap({"A": 0.6, "B": 0.3, "C": 0.05, "D": 0.05}, 0.3)
    assert max(capped.values()) <= 0.3 + 1e-12
    assert math.fsum(capped.values()) == pytest.approx(1.0)


def test_an_impossible_cap_is_refused():
    with pytest.raises(ValueError):
        cap({"A": 0.5, "B": 0.5}, 0.4)


def test_the_order_of_tickers_changes_nothing_bitwise():
    x = inputs(
        rating={"A": 2, "B": -1, "C": 0, "D": 1},
        news={"A": -0.003, "B": 0.001, "C": None, "D": 0.002},
    )
    forward = target_weights(x, VOLS, 0.9, CONFIG.k, EVEN, 0.4)
    shuffled = Inputs(
        rating={t: x.rating[t] for t in ("D", "B", "A", "C")},
        reports={t: x.reports[t] for t in ("C", "A", "D", "B")},
        news={t: x.news[t] for t in ("B", "D", "C", "A")},
    )
    vols = {t: VOLS[t] for t in ("C", "D", "A", "B")}
    backward = target_weights(shuffled, vols, 0.9, CONFIG.k, EVEN, 0.4)
    assert forward == backward


def test_without_vol_scaling_and_tilt_it_is_plain_equal_weight_times_exposure():
    w = target_weights(inputs(), VOLS, 0.6, 0.0, EVEN, 1.0, scale_by_vol=False)
    assert w == pytest.approx(dict.fromkeys(VOLS, 0.15))
