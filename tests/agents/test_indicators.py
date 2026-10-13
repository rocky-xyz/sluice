import math
from datetime import date, timedelta

import pytest

from sluice.agents import indicators
from sluice.agents.indicators import (
    compute,
    excess_return,
    rsi,
    trailing_return,
    versus_average,
    volatility,
    volume_ratio,
)
from sluice.prices.store import Bar
from tests.prices.series import series

LAST = date(2026, 9, 30)


def bars(closes, volumes=None, first=date(2026, 1, 5)) -> list[Bar]:
    volumes = volumes or [1000.0] * len(closes)
    return [
        Bar(day=first + timedelta(days=i), close=close, raw_close=close, volume=volume)
        for i, (close, volume) in enumerate(zip(closes, volumes, strict=True))
    ]


def test_trailing_return_compares_with_the_close_n_days_back():
    assert trailing_return([100, 101, 102, 110], 3) == pytest.approx(0.10)
    assert trailing_return([100, 101, 102, 110], 1) == pytest.approx(110 / 102 - 1)
    assert trailing_return([100, 101, 102], 3) is None  # needs n + 1 closes


def test_excess_return_subtracts_the_benchmark_over_the_same_days():
    stock = bars([100, 105, 110])
    index = bars([200, 202, 206])
    assert excess_return(stock, index, 2) == pytest.approx(0.10 - 0.03)
    assert excess_return(stock, index[1:], 2) is None  # the benchmark lacks the first day
    assert excess_return(stock, index, 3) is None


def test_volatility_is_the_annualised_sample_deviation_of_daily_returns():
    # daily returns +10% and -10%: mean 0, sample variance (0.01 + 0.01) / 1
    assert volatility([100, 110, 99], 2) == pytest.approx(math.sqrt(0.02) * math.sqrt(252))
    assert volatility([100, 100, 110, 121], 2) == pytest.approx(0.0, abs=1e-9)  # +10% twice
    assert volatility([100, 110], 2) is None


def test_versus_average_is_the_gap_to_the_mean_of_the_last_n_closes():
    assert versus_average([50, 90, 100, 110], 3) == pytest.approx(110 / 100 - 1)
    assert versus_average([100, 110], 3) is None


def test_rsi_starts_from_plain_means_then_smooths():
    # changes +2, -1, then +3: means over the first two are gain 1.0, loss 0.5;
    # smoothed with n = 2: gain (1.0 + 3) / 2 = 2.0, loss (0.5 + 0) / 2 = 0.25
    assert rsi([10, 12, 11], 2) == pytest.approx(100 - 100 / (1 + 1.0 / 0.5))
    assert rsi([10, 12, 11, 14], 2) == pytest.approx(100 - 100 / (1 + 2.0 / 0.25))


def test_rsi_at_the_extremes():
    assert rsi([1, 2, 3, 4], 3) == 100.0
    assert rsi([4, 3, 2, 1], 3) == 0.0
    assert rsi([5, 5, 5, 5], 3) == 50.0
    assert rsi([1, 2, 3], 3) is None


def test_volume_ratio_leaves_today_out_of_the_average():
    assert volume_ratio([100, 300, 400], 2) == pytest.approx(400 / 200)
    assert volume_ratio([100, 300], 2) is None
    assert volume_ratio([0, 0, 400], 2) is None


def test_a_long_history_gives_every_indicator_and_only_ratios():
    values = compute(series(LAST, 250, drift=0.002), series(LAST, 250, drift=0.001))
    assert tuple(values) == indicators.NAMES
    assert 0 <= values["rsi_14"] <= 100
    assert values["excess_60d"] > 0  # drifts up faster than the benchmark


def test_a_short_history_gives_only_what_can_be_computed():
    values = compute(series(LAST, 30), series(LAST, 30))
    assert set(values) == {
        "ret_5d", "ret_20d", "excess_5d", "excess_20d", "vol_20d", "rsi_14", "volume_ratio_20d",
    }  # fmt: skip
    assert compute(series(LAST, 1), None) == {}


def test_without_a_benchmark_there_are_no_excess_returns():
    values = compute(series(LAST, 250), None)
    assert not any(name.startswith("excess_") for name in values)
    assert "ret_60d" in values
