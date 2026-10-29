import math
from datetime import date, timedelta

import pytest

from sluice.agents import indicators
from sluice.agents.indicators import (
    atr_ratio,
    bollinger_position,
    bollinger_width,
    compute,
    ema,
    excess_return,
    kdj,
    macd_histogram,
    range_position,
    rsi,
    trailing_return,
    versus_average,
    versus_ema,
    volatility,
    volume_ratio,
)
from sluice.agents.roles import PRICE_LABELS, _price_line
from sluice.agents.state import PriceInput
from sluice.prices.store import Bar
from tests.prices.series import series, with_wicks

LAST = date(2026, 9, 30)


def bars(closes, volumes=None, first=date(2026, 1, 5)) -> list[Bar]:
    volumes = volumes or [1000.0] * len(closes)
    return [
        Bar(day=first + timedelta(days=i), open=close, close=close, raw_close=close, volume=volume)
        for i, (close, volume) in enumerate(zip(closes, volumes, strict=True))
    ]


def candles(rows, first=date(2026, 1, 5)) -> list[Bar]:
    """Bars from (high, low, close) rows."""
    return [
        Bar(
            day=first + timedelta(days=i),
            open=close,
            close=close,
            raw_close=close,
            volume=1000.0,
            high=high,
            low=low,
        )
        for i, (high, low, close) in enumerate(rows)
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


# The indicators of docs/specs/10-configurable-components.md (task 11.2), each checked
# against values worked out by hand on a few bars.


def test_ema_starts_from_the_mean_of_the_first_n_values():
    # n = 3: alpha = 2 / 4 = 0.5; the first average is (1 + 2 + 3) / 3 = 2
    assert ema([1, 2, 3, 4, 5], 3) == pytest.approx([2, 0.5 * 4 + 0.5 * 2, 0.5 * 5 + 0.5 * 3])
    assert ema([1, 2], 3) is None


def test_versus_ema_is_the_gap_to_the_last_average():
    # EMA(3) of 2, 4, 6, 10: (2 + 4 + 6) / 3 = 4, then 0.5 * 10 + 0.5 * 4 = 7
    assert versus_ema([2, 4, 6, 10], 3) == pytest.approx(10 / 7 - 1)
    assert versus_ema([2, 4], 3) is None


def test_macd_histogram_is_dif_minus_its_signal_over_the_close():
    closes = [2, 4, 6, 10, 6, 8]
    # fast EMA(1) is the close itself; slow EMA(3): 4, 7, 6.5, 7.25 from the third close on
    # DIF: 6 - 4 = 2, 10 - 7 = 3, 6 - 6.5 = -0.5, 8 - 7.25 = 0.75
    # DEA = EMA(3) of DIF: (2 + 3 - 0.5) / 3 = 1.5, then 0.5 * 0.75 + 0.5 * 1.5 = 1.125
    assert macd_histogram(closes, 1, 3, 3) == pytest.approx((0.75 - 1.125) / 8)
    assert macd_histogram(closes[:4], 1, 3, 3) is None  # two DIF values, three needed


def test_bollinger_position_and_width():
    # mean 10, population deviation 1: bands 8 and 12
    assert bollinger_position([9, 11, 9, 11], 4, 2) == pytest.approx((11 - 8) / (12 - 8))
    assert bollinger_width([9, 11, 9, 11], 4, 2) == pytest.approx((12 - 8) / 10)
    assert bollinger_position([7, 9, 11, 9, 11], 4, 2) == pytest.approx(0.75)  # last 4 only
    assert bollinger_position([9, 11, 9], 4, 2) is None


def test_a_flat_band_has_no_position_but_a_width_of_zero():
    assert bollinger_position([5, 5, 5, 5], 4, 2) is None  # upper - lower is 0
    assert bollinger_width([5, 5, 5, 5], 4, 2) == 0.0


def test_atr_counts_a_gap_from_the_previous_close():
    history = candles([(10, 8, 9), (12, 9, 11), (15, 14, 14.5), (15, 13, 14)])
    # TR: max(3, 3, 0) = 3; max(1, |15 - 11|, |14 - 11|) = 4 (the gap); max(2, 0.5, 1.5) = 2
    # n = 2: mean of 3 and 4 = 3.5, then (3.5 * 1 + 2) / 2 = 2.75
    assert atr_ratio(history, 2) == pytest.approx(2.75 / 14)
    assert atr_ratio(history[:2], 2) is None  # one TR, two needed


def test_range_position_uses_the_highs_and_lows_of_the_last_n_bars():
    history = candles([(10, 5, 8), (12, 7, 9), (11, 6, 10), (9, 8, 8.5)])
    # the last 3 bars: highest high 12, lowest low 6
    assert range_position(history, 3) == pytest.approx((8.5 - 6) / (12 - 6))
    assert range_position(history, 5) is None
    assert range_position(candles([(5, 5, 5)] * 3), 3) is None  # no range at all


def test_kdj_smooths_rsv_from_fifty():
    history = candles([(10, 8, 9), (11, 8, 10), (12, 9, 11), (12, 10, 10)])
    # n = 3. Day 3: lows 8, highs 12, close 11: RSV 75
    k1 = 2 / 3 * 50 + 1 / 3 * 75
    d1 = 2 / 3 * 50 + 1 / 3 * k1
    # Day 4: lows 8, highs 12, close 10: RSV 50
    k2 = 2 / 3 * k1 + 1 / 3 * 50
    d2 = 2 / 3 * d1 + 1 / 3 * k2
    assert kdj(history, 3) == pytest.approx((k2, d2))
    assert kdj(history[:2], 3) is None


def test_kdj_has_no_value_once_a_window_has_no_range():
    history = candles([(10, 10, 10)] * 3 + [(12, 9, 11)])
    assert kdj(history, 3) is None


def test_indicators_on_highs_and_lows_need_every_bar_to_have_them():
    history = with_wicks(series(LAST, 250))
    history[0] = history[0].model_copy(update={"high": None, "low": None})
    assert atr_ratio(history, 14) is None
    assert kdj(history, 9) is None
    assert range_position(history, 250) is None
    assert range_position(history, 249) is not None  # the bar without them is not used


def test_by_default_only_the_original_eleven_are_computed():
    values = compute(with_wicks(series(LAST, 250)), series(LAST, 250))
    assert tuple(values) == indicators.NAMES


def test_a_long_history_with_wicks_gives_every_indicator():
    values = compute(with_wicks(series(LAST, 250)), series(LAST, 250), indicators.ALL)
    assert tuple(values) == indicators.ALL
    assert 0 <= values["kdj_k"] <= 100 and 0 <= values["kdj_d"] <= 100
    assert 0 <= values["range_250d"] <= 1
    assert values["bb_width_20"] > 0 and values["atr_14_pct"] > 0


def test_indicators_come_back_in_the_registry_order_whatever_was_asked():
    history = with_wicks(series(LAST, 250))
    values = compute(history, None, ["kdj_d", "rsi_14", "bb_pctb_20"])
    assert list(values) == ["rsi_14", "bb_pctb_20", "kdj_d"]


def test_an_unknown_indicator_is_an_error():
    with pytest.raises(ValueError, match="sma_7"):
        compute(series(LAST, 30), None, ["rsi_14", "sma_7"])


def test_without_highs_and_lows_those_indicators_are_left_out():
    values = compute(series(LAST, 250), None, indicators.ALL)
    assert not {"atr_14_pct", "range_250d", "kdj_k", "kdj_d"} & set(values)
    assert "macd_hist_pct" in values and "bb_pctb_20" in values


def test_every_indicator_has_a_label_and_a_format_without_price_levels():
    assert set(PRICE_LABELS) == set(indicators.ALL)
    shown = {
        name: _price_line(PriceInput(ref=f"P:{name}", name=name, value=value))
        for name, value in [
            ("macd_hist_pct", -0.0042),
            ("bb_pctb_20", 0.834),
            ("bb_width_20", 0.0812),
            ("atr_14_pct", 0.0231),
            ("vs_ema10", 0.0125),
            ("range_250d", 0.4167),
            ("kdj_k", 61.234),
        ]
    }
    assert shown["macd_hist_pct"].endswith(": -0.42%")
    assert shown["bb_pctb_20"].endswith(": 0.83")
    assert shown["bb_width_20"].endswith(": 8.12%")
    assert shown["atr_14_pct"].endswith(": 2.31%")
    assert shown["vs_ema10"].endswith(": +1.25%")
    assert shown["range_250d"].endswith(": 0.42")
    assert shown["kdj_k"] == "[P:kdj_k] KDJ K line (9, 3, 3; 0 to 100): 61.2"
