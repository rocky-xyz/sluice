import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from sluice.market_calendar import close_of
from sluice.prices.store import Bar, PriceStore
from sluice.sizing.config import load_sizing_config
from sluice.sizing.regime import exposure, realised_vol
from tests.prices.series import trading_days

DAY = date(2026, 9, 30)
CLOSE = close_of(DAY)
CONFIG = load_sizing_config("v1", Path(__file__).parents[2] / "configs").exposure


def store_with(tmp_path, closes: list[float], last: date = DAY) -> PriceStore:
    """SPY with these closes, the last on `last`."""
    store = PriceStore(tmp_path / "prices.db")
    days = trading_days(last, len(closes))
    bars = [Bar(day=d, open=c, close=c, raw_close=c, volume=1.0) for d, c in zip(days, closes, strict=True)]
    store.replace("SPY", bars)
    return store


def alternating(daily: float, count: int) -> list[float]:
    """Closes whose log returns alternate +daily, -daily."""
    closes, value = [], 100.0
    for i in range(count):
        closes.append(value)
        value *= math.exp(daily if i % 2 == 0 else -daily)
    return closes


def test_realised_vol_by_hand():
    closes = [100.0, 101.0, 100.0, 102.0]
    returns = [math.log(101 / 100), math.log(100 / 101), math.log(102 / 100)]
    mean = sum(returns) / 3
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / 2)
    assert realised_vol(closes) == pytest.approx(sd * math.sqrt(252))


def test_calm_markets_are_fully_invested(tmp_path):
    result = exposure(store_with(tmp_path, alternating(0.002, 30)), CLOSE, CONFIG)
    assert result.vol < CONFIG.target_vol
    assert result.value == 1.0
    assert result.below_trend is None  # the filter is off in v1


def test_volatile_markets_scale_down_to_target_over_realised(tmp_path):
    result = exposure(store_with(tmp_path, alternating(0.012, 30)), CLOSE, CONFIG)
    assert CONFIG.target_vol / result.vol > CONFIG.floor
    assert result.value == pytest.approx(CONFIG.target_vol / result.vol)


def test_exposure_never_falls_below_the_floor(tmp_path):
    result = exposure(store_with(tmp_path, alternating(0.05, 30)), CLOSE, CONFIG)
    assert result.value == CONFIG.floor


def test_only_the_last_vol_lookback_returns_count(tmp_path):
    calm_then_wild = alternating(0.05, 40)[:20] + alternating(0.002, 21)
    result = exposure(store_with(tmp_path, calm_then_wild), CLOSE, CONFIG)
    assert result.value == 1.0


def test_prices_after_the_decision_time_change_nothing(tmp_path):
    closes = alternating(0.012, 30)
    before = exposure(store_with(tmp_path, closes), CLOSE, CONFIG)
    later = closes + [closes[-1] * 0.5, closes[-1] * 2]  # a crash and a surge after the decision day
    (tmp_path / "prices.db").unlink()
    after_day = trading_days(DAY + timedelta(days=10), 3)[-1]
    shifted = store_with(tmp_path, later, last=after_day)
    assert exposure(shifted, CLOSE, CONFIG) == before


def test_during_trading_hours_the_open_session_is_not_used(tmp_path):
    closes = alternating(0.012, 30)
    store = store_with(tmp_path, closes)
    noon = datetime(2026, 9, 30, 16, 0, tzinfo=UTC)  # 12:00 Eastern on DAY
    yesterday = exposure(store, noon, CONFIG)
    assert yesterday.vol == pytest.approx(realised_vol(closes[-22:-1]))


def test_the_trend_filter_caps_exposure_below_the_moving_average(tmp_path):
    config = CONFIG.model_copy(update={"trend_filter": True, "trend_window": 50})
    falling = [200 - i for i in range(60)]  # calm but always below its average
    result = exposure(store_with(tmp_path, falling), CLOSE, config)
    assert result.below_trend is True
    assert result.value == config.trend_exposure
    rising = [100 + i for i in range(60)]
    assert exposure(store_with(tmp_path, rising), CLOSE, config).below_trend is False


def test_too_short_a_history_is_refused(tmp_path):
    with pytest.raises(ValueError, match="needed for exposure"):
        exposure(store_with(tmp_path, alternating(0.01, 10)), CLOSE, CONFIG)


def test_the_shipped_sizing_config_loads_and_refuses_unknown_keys(tmp_path):
    root = Path(__file__).parents[2] / "configs"
    config = load_sizing_config("v1", root)
    assert (config.k, config.stocks.max_weight, config.exposure.target_vol) == (0.5, 0.2, 0.12)
    folder = tmp_path / "sizing"
    folder.mkdir()
    (folder / "v9.toml").write_text("kk = 1\n" + (root / "sizing" / "v1.toml").read_text())
    with pytest.raises(ValueError):
        load_sizing_config("v9", tmp_path)
