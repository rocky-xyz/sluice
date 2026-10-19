"""How much of the book is invested: the market state, computed by code (task 6.9).

docs/specs/04-sizing.md, step 8 of the formula: exposure = target volatility
over the benchmark's realised volatility, capped at 1 and floored; optionally
also capped while the benchmark closes below its long moving average. Only
bars of sessions that had closed by the decision time are used.
"""

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

from sluice.market_calendar import last_closed_session
from sluice.prices.store import PriceStore
from sluice.sizing.config import ExposureConfig

TRADING_DAYS = 252


def realised_vol(closes: list[float]) -> float:
    """Annualised sample standard deviation of the daily log returns between `closes`."""
    returns = [math.log(b / a) for a, b in pairwise(closes)]
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS)


@dataclass(frozen=True)
class Exposure:
    value: float  # share of the book to invest
    vol: float  # the benchmark's realised volatility it was set from
    below_trend: bool | None  # None when the trend filter is off


def exposure(prices: PriceStore, decision_time: datetime, config: ExposureConfig) -> Exposure:
    through = last_closed_session(decision_time)
    needed = max(config.vol_lookback + 1, config.trend_window if config.trend_filter else 0)
    bars = prices.history(config.benchmark, through, needed)
    if len(bars) < needed:
        raise ValueError(
            f"{config.benchmark}: {len(bars)} bars up to {through}, {needed} needed for exposure"
        )
    closes = [bar.close for bar in bars]
    vol = realised_vol(closes[-(config.vol_lookback + 1) :])
    value = min(1.0, max(config.floor, config.target_vol / vol))
    below = None
    if config.trend_filter:
        below = closes[-1] < statistics.fmean(closes[-config.trend_window :])
        if below:
            value = min(value, config.trend_exposure)
    return Exposure(value=value, vol=vol, below_trend=below)
