import math
from datetime import date, timedelta

from sluice.market_calendar import close_of
from sluice.prices.store import Bar


def trading_days(last: date, count: int) -> list[date]:
    """The `count` New York trading days ending on `last`, oldest first."""
    days, day = [], last
    while len(days) < count:
        if close_of(day) is not None:
            days.append(day)
        day -= timedelta(days=1)
    return days[::-1]


def series(last: date, count: int, drift: float = 0.001, start: float = 100.0) -> list[Bar]:
    """A made-up price path: steady drift with a wobble, so that every indicator is defined."""
    bars, close = [], start
    for i, day in enumerate(trading_days(last, count)):
        close *= 1 + drift + 0.01 * math.sin(i)
        volume = 1_000_000 + 50_000 * (i % 7)
        bars.append(Bar(day=day, close=close, raw_close=close * 1.02, volume=volume))
    return bars
