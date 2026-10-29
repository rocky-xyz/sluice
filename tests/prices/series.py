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
        open_price = close  # the prior bar's close, so open and close differ
        close *= 1 + drift + 0.01 * math.sin(i)
        volume = 1_000_000 + 50_000 * (i % 7)
        bars.append(
            Bar(day=day, open=open_price, close=close, raw_close=close * 1.02, volume=volume)
        )
    return bars


def with_wicks(history: list[Bar]) -> list[Bar]:
    """`history` with a high 1% above and a low 1% below the day's open and close."""
    return [
        bar.model_copy(
            update={"high": max(bar.open, bar.close) * 1.01, "low": min(bar.open, bar.close) * 0.99}
        )
        for bar in history
    ]
