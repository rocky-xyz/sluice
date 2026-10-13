import math
import statistics
from collections.abc import Sequence
from itertools import pairwise

from sluice.prices.store import Bar

TRADING_DAYS_PER_YEAR = 252
RETURN_WINDOWS = (5, 20, 60)
VOLATILITY_WINDOW = 20
AVERAGE_WINDOWS = (50, 200)
RSI_WINDOW = 14
VOLUME_WINDOW = 20

# Every indicator, in the order it is shown. All are ratios: none reveals a price level,
# which adjusted prices would get wrong after a later split or dividend.
NAMES = (
    *(f"ret_{n}d" for n in RETURN_WINDOWS),
    *(f"excess_{n}d" for n in RETURN_WINDOWS),
    f"vol_{VOLATILITY_WINDOW}d",
    *(f"vs_sma{n}" for n in AVERAGE_WINDOWS),
    f"rsi_{RSI_WINDOW}",
    f"volume_ratio_{VOLUME_WINDOW}d",
)


def trailing_return(closes: Sequence[float], n: int) -> float | None:
    """C[t] / C[t-n] - 1."""
    if len(closes) <= n:
        return None
    return closes[-1] / closes[-1 - n] - 1


def excess_return(bars: Sequence[Bar], benchmark: Sequence[Bar], n: int) -> float | None:
    """The stock's n-day return minus the benchmark's between the same two days."""
    own = trailing_return([bar.close for bar in bars], n)
    if own is None:
        return None
    index = {bar.day: bar.close for bar in benchmark}
    first, last = index.get(bars[-1 - n].day), index.get(bars[-1].day)
    if first is None or last is None:
        return None
    return own - (last / first - 1)


def volatility(closes: Sequence[float], n: int) -> float | None:
    """Sample standard deviation of the last n daily returns, annualised."""
    if len(closes) <= n:
        return None
    window = closes[-1 - n :]
    returns = [later / earlier - 1 for earlier, later in pairwise(window)]
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)


def versus_average(closes: Sequence[float], n: int) -> float | None:
    """C[t] / mean of the last n closes - 1."""
    if len(closes) < n:
        return None
    return closes[-1] / statistics.fmean(closes[-n:]) - 1


def rsi(closes: Sequence[float], n: int) -> float | None:
    """Wilder's RSI: 100 - 100 / (1 + average gain / average loss).

    The averages start as plain means of the first n changes and are then
    smoothed: new = (old * (n - 1) + today's) / n.
    """
    if len(closes) <= n:
        return None
    changes = [later - earlier for earlier, later in pairwise(closes)]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]
    gain, loss = statistics.fmean(gains[:n]), statistics.fmean(losses[:n])
    for up, down in zip(gains[n:], losses[n:]):
        gain = (gain * (n - 1) + up) / n
        loss = (loss * (n - 1) + down) / n
    if loss == 0:
        return 100.0 if gain > 0 else 50.0  # no falls at all; 50 when the price never moved
    return 100 - 100 / (1 + gain / loss)


def volume_ratio(volumes: Sequence[float], n: int) -> float | None:
    """The last day's volume over the mean of the n days before it."""
    if len(volumes) <= n:
        return None
    usual = statistics.fmean(volumes[-1 - n : -1])
    return volumes[-1] / usual if usual > 0 else None


def compute(bars: Sequence[Bar], benchmark: Sequence[Bar] | None) -> dict[str, float]:
    """Every indicator the history is long enough for, as of the last bar.

    `bars` and `benchmark` must already end at the decision time: this function
    uses whatever it is given. Without a benchmark there are no excess returns.
    """
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    values: dict[str, float | None] = {}
    for n in RETURN_WINDOWS:
        values[f"ret_{n}d"] = trailing_return(closes, n)
    for n in RETURN_WINDOWS:
        values[f"excess_{n}d"] = excess_return(bars, benchmark, n) if benchmark else None
    values[f"vol_{VOLATILITY_WINDOW}d"] = volatility(closes, VOLATILITY_WINDOW)
    for n in AVERAGE_WINDOWS:
        values[f"vs_sma{n}"] = versus_average(closes, n)
    values[f"rsi_{RSI_WINDOW}"] = rsi(closes, RSI_WINDOW)
    values[f"volume_ratio_{VOLUME_WINDOW}d"] = volume_ratio(volumes, VOLUME_WINDOW)
    return {name: value for name, value in values.items() if value is not None}
