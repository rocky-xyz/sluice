import math
import statistics
from collections.abc import Iterable, Sequence
from itertools import pairwise

from sluice.prices.store import Bar

TRADING_DAYS_PER_YEAR = 252
RETURN_WINDOWS = (5, 20, 60)
VOLATILITY_WINDOW = 20
AVERAGE_WINDOWS = (50, 200)
RSI_WINDOW = 14
VOLUME_WINDOW = 20
# The indicators added by docs/specs/10-configurable-components.md (task 11.2).
MACD_WINDOWS = (12, 26, 9)  # fast, slow, signal
BOLLINGER_WINDOW, BOLLINGER_WIDTH = 20, 2
ATR_WINDOW = 14
EMA_WINDOW = 10
RANGE_WINDOW = 250
KDJ_WINDOW = 9

# The indicators a config gets when it names none: the first eleven, in the order shown.
# All are ratios: none reveals a price level, which adjusted prices would get wrong after
# a later split or dividend.
NAMES = (
    *(f"ret_{n}d" for n in RETURN_WINDOWS),
    *(f"excess_{n}d" for n in RETURN_WINDOWS),
    f"vol_{VOLATILITY_WINDOW}d",
    *(f"vs_sma{n}" for n in AVERAGE_WINDOWS),
    f"rsi_{RSI_WINDOW}",
    f"volume_ratio_{VOLUME_WINDOW}d",
)
# Every indicator there is, in the order it is shown whichever of them a config names.
ALL = (
    *NAMES,
    "macd_hist_pct",
    f"bb_pctb_{BOLLINGER_WINDOW}",
    f"bb_width_{BOLLINGER_WINDOW}",
    f"atr_{ATR_WINDOW}_pct",
    f"vs_ema{EMA_WINDOW}",
    f"range_{RANGE_WINDOW}d",
    "kdj_k",
    "kdj_d",
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


def ema(values: Sequence[float], n: int) -> list[float] | None:
    """Exponential averages from the n-th value on: the first is the mean of the first n
    values, then new = α · value + (1 − α) · old with α = 2 / (n + 1)."""
    if len(values) < n:
        return None
    alpha = 2 / (n + 1)
    averages = [statistics.fmean(values[:n])]
    for value in values[n:]:
        averages.append(alpha * value + (1 - alpha) * averages[-1])
    return averages


def versus_ema(closes: Sequence[float], n: int) -> float | None:
    """C[t] / EMA_n - 1."""
    averages = ema(closes, n)
    return closes[-1] / averages[-1] - 1 if averages else None


def macd_histogram(
    closes: Sequence[float], fast: int, slow: int, signal: int
) -> float | None:
    """(DIF − DEA) / C[t], where DIF = EMA_fast − EMA_slow and DEA = EMA_signal of DIF.

    The histogram is the international one, not doubled as Chinese charts draw it.
    """
    fast_averages, slow_averages = ema(closes, fast), ema(closes, slow)
    if fast_averages is None or slow_averages is None:
        return None
    dif = [f - s for f, s in zip(fast_averages[-len(slow_averages) :], slow_averages)]
    dea = ema(dif, signal)
    return (dif[-1] - dea[-1]) / closes[-1] if dea else None


def _band(closes: Sequence[float], n: int, k: float) -> tuple[float, float, float] | None:
    """Bollinger lower band, middle line and upper band over the last n closes.

    The deviation is the population one (divided by n), as the bands are usually drawn.
    """
    if len(closes) < n:
        return None
    window = closes[-n:]
    middle, deviation = statistics.fmean(window), statistics.pstdev(window)
    return middle - k * deviation, middle, middle + k * deviation


def bollinger_position(closes: Sequence[float], n: int, k: float) -> float | None:
    """(C[t] − lower) / (upper − lower): 0 on the lower band, 1 on the upper."""
    band = _band(closes, n, k)
    if band is None or band[2] == band[0]:  # no movement at all: no band to sit in
        return None
    lower, _, upper = band
    return (closes[-1] - lower) / (upper - lower)


def bollinger_width(closes: Sequence[float], n: int, k: float) -> float | None:
    """(upper − lower) / middle."""
    band = _band(closes, n, k)
    if band is None or band[1] == 0:
        return None
    lower, middle, upper = band
    return (upper - lower) / middle


def _wicks(bars: Sequence[Bar]) -> tuple[list[float], list[float]] | None:
    """Highs and lows, or None when any bar lacks them (bars stored before task 10.2)."""
    if any(bar.high is None or bar.low is None for bar in bars):
        return None
    return [bar.high for bar in bars], [bar.low for bar in bars]


def atr_ratio(bars: Sequence[Bar], n: int) -> float | None:
    """Wilder's average true range over C[t].

    TR = max(H − L, |H − previous C|, |L − previous C|), from the second bar on. The
    average starts as the plain mean of the first n TRs and is then smoothed as in RSI:
    new = (old · (n − 1) + today's) / n.
    """
    wicks = _wicks(bars)
    if wicks is None or len(bars) <= n:
        return None
    highs, lows = wicks
    ranges = [
        max(high - low, abs(high - before.close), abs(low - before.close))
        for before, high, low in zip(bars, highs[1:], lows[1:])
    ]
    average = statistics.fmean(ranges[:n])
    for today in ranges[n:]:
        average = (average * (n - 1) + today) / n
    return average / bars[-1].close


def range_position(bars: Sequence[Bar], n: int) -> float | None:
    """(C[t] − lowest low) / (highest high − lowest low) over the last n bars, t included."""
    if len(bars) < n:
        return None
    wicks = _wicks(bars[-n:])
    if wicks is None:
        return None
    highest, lowest = max(wicks[0]), min(wicks[1])
    if highest == lowest:
        return None
    return (bars[-1].close - lowest) / (highest - lowest)


def kdj(bars: Sequence[Bar], n: int) -> tuple[float, float] | None:
    """K and D of KDJ (n, 3, 3), as Chinese charts compute it.

    RSV = (C − lowest low of n days) / (highest high of n days − lowest low) · 100;
    K = 2/3 · K before + 1/3 · RSV, D = 2/3 · D before + 1/3 · K, both starting from 50
    before the first day RSV can be computed. A day whose n-day range is zero leaves RSV
    undefined, and with it everything after, so there is then no value at all.
    """
    wicks = _wicks(bars)
    if wicks is None or len(bars) < n:
        return None
    highs, lows = wicks
    k = d = 50.0
    for t in range(n - 1, len(bars)):
        highest, lowest = max(highs[t - n + 1 : t + 1]), min(lows[t - n + 1 : t + 1])
        if highest == lowest:
            return None
        rsv = (bars[t].close - lowest) / (highest - lowest) * 100
        k = 2 / 3 * k + 1 / 3 * rsv
        d = 2 / 3 * d + 1 / 3 * k
    return k, d


def compute(
    bars: Sequence[Bar], benchmark: Sequence[Bar] | None, names: Iterable[str] = NAMES
) -> dict[str, float]:
    """The indicators in `names` the history is long enough for, as of the last bar.

    They come back in the order of `ALL`, whatever the order of `names`. `bars` and
    `benchmark` must already end at the decision time: this function uses whatever it
    is given. Without a benchmark there are no excess returns.
    """
    wanted = set(names)
    if unknown := sorted(wanted - set(ALL)):
        raise ValueError(f"unknown indicators: {', '.join(unknown)}")
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
    values["macd_hist_pct"] = macd_histogram(closes, *MACD_WINDOWS)
    values[f"bb_pctb_{BOLLINGER_WINDOW}"] = bollinger_position(
        closes, BOLLINGER_WINDOW, BOLLINGER_WIDTH
    )
    values[f"bb_width_{BOLLINGER_WINDOW}"] = bollinger_width(
        closes, BOLLINGER_WINDOW, BOLLINGER_WIDTH
    )
    values[f"atr_{ATR_WINDOW}_pct"] = atr_ratio(bars, ATR_WINDOW)
    values[f"vs_ema{EMA_WINDOW}"] = versus_ema(closes, EMA_WINDOW)
    values[f"range_{RANGE_WINDOW}d"] = range_position(bars, RANGE_WINDOW)
    values["kdj_k"], values["kdj_d"] = kdj(bars, KDJ_WINDOW) or (None, None)
    return {
        name: value
        for name in ALL
        if name in wanted and (value := values[name]) is not None
    }
