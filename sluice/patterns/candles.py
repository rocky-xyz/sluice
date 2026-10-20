"""What the pattern rules are written in: candle geometry, volume ratios, pivots, weeks (task 10.3).

docs/specs/09-patterns.md. Every quantity of a bar looks back only: averages of
volume and body are over the bars before it, moving averages and the 20-day
return end at its own close. A bar with no volume, no range or prices that
contradict each other is marked invalid, and no pattern may use it (a zero
volume would read as the deepest shrinkage, a zero range makes shadow ratios
meaningless).

A pivot high is a bar whose high is strictly above the highs of the `k` bars
on each side; a tie on either side is no pivot. It is known only once the `k`
bars to its right have closed, and carries that day as `known_on`: a rule run
on day T may use only pivots known by T. A weekly bar covers a calendar week,
Monday to Friday, and exists only once the week's last session has closed.
"""

import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from sluice.market_calendar import close_of
from sluice.prices.store import Bar

PIVOT_SIDE = 3  # bars on each side of a pivot (spec 09)
TOLERANCE = 1e-9  # a price rounded on the vendor's side may sit a hair outside the range


@dataclass(frozen=True)
class Candle:
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    valid: bool
    body: float  # |close - open|
    upper: float  # high - max(open, close)
    lower: float  # min(open, close) - low
    range: float  # high - low
    vol_prev: float | None  # volume over the previous bar's
    vol_avg5: float | None  # volume over the average of the 5 bars before
    vol_avg20: float | None  # ... of the 20 bars before
    avg_body20: float | None  # average body of the 20 bars before
    sma20: float | None  # average close of the last 20 bars, this one included
    sma50: float | None
    sma60: float | None
    ret20: float | None  # close over the close 20 bars before, minus 1

    @property
    def bearish(self) -> bool:
        return self.close < self.open


def _valid(bar: Bar) -> bool:
    if bar.high is None or bar.low is None or bar.volume <= 0 or bar.high - bar.low <= 0:
        return False
    return bar.low <= min(bar.open, bar.close) + TOLERANCE and (
        bar.high >= max(bar.open, bar.close) - TOLERANCE
    )


def _ratio(value: float, before: list[Bar], valid: list[bool]) -> float | None:
    """`value` over the average volume of `before`; None if it is short or holds a bad bar."""
    if not before or not all(valid):
        return None
    return value / statistics.fmean(b.volume for b in before)


def candles(bars: list[Bar]) -> list[Candle]:
    """Every bar with its geometry and the look-back measures the rules need, oldest first."""
    ok = [_valid(bar) for bar in bars]
    out = []
    for i, bar in enumerate(bars):
        high = bar.high if bar.high is not None else max(bar.open, bar.close)
        low = bar.low if bar.low is not None else min(bar.open, bar.close)
        closes = [b.close for b in bars[max(0, i - 59) : i + 1]]

        def back(n: int, i: int = i) -> tuple[list[Bar], list[bool]]:
            if i < n:
                return [], []
            return bars[i - n : i], ok[i - n : i]

        def sma(n: int, closes: list[float] = closes) -> float | None:
            return statistics.fmean(closes[-n:]) if len(closes) >= n else None

        b20, v20 = back(20)
        out.append(
            Candle(
                day=bar.day, open=bar.open, high=high, low=low, close=bar.close,
                volume=bar.volume, valid=ok[i],
                body=abs(bar.close - bar.open),
                upper=high - max(bar.open, bar.close),
                lower=min(bar.open, bar.close) - low,
                range=high - low,
                vol_prev=_ratio(bar.volume, *back(1)),
                vol_avg5=_ratio(bar.volume, *back(5)),
                vol_avg20=_ratio(bar.volume, b20, v20),
                avg_body20=statistics.fmean(abs(b.close - b.open) for b in b20) if b20 else None,
                sma20=sma(20), sma50=sma(50), sma60=sma(60),
                ret20=bar.close / bars[i - 20].close - 1 if i >= 20 else None,
            )
        )  # fmt: skip
    return out


@dataclass(frozen=True)
class Pivot:
    kind: Literal["high", "low"]
    index: int  # into the candles it was found in
    day: date
    price: float
    known_on: date  # the close of the k-th bar to its right


def pivots(series: list[Candle], k: int = PIVOT_SIDE) -> list[Pivot]:
    """Strict pivot highs and lows, oldest first; a window holding a bad bar gives none."""
    found = []
    for i in range(k, len(series) - k):
        window = series[i - k : i + k + 1]
        if not all(c.valid for c in window):
            continue
        sides = window[:k] + window[k + 1 :]
        here = series[i]
        if all(here.high > c.high for c in sides):
            found.append(Pivot("high", i, here.day, here.high, series[i + k].day))
        if all(here.low < c.low for c in sides):
            found.append(Pivot("low", i, here.day, here.low, series[i + k].day))
    return found


def _week_over(day: date) -> bool:
    """No session is left in `day`'s calendar week after it."""
    later = day + timedelta(days=1)
    while later.isocalendar()[:2] == day.isocalendar()[:2]:
        if close_of(later) is not None:
            return False
        later += timedelta(days=1)
    return True


def weekly(bars: list[Bar]) -> list[Bar]:
    """Calendar-week bars, dated by the week's last session; a week still running is left out.

    A week with a bad day keeps it: `candles` then marks the weekly bar invalid
    when it lacks a high or low, and its volume is the sum of the days given.
    """
    weeks: dict[tuple[int, int], list[Bar]] = {}
    for bar in bars:
        weeks.setdefault(bar.day.isocalendar()[:2], []).append(bar)
    out = []
    for days in weeks.values():
        last = days[-1]
        if not _week_over(last.day):
            continue
        highs = [b.high for b in days]
        lows = [b.low for b in days]
        out.append(
            Bar(
                day=last.day,
                open=days[0].open,
                close=last.close,
                raw_close=last.raw_close,
                volume=sum(b.volume for b in days),
                high=None if None in highs else max(highs),
                low=None if None in lows else min(lows),
            )
        )
    return out
