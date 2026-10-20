"""The first three patterns, as rules on candles (task 10.4).

docs/specs/09-patterns.md, with every threshold from `configs/patterns/`. Each
detector reads candles oldest first and returns events dated by the day they
became known: a two-bar pattern on its second bar, a weekly harami on the last
session of its week, a wedge breakout on the breakout day (candidate) and again
on the day it is confirmed (by holding above the breakout level, or by a
shrinking doji on a retest). Nothing a detector does on day T looks past T, so
running it on data up to T gives exactly the events dated up to T; the tests
hold it to that.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from typing import Literal

from sluice.patterns.candles import Candle, Pivot, pivots
from sluice.patterns.config import CuorouRules, HaramiRules, PatternConfig, WedgeRules

Pattern = Literal["cuorou", "harami_daily", "harami_weekly", "wedge"]
Stage = Literal["signal", "candidate", "confirmed"]


@dataclass(frozen=True)
class PatternEvent:
    pattern: Pattern
    stage: Stage  # "signal" for one-step patterns; a wedge has "candidate" then "confirmed"
    day: date  # when it became known: trading starts at the next open
    levels: dict[str, float] = field(default_factory=dict)  # prices and ratios worth showing
    how: str = ""  # a confirmed wedge: "hold" or "retest"


def _sma(c: Candle, n: int) -> float | None:
    return {20: c.sma20, 50: c.sma50, 60: c.sma60}[n]


def cuorou(series: list[Candle], rules: CuorouRules) -> list[PatternEvent]:
    """搓揉线: a long upper shadow, then a long lower shadow, volume shrinking, on a platform."""
    out = []
    for first, second in pairwise(series):
        if not (first.valid and second.valid):
            continue
        upper = first.upper >= rules.shadow_to_body * first.body and (
            first.upper >= rules.shadow_to_range * first.range
        )
        lower = second.lower >= rules.shadow_to_body * second.body and (
            second.lower >= rules.shadow_to_range * second.range
        )
        shrinking = (
            second.vol_prev is not None
            and second.vol_prev <= rules.max_volume_vs_prev
            and None not in (first.vol_avg5, second.vol_avg5)
            and first.vol_avg5 <= rules.max_volume_vs_avg5
            and second.vol_avg5 <= rules.max_volume_vs_avg5
        )
        average = _sma(second, rules.above_sma)
        platform = (
            average is not None
            and second.close > average
            and second.ret20 is not None
            and abs(second.ret20) <= rules.max_abs_ret20
        )
        if upper and lower and shrinking and platform:
            levels = {
                "day1_high": first.high,
                "day2_low": second.low,
                "volume_vs_day1": second.vol_prev,
            }
            out.append(PatternEvent("cuorou", "signal", second.day, levels))
    return out


def _mother(c: Candle, rules: HaramiRules) -> bool:
    big_bearish = (
        c.bearish and c.avg_body20 is not None and c.body >= rules.mother_body_to_avg * c.avg_body20
    )
    return big_bearish or c.lower >= rules.mother_lower_to_range * c.range


def harami(series: list[Candle], rules: HaramiRules, weekly: bool = False) -> list[PatternEvent]:
    """孕线: a small bar inside a big bearish one or a long lower shadow, after a decline."""
    name: Pattern = "harami_weekly" if weekly else "harami_daily"
    out = []
    for i in range(1, len(series)):
        mother, baby = series[i - 1], series[i]
        if not (mother.valid and baby.valid and _mother(mother, rules)):
            continue
        inside = baby.high <= mother.high and baby.low >= mother.low
        small = baby.range <= rules.baby_to_mother_range * mother.range
        if weekly:
            n = rules.weekly_bearish_weeks
            before = series[i - n : i]
            declining = len(before) == n and all(c.valid and c.bearish for c in before)
        else:
            average = _sma(baby, rules.daily_below_sma)
            declining = average is not None and baby.close < average
        if inside and small and declining:
            levels = {"mother_high": mother.high, "mother_low": mother.low}
            out.append(PatternEvent(name, "signal", baby.day, levels))
    return out


@dataclass(frozen=True)
class _Line:
    first: Pivot
    last: Pivot

    def at(self, index: int) -> float:
        slope = (self.last.price - self.first.price) / (self.last.index - self.first.index)
        return self.first.price + slope * (index - self.first.index)

    @property
    def slope(self) -> float:
        return (self.last.price - self.first.price) / (self.last.index - self.first.index)


def _descending(known: list[Pivot], rules: WedgeRules) -> list[Pivot] | None:
    """The latest pivots of one kind, counted back while each earlier one is higher."""
    run = known[-1:]
    for pivot in reversed(known[:-1]):
        if len(run) == rules.max_pivots or pivot.price <= run[0].price:
            break
        run.insert(0, pivot)
    return run if len(run) >= rules.min_pivots else None


def _wedge_at(
    series: list[Candle], t: int, highs: list[Pivot], lows: list[Pivot], rules: WedgeRules
) -> tuple[_Line, int] | None:
    """The wedge drawn from pivots known before day t, if there is one: (upper line, start)."""
    tops, bottoms = _descending(highs, rules), _descending(lows, rules)
    if tops is None or bottoms is None:
        return None
    start = min(tops[0].index, bottoms[0].index)
    if not rules.min_days <= t - start <= rules.max_days:
        return None
    upper, lower = _Line(tops[0], tops[-1]), _Line(bottoms[0], bottoms[-1])
    if not (upper.slope < 0 and lower.slope < 0 and upper.slope < lower.slope):
        return None  # both falling, the upper faster: they converge
    if upper.at(t) <= lower.at(t):
        return None  # the lines have crossed already
    drops = [a.price - b.price for a, b in pairwise(bottoms)]
    if any(later >= earlier for earlier, later in pairwise(drops)):
        return None  # the lows must fall ever more slowly
    if any(p.price > upper.at(p.index) * (1 + rules.middle_tolerance) for p in tops[1:-1]):
        return None
    span = series[start:t]
    if not all(c.valid for c in span):
        return None
    half = len(span) // 2
    if sum(c.volume for c in span[half:]) / (len(span) - half) >= (
        sum(c.volume for c in span[:half]) / half
    ):
        return None  # volume must shrink inside the wedge
    gaps = sum(abs(b.open / a.close - 1) > rules.gap for a, b in pairwise(span))
    if gaps > rules.max_gaps:
        return None
    return upper, start


def wedge(
    series: list[Candle], market: dict[date, Candle], rules: WedgeRules, side: int = 3
) -> list[PatternEvent]:
    """楔形突破: a breakout candidate, then its confirmation by a hold or a retest."""
    found = pivots(series, side)
    out: list[PatternEvent] = []
    known_highs: list[Pivot] = []
    known_lows: list[Pivot] = []
    upcoming = iter(sorted(found, key=lambda p: (p.known_on, p.index)))
    pending = next(upcoming, None)
    for t in range(1, len(series)):
        # pivots known by the day before t: the lines are drawn before the breakout
        while pending is not None and pending.known_on <= series[t - 1].day:
            (known_highs if pending.kind == "high" else known_lows).append(pending)
            pending = next(upcoming, None)
        today, yesterday = series[t], series[t - 1]
        if not (today.valid and yesterday.valid) or today.vol_avg20 is None:
            continue
        drawn = _wedge_at(series, t, known_highs, known_lows, rules)
        if drawn is None:
            continue
        upper, start = drawn
        level = upper.at(t)
        crossed = today.close > level * (1 + rules.breakout)
        first_time = yesterday.close <= upper.at(t - 1) * (1 + rules.breakout)
        spy = market.get(today.day)
        calm = (
            spy is not None
            and _sma(spy, rules.market_sma) is not None
            and (spy.close > _sma(spy, rules.market_sma))
        )
        if not (crossed and first_time and today.vol_avg20 >= rules.breakout_volume and calm):
            continue
        levels = {
            "breakout_level": level,
            "volume_vs_avg20": today.vol_avg20,
            "days": float(t - start),
        }
        out.append(PatternEvent("wedge", "candidate", today.day, levels))
        confirmed = _confirm(series, t, level, rules)
        if confirmed is not None:
            j, how = confirmed
            out.append(PatternEvent("wedge", "confirmed", series[j].day, levels, how))
    return out


def _confirm(series: list[Candle], t: int, level: float, rules: WedgeRules):
    """(index, how) of the confirmation after a breakout on t, or None if it fell back or ran out."""
    for j in range(t + 1, min(len(series), t + rules.retest_days + 1)):
        c = series[j]
        if not c.valid or c.close < level:
            return None  # a bad bar, or back inside the wedge
        doji = c.body <= rules.doji_body * c.range
        quiet = c.vol_avg5 is not None and c.vol_avg5 <= rules.doji_volume
        if c.low <= level * (1 + rules.retest_band) and doji and quiet:
            return j, "retest"
        if j - t == rules.hold_days:
            return j, "hold"  # every close since the breakout stayed at or above the level
    return None


def detect(
    daily: list[Candle],
    weekly: list[Candle],
    market: dict[date, Candle],
    config: PatternConfig,
) -> list[PatternEvent]:
    """Every pattern of the first batch, sorted by the day it became known."""
    events: Iterable[PatternEvent] = [
        *cuorou(daily, config.cuorou),
        *harami(daily, config.harami),
        *harami(weekly, config.harami, weekly=True),
        *wedge(daily, market, config.wedge),
    ]
    return sorted(events, key=lambda e: (e.day, e.pattern, e.stage))
