import math
from datetime import date, timedelta

import pytest

from sluice.market_calendar import close_of
from sluice.patterns.candles import candles, pivots, weekly
from sluice.prices.store import Bar


def sessions(first: date, count: int) -> list[date]:
    days, day = [], first
    while len(days) < count:
        if close_of(day) is not None:
            days.append(day)
        day += timedelta(days=1)
    return days


def bar(day, o, h, l, c, v=1000.0) -> Bar:
    return Bar(day=day, open=o, high=h, low=l, close=c, raw_close=c, volume=v)


def wobble(count: int, first: date = date(2026, 1, 2)) -> list[Bar]:
    """A made-up OHLC path with swings, so that pivots appear on both sides."""
    out, close = [], 100.0
    for i, day in enumerate(sessions(first, count)):
        o = close
        close = o * (1 + 0.02 * math.sin(i / 2.3) + 0.004 * math.cos(i * 1.7))
        h = max(o, close) * (1 + 0.004 + 0.003 * abs(math.sin(i)))
        l = min(o, close) * (1 - 0.004 - 0.003 * abs(math.cos(i)))
        out.append(bar(day, o, h, l, close, 1000 + 100 * (i % 7)))
    return out


def test_geometry_by_hand():
    (c,) = candles([bar(date(2026, 3, 2), o=10, h=13, l=9, c=11)])
    assert (c.body, c.upper, c.lower, c.range) == (1, 2, 1, 4)
    assert c.valid and not c.bearish
    assert c.vol_prev is None and c.sma20 is None  # nothing before it


def test_volume_ratios_look_only_backwards():
    days = sessions(date(2026, 3, 2), 21)
    bars = [bar(d, 10, 11, 9, 10, v=100.0) for d in days[:-1]] + [bar(days[-1], 10, 11, 9, 10, 50)]
    last = candles(bars)[-1]
    assert (last.vol_prev, last.vol_avg5, last.vol_avg20) == (0.5, 0.5, 0.5)
    first_ratio = candles(bars)[5]
    assert first_ratio.vol_avg5 == 1.0 and first_ratio.vol_avg20 is None


def test_moving_averages_and_return_include_todays_close():
    days = sessions(date(2026, 3, 2), 60)
    bars = [bar(d, i + 1, i + 2, i, i + 1) for i, d in enumerate(days)]
    last = candles(bars)[-1]
    assert last.sma20 == pytest.approx(sum(range(41, 61)) / 20)
    assert last.sma60 == pytest.approx(sum(range(1, 61)) / 60)
    assert last.ret20 == pytest.approx(60 / 40 - 1)
    assert last.avg_body20 == 0  # open == close every day


@pytest.mark.parametrize(
    ("o", "h", "l", "c", "v"),
    [(10, 11, 9, 10, 0), (10, 10, 10, 10, 100), (10, 11, 10.5, 10, 100), (10, 9.5, 9, 10, 100)],
)
def test_bad_bars_are_marked(o, h, l, c, v):
    (candle,) = candles([bar(date(2026, 3, 2), o, h, l, c, v)])
    assert not candle.valid


def test_a_bad_bar_spoils_the_ratios_that_would_lean_on_it():
    days = sessions(date(2026, 3, 2), 7)
    bars = [bar(d, 10, 11, 9, 10) for d in days]
    bars[3] = bar(days[3], 10, 11, 9, 10, v=0)
    assert candles(bars)[6].vol_avg5 is None and candles(bars)[6].vol_prev == 1.0


def test_pivots_are_strict_and_known_three_bars_later():
    days = sessions(date(2026, 3, 2), 9)
    highs = [10, 11, 12, 15, 12, 11, 10, 9, 8]
    bars = [bar(d, h - 1, h, h - 2, h - 0.5) for d, h in zip(days, highs, strict=True)]
    (top,) = [p for p in pivots(candles(bars)) if p.kind == "high"]
    assert (top.day, top.price, top.known_on) == (days[3], 15, days[6])
    tied = [*highs[:5], 15, *highs[6:]]
    tie_bars = [bar(d, h - 1, h, h - 2, h - 0.5) for d, h in zip(days, tied, strict=True)]
    assert [p for p in pivots(candles(tie_bars)) if p.kind == "high"] == []


def test_pivots_see_no_further_than_their_known_on_day():
    bars = wobble(160)
    full = pivots(candles(bars))
    assert len(full) > 10
    for t in range(20, len(bars)):
        upto = bars[: t + 1]
        seen = [(p.kind, p.day, p.price) for p in pivots(candles(upto))]
        known = [(p.kind, p.day, p.price) for p in full if p.known_on <= upto[-1].day]
        assert seen == known, upto[-1].day


def test_weeks_are_calendar_weeks_and_a_running_week_is_left_out():
    days = sessions(date(2026, 3, 2), 13)  # Mon 2 Mar .. Wed 18 Mar
    bars = [bar(d, 10 + i, 12 + i, 9 + i, 11 + i, 100) for i, d in enumerate(days)]
    weeks = weekly(bars)
    assert [w.day for w in weeks] == [date(2026, 3, 6), date(2026, 3, 13)]
    first = weeks[0]
    assert (first.open, first.high, first.low, first.close, first.volume) == (10, 16, 9, 15, 500)


def test_a_holiday_week_is_shorter_and_weeks_obey_point_in_time():
    days = sessions(date(2026, 3, 30), 5)  # Good Friday, 3 April 2026, has no session
    assert len([d for d in days if d.isocalendar()[1] == 14]) == 4
    bars = wobble(120, date(2026, 3, 2))
    full = weekly(bars)
    assert date(2026, 4, 2) in [w.day for w in full]  # dated by its Thursday, the last session
    for t in range(len(bars)):
        upto = bars[: t + 1]
        assert weekly(upto) == [w for w in full if w.day <= upto[-1].day], upto[-1].day
