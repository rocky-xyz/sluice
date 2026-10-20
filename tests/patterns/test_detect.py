from datetime import date

import pytest

from sluice.patterns.candles import candles, weekly
from sluice.patterns.config import load_pattern_config
from sluice.patterns.detect import cuorou, detect, harami, wedge
from sluice.prices.store import Bar
from tests.patterns.test_candles import sessions, wobble

RULES = load_pattern_config("v1")


def ohlc(day, o, h, l, c, v=1000.0) -> Bar:
    return Bar(day=day, open=o, high=h, low=l, close=c, raw_close=c, volume=v)


def drift(days, start, step, volume=1000.0, body=0.1):
    """Quiet bars moving by `step` a day: small bodies, short shadows."""
    out, price = [], start
    for d in days:
        o, price = price, price * (1 + step)
        out.append(ohlc(d, o, max(o, price) + body, min(o, price) - body, price, volume))
    return out


# --- 搓揉线 ---


def cuorou_bars(up=3.0, down=2.8, v1=900.0, v2=600.0, step=0.0015, start=90.0):
    """70 quiet bars, then a long upper shadow (body 0.3) and a long lower shadow (body 0.1)."""
    days = sessions(date(2026, 1, 2), 72)
    base = drift(days[:70], start, step)
    o = base[-1].close
    day1 = ohlc(days[70], o, o + 0.3 + up, o - 0.2, o + 0.3, v1)
    o2 = day1.close
    day2 = ohlc(days[71], o2 - 0.1, o2 + 0.2, o2 - 0.1 - down, o2, v2)
    return [*base, day1, day2]


def test_cuorou_by_the_rules():
    bars = cuorou_bars()
    (event,) = cuorou(candles(bars), RULES.cuorou)
    assert (event.pattern, event.stage, event.day) == ("cuorou", "signal", bars[-1].day)
    assert event.levels["volume_vs_day1"] == pytest.approx(600 / 900)


@pytest.mark.parametrize(
    "spoil",
    [
        {"v2": 700.0},  # day 2's volume 0.78 of day 1's: not shrinking enough
        {"v1": 1500.0},  # day 1 above its 5-day average
        {"up": 0.5},  # day 1's upper shadow under two bodies
        {"down": 0.15},  # day 2's lower shadow under two bodies
        {"step": -0.002, "start": 110.0},  # falling: below the 60-day line
    ],
)
def test_cuorou_needs_every_condition(spoil):
    assert cuorou(candles(cuorou_bars(**spoil)), RULES.cuorou) == []


# --- 孕线 ---


def falling_then(*last: tuple[float, float, float, float]) -> list[Bar]:
    """30 falling bars, then bars given as offsets (open, high, low, close) from the last close."""
    days = sessions(date(2026, 1, 2), 30 + len(last))
    bars = drift(days[:30], 110, -0.004, body=0.05)
    o = bars[-1].close
    for d, (a, h, l, c) in zip(days[30:], last, strict=True):
        bars.append(ohlc(d, o + a, o + h, o + l, o + c))
    return bars


def test_daily_harami_after_a_decline():
    bars = falling_then((0, 0.3, -6.5, -6.0), (-5.5, -4.5, -6.0, -5.0))  # big bearish mother
    (event,) = harami(candles(bars), RULES.harami)
    assert (event.pattern, event.day) == ("harami_daily", bars[-1].day)
    outside = falling_then((0, 0.3, -6.5, -6.0), (-5.5, 0.4, -6.0, -5.0))  # pokes above
    assert harami(candles(outside), RULES.harami) == []


def test_a_long_lower_shadow_can_be_the_mother():
    bars = falling_then((0, 0.2, -4.0, 0.1), (-0.5, 0, -1.5, -0.8))  # bullish, shadow 4 of 4.2
    assert [e.pattern for e in harami(candles(bars), RULES.harami)] == ["harami_daily"]


def test_weekly_harami_needs_three_bearish_weeks_before_it():
    def week(i, o, h, l, c):
        return ohlc(date(2026, 1, 9 + 7 * i), o, h, l, c, 5000)

    weeks = [
        week(0, 100, 101, 95, 96),
        week(1, 96, 97, 91, 92),
        week(2, 92, 92.5, 80, 91),  # bearish, lower shadow 11 of 12.5: the mother
        week(3, 86, 88, 84, 87),  # inside it, a third of its range
    ]
    (event,) = harami(candles(weeks), RULES.harami, weekly=True)
    assert (event.pattern, event.day) == ("harami_weekly", weeks[3].day)
    bullish_start = [weeks[0].model_copy(update={"close": 100.5}), *weeks[1:]]
    assert harami(candles(bullish_start), RULES.harami, weekly=True) == []


# --- 楔形突破, with the notes' numbers ---

HIGHS = [30, 27, 25, 23, 22]
LOWS = [26, 23, 21.5, 20.8, 20.5]
SEGMENT = 5  # bars from one pivot to the next


def wedge_bars(after, lows=LOWS, first=date(2025, 6, 2)) -> tuple[list[Bar], int]:
    """A rise into 30, the notes' wedge, a breakout bar, then `after` bars as (close, low).

    Returns the bars and the breakout's index.
    """
    path = [28 + 2 * i / 25 for i in range(25)]  # rising into the first high
    turns = [p for pair in zip(HIGHS, lows, strict=True) for p in pair]
    for a, b in zip([path[-1], *turns], turns, strict=False):
        path += [a + (b - a) * (k + 1) / SEGMENT for k in range(SEGMENT)]
    path += [20.55, 20.6, 20.6]  # the last low settles; no breakout yet
    days = sessions(first, len(path) + 1 + len(after))
    bars, prev = [], path[0]
    for i, (d, c) in enumerate(zip(days, path, strict=False)):
        o = (prev + c) / 2  # halfway: no tie between a pivot and its neighbours
        bars.append(ohlc(d, o, max(o, c) + 0.05, min(o, c) - 0.05, c, 2000 - 1200 * i / len(path)))
        prev = c
    t = len(bars)
    bars.append(ohlc(days[t], 20.7, 21.6, 20.6, 21.5, 3000))  # the breakout, on heavy volume
    for d, (c, low) in zip(days[t + 1 :], after, strict=True):
        o = bars[-1].close
        bars.append(ohlc(d, o, max(o, c) + 0.05, low, c, 800))
    return bars, t


def calm(bars, step=0.001):
    return {c.day: c for c in candles(drift([b.day for b in bars], 400, step))}


def test_the_notes_wedge_breaks_out_and_holds():
    bars, t = wedge_bars([(21.6, 21.5), (21.7, 21.55), (21.8, 21.65)])
    events = wedge(candles(bars), calm(bars), RULES.wedge)
    assert [(e.stage, e.day, e.how) for e in events] == [
        ("candidate", bars[t].day, ""),
        ("confirmed", bars[t + 3].day, "hold"),
    ]
    level = events[0].levels["breakout_level"]
    assert 20 < level < 21 and events[0].levels["volume_vs_avg20"] > 1.5
    assert events[1].levels == events[0].levels  # the level stays where the breakout put it


def test_a_retest_with_a_quiet_doji_confirms_it_early():
    bars, t = wedge_bars([(21.0, 20.6)])
    (candidate,) = wedge(candles(bars[: t + 1]), calm(bars), RULES.wedge)
    level = candidate.levels["breakout_level"]
    o = bars[t].close
    bars[t + 1] = ohlc(bars[t + 1].day, o - 0.6, o - 0.4, level * 1.005, o - 0.61, 500)
    events = wedge(candles(bars), calm(bars), RULES.wedge)
    assert [(e.stage, e.how) for e in events] == [("candidate", ""), ("confirmed", "retest")]


def test_falling_back_inside_the_wedge_leaves_only_the_candidate():
    bars, _ = wedge_bars([(21.6, 21.5), (19.9, 19.8), (21.8, 21.6)])
    assert [e.stage for e in wedge(candles(bars), calm(bars), RULES.wedge)] == ["candidate"]


def test_no_breakout_without_volume_or_in_a_weak_market():
    bars, t = wedge_bars([])
    quiet = [*bars[:t], bars[t].model_copy(update={"volume": 1000.0})]
    assert wedge(candles(quiet), calm(quiet), RULES.wedge) == []
    assert wedge(candles(bars), calm(bars, step=-0.002), RULES.wedge) == []


def test_lows_that_stop_slowing_down_are_no_wedge():
    unsteady = [26, 23, 22.2, 21.2, 20.5]  # drops 3, 0.8, 1.0, 0.7: the third is larger
    bars, _ = wedge_bars([(21.6, 21.5)], lows=unsteady)
    assert wedge(candles(bars), calm(bars), RULES.wedge) == []


# --- point in time ---


def test_every_detector_sees_no_further_than_the_day_it_is_run_on():
    head = wobble(150, date(2024, 6, 3))  # more history before the wedge
    tail, _ = wedge_bars([(21.6, 21.5), (21.7, 21.55), (21.8, 21.65), (22.0, 21.8)])
    bars = [*head, *tail]
    market = calm(bars)
    full = detect(candles(bars), candles(weekly(bars)), market, RULES)
    assert {(e.pattern, e.stage) for e in full} >= {("wedge", "candidate"), ("wedge", "confirmed")}
    for t in range(30, len(bars)):
        upto = bars[: t + 1]
        seen = detect(candles(upto), candles(weekly(upto)), market, RULES)
        assert seen == [e for e in full if e.day <= upto[-1].day], upto[-1].day
