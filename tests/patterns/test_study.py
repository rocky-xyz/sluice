import math
from datetime import date

import pytest

from sluice.patterns.detect import PatternEvent
from sluice.patterns.study import (
    MIN_EVENTS,
    Baseline,
    Outcome,
    StoredStats,
    latest_before,
    load_all,
    month_starts,
    outcomes,
    save,
    table,
)
from sluice.prices.store import Bar
from tests.patterns.test_candles import sessions

DAYS = sessions(date(2026, 1, 5), 40)


def flat(price=100.0, rising=0.0) -> list[Bar]:
    out = []
    for i, d in enumerate(DAYS):
        p = price * (1 + rising) ** i
        out.append(
            Bar(day=d, open=p, high=p * 1.01, low=p * 0.99, close=p, raw_close=p, volume=1000)
        )
    return out


SPY = {b.day: b for b in flat(400)}  # flat market: excess is the stock's own return


def test_the_label_runs_from_the_next_open_to_the_horizons_close():
    bars = flat()
    bars[1] = bars[1].model_copy(update={"open": 100.0})
    bars[5] = bars[5].model_copy(update={"close": 110.0})
    (five, ten) = outcomes("AAA", bars, SPY, [PatternEvent("cuorou", "signal", DAYS[0])])
    assert (five.horizon, five.settled) == (5, DAYS[5])
    assert five.excess == pytest.approx(math.log(110 / 100))
    assert (ten.horizon, ten.settled, ten.excess) == (10, DAYS[10], pytest.approx(0.0))


def test_the_market_is_taken_out():
    bars = flat(rising=0.01)
    spy = {b.day: b for b in flat(400, rising=0.01)}  # the market rose as much
    (five, _) = outcomes("AAA", bars, spy, [PatternEvent("cuorou", "signal", DAYS[0])])
    assert five.excess == pytest.approx(0.0, abs=1e-12)


def test_a_repeat_within_ten_days_counts_once_and_a_late_event_has_no_label():
    events = [PatternEvent("harami_daily", "signal", DAYS[i]) for i in (0, 4, 10, 33)]
    found = outcomes("AAA", flat(), SPY, events)
    assert sorted({o.day for o in found}) == [DAYS[0], DAYS[10], DAYS[33]]
    assert [o.horizon for o in found if o.day == DAYS[33]] == [5]  # 10 days do not fit in 40


def outcome(day, excess, settled, pattern="wedge", h=10) -> Outcome:
    return Outcome("AAA", pattern, "confirmed", day, h, settled, excess)


def test_a_table_uses_only_what_settled_before_its_date():
    found = [
        outcome(date(2026, 1, 5), 0.02, date(2026, 1, 20)),
        outcome(date(2026, 1, 26), 0.04, date(2026, 2, 9)),  # settles in February
    ]
    (january,) = table(found, Baseline.empty(), date(2026, 2, 1))
    assert (january.n, january.mean) == (1, pytest.approx(0.02))
    (march,) = table(found, Baseline.empty(), date(2026, 3, 1))
    assert (march.n, march.mean) == (2, pytest.approx(0.03))
    assert march.too_few and MIN_EVENTS == 30


def test_the_interval_resamples_whole_days_and_is_repeatable():
    same_day = [outcome(date(2026, 1, 5), v, date(2026, 1, 20)) for v in (0.05, 0.06, 0.07)]
    other = [outcome(date(2026, 1, 6 + i), -0.01, date(2026, 1, 21)) for i in range(3)]
    (row,) = table(same_day + other, Baseline.empty(), date(2026, 2, 1))
    assert row.low <= row.mean <= row.high
    assert (row.low, row.high) == pytest.approx(
        (table(same_day + other, Baseline.empty(), date(2026, 2, 1))[0].low, row.high)
    )
    assert row.hit == pytest.approx(0.5)


def test_the_baseline_is_every_stock_day_settled_before_the_table():
    base = Baseline.empty()
    base.add(flat(rising=0.001), SPY)
    mean, n = base.before(date(2026, 2, 1), 5)
    # here open equals that day's close, so next open to the 5th close is 4 days of growth
    assert n > 0 and mean == pytest.approx(4 * math.log(1.001), rel=1e-6)
    found = [outcome(date(2026, 1, 5), 0.02, date(2026, 1, 12), pattern="cuorou", h=5)]
    (row,) = table(found, base, date(2026, 2, 1))
    assert row.excess_over_baseline == pytest.approx(0.02 - mean)
    assert base.before(date(2026, 1, 1), 5) == (None, 0)


def test_tables_are_kept_by_month_and_read_as_of_a_moment(tmp_path):
    for moment in month_starts(date(2026, 1, 15), date(2026, 3, 2)):
        save(StoredStats(moment, "v1-x", "sp500-test", []), tmp_path)
    tables = load_all("v1-x", tmp_path)
    assert [t.as_of for t in tables] == [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)]
    assert latest_before(tables, date(2026, 2, 27)).as_of == date(2026, 2, 1)
    assert latest_before(tables, date(2025, 12, 31)) is None
