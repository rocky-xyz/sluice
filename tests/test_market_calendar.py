from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from sluice.market_calendar import (
    close_of,
    last_closed_session,
    next_open_after,
    nth_trading_day,
    session_of,
)
from sluice.news.timeutil import ET


def et(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


@pytest.mark.parametrize(
    ("hour", "minute", "second", "expected"),
    [
        (3, 59, 59, "closed"),
        (4, 0, 0, "pre"),
        (9, 29, 59, "pre"),
        (9, 30, 0, "regular"),
        (15, 59, 59, "regular"),
        (16, 0, 0, "post"),
        (19, 59, 59, "post"),
        (20, 0, 0, "closed"),
    ],
)
@pytest.mark.parametrize("day", [(2026, 3, 2), (2026, 7, 15)])  # winter time, summer time
def test_session_boundaries_follow_the_eastern_clock(day, hour, minute, second, expected):
    assert session_of(et(*day, hour, minute, second)) == expected


def test_the_same_instant_in_another_zone_gives_the_same_answer():
    moment = et(2026, 3, 2, 10)
    assert session_of(moment.astimezone(UTC)) == "regular"
    assert session_of(moment.astimezone(timezone(timedelta(hours=9)))) == "regular"


@pytest.mark.parametrize(
    "day",
    [
        (2026, 3, 7),  # Saturday
        (2026, 3, 8),  # Sunday
        (2026, 7, 3),  # Independence Day, observed on a Friday
        (2025, 11, 27),  # Thanksgiving
        (2026, 4, 3),  # Good Friday
    ],
)
@pytest.mark.parametrize("hour", [5, 10, 17])
def test_weekends_and_holidays_are_closed_all_day(day, hour):
    assert session_of(et(*day, hour)) == "closed"


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(9, 30, "regular"), (12, 59, "regular"), (13, 0, "post"), (16, 59, "post"), (17, 0, "closed")],
)
def test_early_close_moves_the_post_market(hour, minute, expected):
    # The day after Thanksgiving 2025 closed at 13:00.
    assert session_of(et(2025, 11, 28, hour, minute)) == expected


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        session_of(et(2026, 3, 2, 10).replace(tzinfo=None))


def test_close_follows_early_closes_and_is_none_without_trading():
    assert close_of(date(2026, 9, 30)) == datetime(2026, 9, 30, 20, 0, tzinfo=UTC)
    assert close_of(date(2026, 1, 5)) == datetime(2026, 1, 5, 21, 0, tzinfo=UTC)  # winter time
    assert close_of(date(2026, 11, 27)) == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
    assert close_of(date(2026, 10, 3)) is None


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (datetime(2026, 9, 30, 20, 0, tzinfo=UTC), date(2026, 9, 30)),  # at the close
        (datetime(2026, 9, 30, 19, 59, 59, tzinfo=UTC), date(2026, 9, 29)),  # a second before
        (datetime(2026, 10, 3, 15, 0, tzinfo=UTC), date(2026, 10, 2)),  # Saturday
        (datetime(2026, 10, 5, 12, 0, tzinfo=UTC), date(2026, 10, 2)),  # Monday before the open
        (datetime(2026, 11, 27, 18, 0, tzinfo=UTC), date(2026, 11, 27)),  # early close
        (datetime(2026, 11, 27, 3, 0, tzinfo=UTC), date(2026, 11, 25)),  # Thanksgiving night, UTC
    ],
)
def test_last_closed_session(moment, expected):
    assert last_closed_session(moment) == expected


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (et(2026, 9, 30, 7, 0), date(2026, 9, 30)),  # pre-market, same day's open is next
        (et(2026, 9, 30, 9, 29, 59), date(2026, 9, 30)),  # a second before the open
        (et(2026, 9, 30, 9, 30, 0), date(2026, 10, 1)),  # at the open itself: already happened
        (et(2026, 9, 30, 11, 0), date(2026, 10, 1)),  # mid-session: next day's open
        (et(2026, 9, 30, 17, 0), date(2026, 10, 1)),  # after close
        (et(2026, 10, 2, 17, 0), date(2026, 10, 5)),  # Friday after close: skip the weekend
    ],
)
def test_next_open_after(moment, expected):
    assert next_open_after(moment) == expected


def test_nth_trading_day_counts_start_as_day_one():
    assert nth_trading_day(date(2026, 9, 30), 1) == date(2026, 9, 30)
    assert nth_trading_day(date(2026, 9, 30), 3) == date(2026, 10, 2)  # Wed, Thu, Fri


def test_nth_trading_day_skips_the_weekend():
    assert nth_trading_day(date(2026, 10, 2), 3) == date(2026, 10, 6)  # Fri, Mon, Tue
