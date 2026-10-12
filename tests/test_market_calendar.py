from datetime import UTC, datetime, timedelta, timezone

import pytest

from sluice.market_calendar import session_of
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
