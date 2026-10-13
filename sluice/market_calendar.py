from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Literal

from sluice.news.timeutil import ET, to_utc

Session = Literal["pre", "regular", "post", "closed"]

PRE_MARKET_OPENS = time(4, 0)  # US Eastern
POST_MARKET_LENGTH = timedelta(hours=4)  # 16:00-20:00 on a normal day


@lru_cache(maxsize=1)
def _calendar():
    import exchange_calendars  # imported lazily: it is slow to load

    return exchange_calendars.get_calendar("XNYS")


@lru_cache(maxsize=4096)
def _trading_hours(day: date) -> tuple[datetime, datetime] | None:
    """Regular open and close of a New York trading day in UTC, or None when the market is shut."""
    import pandas as pd

    calendar = _calendar()
    session = pd.Timestamp(day)
    if not calendar.is_session(session):
        return None
    return (
        calendar.session_open(session).to_pydatetime(),
        calendar.session_close(session).to_pydatetime(),
    )


def close_of(day: date) -> datetime | None:
    """When regular trading ends on a New York trading day, in UTC; None when the market is shut.

    Follows early closes: 13:00 Eastern on the day after Thanksgiving, not 16:00.
    """
    hours = _trading_hours(day)
    return hours[1] if hours else None


def session_of(moment: datetime) -> Session:
    """Which part of the New York trading day a moment falls in.

    Pre-market runs from 04:00 Eastern to the open, post-market for four hours
    after the close. Intervals include their start and exclude their end.
    Weekends, holidays and the night are "closed"; on a day that closes early,
    post-market starts at the early close.
    """
    moment = to_utc(moment)
    day = moment.astimezone(ET).date()
    hours = _trading_hours(day)
    if hours is None:
        return "closed"
    opens, closes = hours
    if datetime.combine(day, PRE_MARKET_OPENS, tzinfo=ET) <= moment < opens:
        return "pre"
    if opens <= moment < closes:
        return "regular"
    if closes <= moment < closes + POST_MARKET_LENGTH:
        return "post"
    return "closed"
