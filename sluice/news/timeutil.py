from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Fixed-width, so string comparison in SQLite orders the same way as time.
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def to_utc(dt: datetime) -> datetime:
    """Convert an aware datetime to UTC. Naive datetimes are rejected, never guessed."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"naive datetime not allowed: {dt!r}")
    return dt.astimezone(UTC)


def end_of_day_et(d: date) -> datetime:
    """Last second of the US Eastern calendar day, in UTC. Used for date-only items."""
    return datetime.combine(d, time(23, 59, 59), tzinfo=ET).astimezone(UTC)


def format_utc(dt: datetime) -> str:
    """Storage format. Sub-second precision is dropped."""
    return to_utc(dt).strftime(_UTC_FORMAT)


def parse_utc(text: str) -> datetime:
    return datetime.strptime(text, _UTC_FORMAT).replace(tzinfo=UTC)
