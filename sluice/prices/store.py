import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from sluice.market_calendar import close_of
from sluice.universe import canonical

DEFAULT_PATH = Path("data/local/prices.db")
# Far enough back for 250 trading days before the first decision day of the study (2026-01).
HISTORY_START = date(2024, 1, 1)

SCHEMA_VERSION = 4
_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker    TEXT NOT NULL,
    day       TEXT NOT NULL,  -- New York trading day
    open      REAL NOT NULL,  -- adjusted for splits and dividends, like close
    close     REAL NOT NULL,  -- adjusted for splits and dividends
    raw_close REAL NOT NULL,  -- adjusted for splits only
    volume    REAL NOT NULL,
    high      REAL,           -- adjusted like open; absent from bars stored before version 4
    low       REAL,
    PRIMARY KEY (ticker, day)
);
"""
# Version 3 lacked high and low. They are added, not the table dropped: refetching would
# bring adjusted prices rewritten by later dividends, changing every prompt built from
# the stored ones, and with them the cached answers of every run so far.
MIGRATIONS = {3: ("ALTER TABLE bars ADD COLUMN high REAL", "ALTER TABLE bars ADD COLUMN low REAL")}

Fetch = Callable[[str, date], list["Bar"]]


class Bar(BaseModel):
    """One finished trading day."""

    day: date
    open: float  # adjusted for splits and dividends, like close: use for returns
    close: float  # adjusted for splits and dividends: use for returns
    raw_close: float  # adjusted for splits only: the price to value the company at
    volume: float
    high: float | None = None  # adjusted like open (task 10.2); None for bars stored before
    low: float | None = None


class PriceStore:
    """Daily bars in SQLite, fetched when a requested day is not there yet.

    Without `fetch` it only reads what is stored.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_PATH,
        fetch: Fetch | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        history_start: date = HISTORY_START,  # how far back a fetch goes (the pattern study: 10 years)
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Runs of a batch share this store from several threads.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.RLock()
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version in MIGRATIONS:
            for statement in MIGRATIONS[version]:
                self._conn.execute(statement)
        elif version != SCHEMA_VERSION:  # nothing worth keeping: prices can be fetched again
            self._conn.execute("DROP TABLE IF EXISTS bars")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()
        self._fetch = fetch
        self._now = now
        self._history_start = history_start

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def latest(self, ticker: str) -> date | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT max(day) FROM bars WHERE ticker = ?", (canonical(ticker),)
            ).fetchone()
        return date.fromisoformat(row[0]) if row[0] else None

    def replace(self, ticker: str, bars: list[Bar]) -> None:
        """Swap in a whole series.

        Adjusted prices are rewritten after every split and dividend, so old and
        new rows must never be mixed.
        """
        symbol = canonical(ticker)
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM bars WHERE ticker = ?", (symbol,))
            self._conn.executemany(
                "INSERT INTO bars (ticker, day, open, close, raw_close, volume, high, low)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        symbol,
                        bar.day.isoformat(),
                        bar.open,
                        bar.close,
                        bar.raw_close,
                        bar.volume,
                        bar.high,
                        bar.low,
                    )
                    for bar in bars
                ],
            )

    def history(self, ticker: str, through: date, limit: int) -> list[Bar]:
        """The last `limit` bars up to and including `through`, oldest first.

        Nothing after `through` is returned, whatever is stored. When the store
        ends before `through`, the series is fetched again first; only sessions
        that had closed by then are kept, so a half-finished day is never stored.
        """
        with self._lock:  # also keeps two threads from fetching the same series
            latest = self.latest(ticker)
            if self._fetch is not None and (latest is None or latest < through):
                now = self._now()
                fetched = self._fetch(canonical(ticker), self._history_start)
                finished = [bar for bar in fetched if (close := close_of(bar.day)) and close <= now]
                if finished:
                    self.replace(ticker, finished)
            rows = self._conn.execute(
                "SELECT day, open, close, raw_close, volume, high, low FROM bars"
                " WHERE ticker = ? AND day <= ? ORDER BY day DESC LIMIT ?",
                (canonical(ticker), through.isoformat(), limit),
            ).fetchall()
            return [
                Bar(
                    day=day,
                    open=open_,
                    close=close,
                    raw_close=raw,
                    volume=volume,
                    high=high,
                    low=low,
                )
                for day, open_, close, raw, volume, high, low in reversed(rows)
            ]

    def get(self, ticker: str, day: date) -> Bar | None:
        """The single bar for `day`, fetching first if the store does not reach it yet."""
        bars = self.history(ticker, through=day, limit=1)
        return bars[0] if bars and bars[0].day == day else None
