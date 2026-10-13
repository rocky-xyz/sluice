import sqlite3
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

SCHEMA_VERSION = 2
# Nothing to migrate: prices can be fetched again at any time, so an old table is dropped.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker    TEXT NOT NULL,
    day       TEXT NOT NULL,  -- New York trading day
    close     REAL NOT NULL,  -- adjusted for splits and dividends
    raw_close REAL NOT NULL,  -- adjusted for splits only
    volume    REAL NOT NULL,
    PRIMARY KEY (ticker, day)
);
"""

Fetch = Callable[[str, date], list["Bar"]]


class Bar(BaseModel):
    """One finished trading day."""

    day: date
    close: float  # adjusted for splits and dividends: use for returns
    raw_close: float  # adjusted for splits only: the price to value the company at
    volume: float


class PriceStore:
    """Daily bars in SQLite, fetched when a requested day is not there yet.

    Without `fetch` it only reads what is stored.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_PATH,
        fetch: Fetch | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        if self._conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            self._conn.execute("DROP TABLE IF EXISTS bars")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()
        self._fetch = fetch
        self._now = now

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def latest(self, ticker: str) -> date | None:
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
        with self._conn:
            self._conn.execute("DELETE FROM bars WHERE ticker = ?", (symbol,))
            self._conn.executemany(
                "INSERT INTO bars (ticker, day, close, raw_close, volume) VALUES (?, ?, ?, ?, ?)",
                [
                    (symbol, bar.day.isoformat(), bar.close, bar.raw_close, bar.volume)
                    for bar in bars
                ],
            )

    def history(self, ticker: str, through: date, limit: int) -> list[Bar]:
        """The last `limit` bars up to and including `through`, oldest first.

        Nothing after `through` is returned, whatever is stored. When the store
        ends before `through`, the series is fetched again first; only sessions
        that had closed by then are kept, so a half-finished day is never stored.
        """
        latest = self.latest(ticker)
        if self._fetch is not None and (latest is None or latest < through):
            now = self._now()
            fetched = self._fetch(canonical(ticker), HISTORY_START)
            finished = [bar for bar in fetched if (close := close_of(bar.day)) and close <= now]
            if finished:
                self.replace(ticker, finished)
        rows = self._conn.execute(
            "SELECT day, close, raw_close, volume FROM bars WHERE ticker = ? AND day <= ?"
            " ORDER BY day DESC LIMIT ?",
            (canonical(ticker), through.isoformat(), limit),
        ).fetchall()
        return [
            Bar(day=day, close=close, raw_close=raw, volume=volume)
            for day, close, raw, volume in reversed(rows)
        ]
