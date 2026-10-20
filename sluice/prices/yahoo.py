import math
from datetime import date

import yfinance as yf

from sluice.news.yahoo import _retry
from sluice.prices.store import Bar


def fetch_bars(ticker: str, start: date) -> list[Bar]:
    """Daily bars from `start` to now, with the close both fully and split-only adjusted.

    A ticker Yahoo does not know gives an empty list. Any other failure raises,
    so that an outage is not mistaken for a stock without prices.
    """

    def call() -> list[Bar]:
        try:
            frame = yf.Ticker(ticker).history(
                start=start.isoformat(), interval="1d", auto_adjust=False, actions=False
            )
        except Exception as exc:
            if "404" in str(exc):
                return []
            raise
        bars = []
        for stamp, row in frame.iterrows():
            if any(math.isnan(row[c]) for c in ("Open", "Adj Close", "Close", "Volume")):
                continue
            # Yahoo gives no adjusted open, high or low; apply the same day's split+dividend
            # adjustment ratio (Adj Close / Close) that `close` already carries.
            ratio = float(row["Adj Close"]) / float(row["Close"])
            high, low = float(row["High"]), float(row["Low"])
            bars.append(
                Bar(
                    day=stamp.date(),
                    open=float(row["Open"]) * ratio,
                    close=float(row["Adj Close"]),
                    raw_close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    high=None if math.isnan(high) else high * ratio,
                    low=None if math.isnan(low) else low * ratio,
                )
            )
        return bars

    return _retry(call)
