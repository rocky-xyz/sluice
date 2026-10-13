import math
from datetime import date

import yfinance as yf

from sluice.news.yahoo import _retry
from sluice.prices.store import Bar


def fetch_bars(ticker: str, start: date) -> list[Bar]:
    """Daily bars from `start` to now, closes adjusted for splits and dividends.

    A ticker Yahoo does not know gives an empty list. Any other failure raises,
    so that an outage is not mistaken for a stock without prices.
    """

    def call() -> list[Bar]:
        try:
            frame = yf.Ticker(ticker).history(
                start=start.isoformat(), interval="1d", auto_adjust=True, actions=False
            )
        except Exception as exc:
            if "404" in str(exc):
                return []
            raise
        return [
            Bar(day=stamp.date(), close=float(row.Close), volume=float(row.Volume))
            for stamp, row in frame.iterrows()
            if not math.isnan(row.Close) and not math.isnan(row.Volume)
        ]

    return _retry(call)
