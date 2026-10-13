import time
from collections.abc import Callable

import requests

URL = "https://www.alphavantage.co/query"
# One call each per ticker; the free key allows 25 calls a day in total.
FUNCTIONS = ("INCOME_STATEMENT", "CASH_FLOW", "BALANCE_SHEET")
MIN_INTERVAL = 1.5  # seconds between requests; the free key allows one a second


class AlphaVantageError(RuntimeError):
    pass


class QuotaExceeded(AlphaVantageError):
    """The key's rate or daily limit was hit. Nothing is wrong with the request itself."""


class Client:
    """Quarterly financial statements. Error texts never include the request, which holds the key."""

    def __init__(
        self,
        key: str,
        get: Callable[..., requests.Response] = requests.get,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._key = key
        self._get = get
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = MIN_INTERVAL - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def statement(self, function: str, ticker: str) -> dict:
        """The vendor's answer for one statement of one ticker, with its quarterly reports."""
        self._throttle()
        try:
            response = self._get(
                URL, params={"function": function, "symbol": ticker, "apikey": self._key}, timeout=30
            )
        except requests.RequestException as exc:
            raise AlphaVantageError(type(exc).__name__) from None
        if response.status_code != 200:
            raise AlphaVantageError(f"HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise AlphaVantageError("answer is not JSON") from None
        if not isinstance(body, dict):
            raise AlphaVantageError("unexpected answer")
        if body.get("quarterlyReports"):
            return body
        # Limits are reported with HTTP 200 and a message instead of data.
        if "Information" in body or "Note" in body:
            raise QuotaExceeded("rate or daily limit reached")
        raise AlphaVantageError(f"no {function} for {ticker}")
