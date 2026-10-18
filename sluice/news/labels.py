"""Per-item labels for the news event study (docs/specs/03-news-weights.md).

Entry is the next trading session's open after a news item's publish time; the
label is a horizon's log excess return over the benchmark, with the entry day
itself counted as day 1. A label is None whenever its window has not fully
closed yet, or a needed bar is missing.
"""

import math
import statistics
from datetime import datetime

from sluice.market_calendar import close_of, next_open_after, nth_trading_day
from sluice.prices.store import PriceStore

BENCHMARK = "SPY"


def item_label(
    tickers: list[str],
    published_at: datetime,
    horizon_days: int,
    prices: PriceStore,
    now: datetime,
) -> float | None:
    """A news item's label: the equal-weighted average of `tickers`' own labels.

    One news item can be tagged to several tickers (e.g. a macro story), but
    gets a single Jev score that never names which one it is about (see
    docs/specs/03-news-weights.md, "已确认"). Averaging keeps one item one
    observation instead of turning a shared score into several independent
    ones. `tickers` is the caller's choice of which of the item's tags to use
    (normally intersected with whatever universe it is studying); unresolved
    or unknown for any of them makes the whole average None.
    """
    if not tickers:
        return None
    per_ticker = [excess_return_label(t, published_at, horizon_days, prices, now) for t in tickers]
    if any(label is None for label in per_ticker):
        return None
    return statistics.fmean(per_ticker)


def excess_return_label(
    ticker: str,
    published_at: datetime,
    horizon_days: int,
    prices: PriceStore,
    now: datetime,
) -> float | None:
    """Log excess return of `ticker` over `BENCHMARK`, `horizon_days` trading days out."""
    entry_day = next_open_after(published_at)
    exit_day = nth_trading_day(entry_day, horizon_days)
    exit_close_time = close_of(exit_day)
    if exit_close_time is None or exit_close_time > now:
        return None  # the window has not finished as of `now`
    stock = _log_return(ticker, entry_day, exit_day, prices)
    benchmark = _log_return(BENCHMARK, entry_day, exit_day, prices)
    if stock is None or benchmark is None:
        return None
    return stock - benchmark


def _log_return(ticker: str, entry_day, exit_day, prices: PriceStore) -> float | None:
    entry = prices.get(ticker, entry_day)
    exit_ = prices.get(ticker, exit_day)
    if entry is None or exit_ is None:
        return None
    return math.log(exit_.close / entry.open)
