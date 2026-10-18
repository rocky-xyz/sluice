import math
from datetime import UTC, date, datetime

import pytest

from sluice.news.labels import excess_return_label, item_label
from sluice.news.timeutil import ET
from sluice.prices.store import Bar, PriceStore

PUBLISHED = datetime(2026, 9, 28, 11, 0, tzinfo=ET)  # Monday, mid-session
ENTRY_DAY = date(2026, 9, 29)  # next trading day
EXIT_DAY = date(2026, 10, 1)  # 3rd trading day counting the entry day as day 1
AFTER_EXIT_CLOSE = datetime(2026, 10, 1, 21, 0, tzinfo=UTC)  # after the exit day's close


def make_store(tmp_path, bars: dict[tuple[str, date], tuple[float, float]]) -> PriceStore:
    """bars maps (ticker, day) -> (open, close); raw_close and volume are unused here."""
    store = PriceStore(tmp_path / "prices.db")
    by_ticker: dict[str, list[Bar]] = {}
    for (ticker, day), (open_, close) in bars.items():
        by_ticker.setdefault(ticker, []).append(
            Bar(day=day, open=open_, close=close, raw_close=close, volume=1.0)
        )
    for ticker, bar_list in by_ticker.items():
        store.replace(ticker, sorted(bar_list, key=lambda b: b.day))
    return store


def test_excess_return_is_the_log_return_gap_over_the_benchmark(tmp_path):
    prices = make_store(
        tmp_path,
        {
            ("NVDA", ENTRY_DAY): (100.0, 0.0),
            ("NVDA", EXIT_DAY): (0.0, 110.0),
            ("SPY", ENTRY_DAY): (400.0, 0.0),
            ("SPY", EXIT_DAY): (0.0, 404.0),
        },
    )
    label = excess_return_label("NVDA", PUBLISHED, 3, prices, now=AFTER_EXIT_CLOSE)
    assert label == pytest.approx(math.log(110 / 100) - math.log(404 / 400))


def test_label_is_none_before_the_window_closes(tmp_path):
    prices = make_store(
        tmp_path,
        {
            ("NVDA", ENTRY_DAY): (100.0, 0.0),
            ("NVDA", EXIT_DAY): (0.0, 110.0),
            ("SPY", ENTRY_DAY): (400.0, 0.0),
            ("SPY", EXIT_DAY): (0.0, 404.0),
        },
    )
    still_open = datetime(2026, 10, 1, 19, 0, tzinfo=UTC)  # before the 20:00 UTC close
    assert excess_return_label("NVDA", PUBLISHED, 3, prices, now=still_open) is None


def test_label_is_none_when_a_bar_is_missing(tmp_path):
    prices = make_store(
        tmp_path,
        {
            ("NVDA", ENTRY_DAY): (100.0, 0.0),
            # NVDA's exit-day bar is missing.
            ("SPY", ENTRY_DAY): (400.0, 0.0),
            ("SPY", EXIT_DAY): (0.0, 404.0),
        },
    )
    assert excess_return_label("NVDA", PUBLISHED, 3, prices, now=AFTER_EXIT_CLOSE) is None


def test_item_label_averages_across_its_tickers(tmp_path):
    prices = make_store(
        tmp_path,
        {
            ("NVDA", ENTRY_DAY): (100.0, 0.0),
            ("NVDA", EXIT_DAY): (0.0, 110.0),  # +10%
            ("AMD", ENTRY_DAY): (50.0, 0.0),
            ("AMD", EXIT_DAY): (0.0, 65.0),  # +30%
            ("SPY", ENTRY_DAY): (400.0, 0.0),
            ("SPY", EXIT_DAY): (0.0, 400.0),  # flat
        },
    )
    label = item_label(["NVDA", "AMD"], PUBLISHED, 3, prices, now=AFTER_EXIT_CLOSE)
    nvda = math.log(110 / 100)
    amd = math.log(65 / 50)
    assert label == pytest.approx((nvda + amd) / 2)


def test_item_label_is_none_if_any_ticker_is_unresolved(tmp_path):
    prices = make_store(
        tmp_path,
        {
            ("NVDA", ENTRY_DAY): (100.0, 0.0),
            ("NVDA", EXIT_DAY): (0.0, 110.0),
            # AMD's exit-day bar is missing: its own label is None.
            ("AMD", ENTRY_DAY): (50.0, 0.0),
            ("SPY", ENTRY_DAY): (400.0, 0.0),
            ("SPY", EXIT_DAY): (0.0, 400.0),
        },
    )
    assert item_label(["NVDA", "AMD"], PUBLISHED, 3, prices, now=AFTER_EXIT_CLOSE) is None


def test_item_label_of_no_tickers_is_none(tmp_path):
    prices = make_store(tmp_path, {})
    assert item_label([], PUBLISHED, 3, prices, now=AFTER_EXIT_CLOSE) is None
