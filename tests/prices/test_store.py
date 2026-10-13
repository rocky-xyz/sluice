from datetime import UTC, date, datetime

import pytest

from sluice.prices.store import HISTORY_START, PriceStore
from tests.prices.series import series, trading_days

LAST = date(2026, 9, 30)
AFTER_CLOSE = datetime(2026, 9, 30, 21, 0, tzinfo=UTC)
MIDDAY = datetime(2026, 9, 30, 16, 0, tzinfo=UTC)  # noon Eastern, the session still open


class Vendor:
    def __init__(self, bars):
        self.bars = bars
        self.calls = []

    def __call__(self, ticker, start):
        self.calls.append((ticker, start))
        if isinstance(self.bars, Exception):
            raise self.bars
        return self.bars


def store(tmp_path, vendor=None, now=AFTER_CLOSE):
    return PriceStore(tmp_path / "prices.db", fetch=vendor, now=lambda: now)


def test_history_ends_at_the_requested_day_whatever_is_stored(tmp_path):
    with store(tmp_path) as prices:
        prices.replace("NVDA", series(LAST, 30))
        days = trading_days(LAST, 30)

        got = prices.history("NVDA", days[-3], limit=5)
        assert [bar.day for bar in got] == days[-7:-2]
        assert prices.history("NVDA", date(2020, 1, 1), limit=5) == []
        assert prices.latest("NVDA") == LAST


def test_a_missing_ticker_is_fetched_once(tmp_path):
    vendor = Vendor(series(LAST, 30))
    with store(tmp_path, vendor) as prices:
        assert len(prices.history("nvda", LAST, limit=250)) == 30
        assert len(prices.history("NVDA", LAST, limit=10)) == 10
        assert len(prices.history("NVDA", trading_days(LAST, 5)[0], limit=250)) == 26
    assert vendor.calls == [("NVDA", HISTORY_START)]


def test_a_stale_series_is_fetched_again_and_replaced_whole(tmp_path):
    with store(tmp_path) as prices:
        prices.replace("NVDA", series(trading_days(LAST, 2)[0], 30, start=100.0))

    # after a split every adjusted close is different
    vendor = Vendor(series(LAST, 31, start=10.0))
    with store(tmp_path, vendor) as prices:
        got = prices.history("NVDA", LAST, limit=250)
    assert len(vendor.calls) == 1
    assert len(got) == 31
    assert got[0].close == pytest.approx(vendor.bars[0].close)


def test_a_session_still_open_when_fetched_is_not_stored(tmp_path):
    vendor = Vendor(series(LAST, 30))
    with store(tmp_path, vendor, now=MIDDAY) as prices:
        got = prices.history("NVDA", LAST, limit=250)
        assert got[-1].day == trading_days(LAST, 2)[0]
        assert prices.latest("NVDA") == trading_days(LAST, 2)[0]


def test_an_unknown_ticker_has_no_history(tmp_path):
    with store(tmp_path, Vendor([])) as prices:
        assert prices.history("ZZZZ", LAST, limit=250) == []


def test_a_vendor_failure_is_raised_not_read_as_no_prices(tmp_path):
    with store(tmp_path, Vendor(ConnectionError("down"))) as prices, pytest.raises(ConnectionError):
        prices.history("NVDA", LAST, limit=250)


def test_without_a_vendor_only_stored_bars_are_read(tmp_path):
    with store(tmp_path) as prices:
        assert prices.history("NVDA", LAST, limit=250) == []
