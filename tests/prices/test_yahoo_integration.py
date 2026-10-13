from datetime import date

import pytest

from sluice.prices.yahoo import fetch_bars

pytestmark = pytest.mark.integration


def test_real_bars_and_an_unknown_ticker():
    bars = fetch_bars("SPY", date(2026, 9, 1))
    assert bars[0].day == date(2026, 9, 1)
    assert all(bar.close > 0 and bar.volume > 0 for bar in bars)
    assert [bar.day for bar in bars] == sorted({bar.day for bar in bars})

    assert fetch_bars("ZZZZQQX", date(2026, 9, 1)) == []
