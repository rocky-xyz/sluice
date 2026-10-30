import threading
from datetime import UTC, datetime, timedelta

import pytest

from sluice.agents.events import JevCalled, LLMCalled, NodeStarted
from sluice.costs import (
    BEIJING,
    JEV_PER_MILLION,
    Spend,
    Tally,
    Usage,
    is_peak,
    peak_share,
    spend,
)


def beijing(day: int, hour: int, minute: int = 0) -> datetime:
    """A moment in October 2026, Beijing time: the 30th is a Friday, the 31st a Saturday."""
    return datetime(2026, 10, day, hour, minute, tzinfo=BEIJING)


def flash(input_tokens: int, output_tokens: int, cached: bool = False) -> LLMCalled:
    return LLMCalled("news_analyst", "deepseek-flash", input_tokens, output_tokens, cached)


@pytest.mark.parametrize("moment, peak", [
    (beijing(30, 10), True),
    (beijing(30, 12, 30), False),  # lunch
    (beijing(30, 17, 59), True),
    (beijing(30, 18), False),
    (beijing(31, 10), False),  # Saturday
    (beijing(30, 2).astimezone(UTC), False),  # a UTC moment is read in Beijing time
])  # fmt: skip
def test_peak_hours_are_weekday_mornings_and_afternoons_in_beijing(moment, peak):
    assert is_peak(moment) is peak


def test_the_peak_share_counts_only_the_peak_windows_between_two_moments():
    assert peak_share(beijing(30, 11), beijing(30, 13)) == pytest.approx(0.5)
    assert peak_share(beijing(30, 12, 10), beijing(30, 13, 50)) == 0  # inside lunch
    assert peak_share(beijing(30, 8), beijing(30, 13)) == pytest.approx(3 / 5)  # ends off-peak
    assert peak_share(beijing(30, 9, 30), beijing(30, 11)) == 1
    assert peak_share(beijing(30, 17), beijing(31, 10)) == pytest.approx(1 / 17)  # into Saturday
    assert peak_share(beijing(30, 10), beijing(30, 10)) == 1  # an instant


def tally(*events) -> Tally:
    t = Tally()
    for event in events:
        t(event)
    return t


def test_only_calls_made_now_are_paid_for_at_the_hour_they_ran():
    t = tally(flash(1_000_000, 500_000), flash(9_000_000, 9_000_000, cached=True))
    peak = 0.3 + 0.5 * 1.2
    at_peak = spend(t, beijing(30, 10), beijing(30, 10, 5))
    assert (at_peak.low, at_peak.high, at_peak.share) == (pytest.approx(peak),) * 2 + (1.0,)
    off = spend(t, beijing(31, 10), beijing(31, 11))
    assert (off.low, off.high, off.share) == (pytest.approx(peak / 2),) * 2 + (0.0,)
    across = spend(t, beijing(30, 11, 30), beijing(30, 12, 30))
    assert (across.low, across.high) == (pytest.approx(peak / 2), pytest.approx(peak))
    assert across.share == pytest.approx(0.5)


def test_jev_bills_new_input_tokens_at_one_rate_whatever_the_hour():
    t = tally(JevCalled(1_000_000, 50, cached=False), JevCalled(5_000_000, 0, cached=True))
    assert spend(t, beijing(30, 10), beijing(30, 10)).high == pytest.approx(JEV_PER_MILLION)
    assert spend(t, beijing(31, 10), beijing(31, 10)).low == pytest.approx(JEV_PER_MILLION)
    assert JEV_PER_MILLION == pytest.approx(0.042, abs=0.001)


def test_a_model_without_a_price_is_named_not_guessed():
    t = tally(LLMCalled(None, "deepseek-next", 100, 100, cached=False))
    result = spend(t, beijing(31, 10), beijing(31, 10))
    assert (result.high, result.unpriced) == (0.0, ("deepseek-next",))
    assert "no price for deepseek-next" in result.describe()


def test_the_tally_keeps_cached_calls_apart_and_ignores_other_events():
    t = tally(flash(100, 20), flash(300, 40, cached=True), NodeStarted("bull"))
    llm, jev = t.snapshot()
    assert llm == {"deepseek-flash": Usage(1, 1, 100, 20, 400, 60)}
    assert jev == Usage()


def test_the_tally_loses_nothing_across_threads():
    t = Tally()
    workers = [
        threading.Thread(target=lambda: [t(flash(1, 1)) for _ in range(1000)]) for _ in range(8)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert t.snapshot()[0]["deepseek-flash"].new_calls == 8000


def test_the_snapshot_is_a_copy():
    t = tally(flash(1, 1))
    llm, _ = t.snapshot()
    llm["deepseek-flash"].new_calls = 99
    assert t.snapshot()[0]["deepseek-flash"].new_calls == 1


@pytest.mark.parametrize("result, text", [
    (Spend(0.41, 0.41, 0.0), "about $0.41 (off-peak)"),
    (Spend(0.0, 0.0, 0.5), "about $0.00"),  # nothing new across hours
    (Spend(0.82, 0.82, 1.0), "about $0.82 (peak)"),
    (Spend(0.41, 0.82, 0.3), "$0.41 to $0.82 (ran across peak and off-peak hours)"),
])  # fmt: skip
def test_the_spend_reads_as_one_line(result, text):
    assert result.describe() == text


def test_timedelta_arithmetic_crosses_days_in_beijing_not_utc():
    # 01:00 to 23:00 Beijing on a Friday is one day there but two in UTC.
    start = beijing(30, 1)
    assert peak_share(start, start + timedelta(hours=22)) == pytest.approx(7 / 22)
