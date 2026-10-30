"""What one command spent, in dollars (docs/specs/12, task 13.3).

Only calls made by this command count: a `Tally` hears every answer the LLM
and Jev caches hand out (sluice.agents.events) and keeps the cached ones
apart, so a batch that resumes, or a run already in its checkpoint, adds
nothing it did not pay for now.

DeepSeek charges by the hour in Beijing: list price on weekdays 9:00-12:00 and
14:00-18:00, half price otherwise. China's public holidays are off-peak too
but cannot be known offline, so they count as weekdays here and are
overestimated. Input tokens are all priced as missing DeepSeek's own prompt
cache, whose hits are cheaper; that also overestimates a little.
"""

import threading
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from sluice.agents.events import Event, JevCalled, LLMCalled

BEIJING = ZoneInfo("Asia/Shanghai")
# Dollars per million tokens (input, output) at peak hours, from DeepSeek's price list on
# 2026-10-14 (moved here from sluice.agents.batch).
PEAK_PRICES = {"deepseek-flash": (0.3, 1.2), "deepseek-v4-pro": (1.32, 3.96)}
OFF_PEAK = 0.5  # share of the list price outside peak hours
PEAK_HOURS = ((time(9), time(12)), (time(14), time(18)))  # Beijing, Monday to Friday
# Jev bills input tokens only: 132.6M of them cost $5.57 in the full scoring (ADR 018).
JEV_PER_MILLION = 5.57 / 132.6


def is_peak(moment: datetime) -> bool:
    local = moment.astimezone(BEIJING)
    return local.weekday() < 5 and any(start <= local.time() < end for start, end in PEAK_HOURS)


def peak_share(start: datetime, end: datetime) -> float:
    """The share of [start, end] that falls in peak hours; at one instant, 1 or 0."""
    if end <= start:
        return 1.0 if is_peak(start) else 0.0
    first, last = start.astimezone(BEIJING), end.astimezone(BEIJING)
    overlap = timedelta()
    day = first.date()
    while day <= last.date():
        if day.weekday() < 5:
            for open_, close in PEAK_HOURS:
                lo = max(first, datetime.combine(day, open_, BEIJING))
                hi = min(last, datetime.combine(day, close, BEIJING))
                overlap += max(hi - lo, timedelta())
        day += timedelta(days=1)
    return overlap / (last - first)


@dataclass
class Usage:
    new_calls: int = 0
    cached_calls: int = 0
    new_in: int = 0  # tokens of the calls made now: the ones paid for
    new_out: int = 0
    all_in: int = 0  # tokens of every answer handed out, cached ones included
    all_out: int = 0

    def add(self, input_tokens: int, output_tokens: int, cached: bool) -> None:
        if cached:
            self.cached_calls += 1
        else:
            self.new_calls += 1
            self.new_in += input_tokens
            self.new_out += output_tokens
        self.all_in += input_tokens
        self.all_out += output_tokens


class Tally:
    """Calls and tokens of one command, from the cache events of any thread.

    Pass it as the `on_event` of `CachedLLM` and `CachedJev`; read it with `snapshot`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm: dict[str, Usage] = {}
        self._jev = Usage()

    def __call__(self, event: Event) -> None:
        with self._lock:
            if isinstance(event, LLMCalled):
                usage = self._llm.setdefault(event.model, Usage())
                usage.add(event.input_tokens, event.output_tokens, event.cached)
            elif isinstance(event, JevCalled):
                self._jev.add(event.input_tokens, event.output_tokens, event.cached)

    def snapshot(self) -> tuple[dict[str, Usage], Usage]:
        """Copies: per LLM model, and Jev."""
        with self._lock:
            llm = {model: Usage(**vars(u)) for model, u in sorted(self._llm.items())}
            return llm, Usage(**vars(self._jev))


@dataclass(frozen=True)
class Spend:
    low: float  # every DeepSeek call at the cheapest price the command's hours allow
    high: float
    share: float  # of the command's time in peak hours
    unpriced: tuple[str, ...] = ()  # models without a price; left out of the sum

    def describe(self) -> str:
        if self.low == self.high:  # one price applied, or nothing was spent
            when = {1.0: " (peak)", 0.0: " (off-peak)"}.get(self.share, "")
            text = f"about ${self.high:.2f}{when}"
        else:
            text = f"${self.low:.2f} to ${self.high:.2f} (ran across peak and off-peak hours)"
        if self.unpriced:
            text += f"; no price for {', '.join(self.unpriced)}"
        return text


def spend(tally: Tally, start: datetime, end: datetime) -> Spend:
    """Dollars for the calls made between `start` and `end`, DeepSeek by the hour, Jev flat."""
    llm, jev = tally.snapshot()
    peak = sum(
        (u.new_in * PEAK_PRICES[m][0] + u.new_out * PEAK_PRICES[m][1]) / 1_000_000
        for m, u in llm.items()
        if m in PEAK_PRICES
    )
    flat = jev.new_in * JEV_PER_MILLION / 1_000_000
    unpriced = tuple(m for m, u in llm.items() if m not in PEAK_PRICES and u.new_calls)
    share = peak_share(start, end)
    low = peak * (1 if share == 1 else OFF_PEAK) + flat
    high = peak * (OFF_PEAK if share == 0 else 1) + flat
    return Spend(low, high, share, unpriced)
