"""What `get_patterns` tells an analyst: recent pattern events and how they fared (task 10.7).

docs/specs/09-patterns.md. The events come from the rules (`detect`) run on the
bars that had closed by the decision time, so nothing later shows; the figures
come from the latest monthly table fitted by then (`study`), which holds only
events settled before its date. Bars are read from the pattern study's store,
which has highs and lows; a ticker it lacks (TSM is no S&P 500 member) is
fetched into it on first use. The pipeline's own price store is never touched.

The tool is built and tested here but offered to no analyst yet (task 10.8
decides): `tools.ROLE_TOOLS` does not list it.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sluice.agents.state import PatternItem
from sluice.market_calendar import last_closed_session
from sluice.news.timeutil import ET
from sluice.patterns.candles import candles, weekly
from sluice.patterns.config import PatternConfig, load_pattern_config
from sluice.patterns.detect import PatternEvent, detect
from sluice.patterns.prices import DEFAULT_DB, START
from sluice.patterns.study import Row, StoredStats, latest_before, load_all
from sluice.prices.store import PriceStore

BARS = 400  # enough for a 120-day wedge, a 60-day average and weeks of context
MAX_DAYS = 60  # how far back get_patterns may look for events
NAMES = {
    "cuorou": "cuorou (long upper shadow, then long lower shadow, on shrinking volume)",
    "harami_daily": "harami, daily (a small bar inside a big bearish one or a long lower shadow)",
    "harami_weekly": "harami, weekly (the same on weekly bars, after bearish weeks)",
    "wedge": "falling wedge breakout",
}


@dataclass
class PatternSource:
    store: PriceStore  # with highs and lows, and a fetcher for a ticker it lacks
    tables: list[StoredStats]
    config: PatternConfig

    @classmethod
    def open(cls, version: str = "v1", db: Path = DEFAULT_DB) -> "PatternSource":
        from sluice.prices.yahoo import fetch_bars

        config = load_pattern_config(version)
        store = PriceStore(db, fetch=fetch_bars, history_start=START)
        return cls(store, load_all(f"{config.version}-{config.digest}"), config)


def ref(event: PatternEvent) -> str:
    return f"P:pat_{event.pattern}_{event.stage}_{event.day:%Y%m%d}"


def _stats(rows: list[Row], event: PatternEvent) -> str:
    mine = sorted(
        (r for r in rows if r.pattern == event.pattern and r.stage == event.stage),
        key=lambda r: r.horizon,
    )
    if not mine:
        return "no history yet for this pattern"
    parts = []
    for r in mine:
        if r.excess_over_baseline is None or r.baseline is None:
            continue
        low, high = r.low - r.baseline, r.high - r.baseline
        parts.append(
            f"{r.horizon} days after: {r.excess_over_baseline:+.2%} against the average "
            f"stock-day (95% interval {low:+.2%} to {high:+.2%}), up {r.hit:.0%} of the time"
        )
    n = mine[0].n
    caution = "; too few to rely on" if n < 30 else ""
    clear = any(
        r.baseline is not None and (r.low - r.baseline > 0 or r.high - r.baseline < 0) for r in mine
    )
    verdict = "" if clear else "; no reliable edge either way"
    return f"{n} times before; " + "; ".join(parts) + caution + verdict


PRICES = {"day1_high", "day2_low", "mother_high", "mother_low", "breakout_level"}


def _describe(event: PatternEvent, last_close: float) -> str:
    """In words, with prices as distances from the last close: no price level is shown
    (docs/specs/07: a level could be matched against what the model remembers)."""
    stage = {"signal": "", "candidate": ", breakout (unconfirmed)", "confirmed": ", confirmed"}
    how = f" by {event.how}" if event.how else ""
    shown = []
    for key, value in event.levels.items():
        name = key.replace("_", " ")
        if key in PRICES:
            shown.append(f"{name} {value / last_close - 1:+.1%} from the last close")
        elif key == "days":
            shown.append(f"{value:.0f} trading days long")
        else:
            shown.append(f"{name} {value:.2f}x")
    return f"{NAMES[event.pattern]}{stage[event.stage]}{how}, on {event.day}; {', '.join(shown)}"


def recent(
    source: PatternSource, ticker: str, decision_time: datetime, days: int, names: list[str]
) -> tuple[list[str], list[PatternItem]]:
    """Lines for the model and the items cited, for events in the last `days` sessions."""
    through = last_closed_session(decision_time)
    bars = source.store.history(ticker, through, BARS)
    if len(bars) < 2:
        return ["Missing: no daily bars with highs and lows for this stock."], []
    spy = candles(source.store.history("SPY", through, BARS))
    events = detect(candles(bars), candles(weekly(bars)), {c.day: c for c in spy}, source.config)
    since = bars[max(0, len(bars) - days)].day
    chosen = [e for e in events if e.day >= since and (not names or e.pattern in names)]
    table = latest_before(source.tables, decision_time.astimezone(ET).date())
    rows = table.rows if table else []
    head = (
        f"Pattern events in the last {days} trading days to {through}"
        + (f"; history from the table of {table.as_of}" if table else "; no history table yet")
        + ":"
    )
    if not chosen:
        return [head, "None."], []
    lines, items = [head], []
    for event in chosen:
        described = _describe(event, bars[-1].close)
        lines.append(f"[{ref(event)}] {described}. History: {_stats(rows, event)}.")
        items.append(
            PatternItem(
                ref=ref(event), pattern=event.pattern, stage=event.stage, day=event.day,
                how=event.how, levels=event.levels,
            )
        )  # fmt: skip
    return lines, items
