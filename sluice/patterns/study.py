"""The pattern event study: what followed each pattern, month by month (task 10.6).

docs/specs/09-patterns.md ("已确认（任务 10.5、10.6）"). Each event is labelled
like the project's news: bought at the next open, sold at the close of the 5th
or 10th trading day, log return minus SPY's over the same days; it is settled
on that last day. The same pattern and stage on one stock within 10 trading
days counts once. A baseline over every valid stock-day of the same stocks
takes out what merely holding today's index members would have earned (they
are the survivors). A table is fitted as of each month's first day from the
events settled before it, kept once written, and read later by the tool as of
a decision time, like the news weight tables.
"""

import json
import math
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np

from sluice.patterns.detect import PatternEvent
from sluice.prices.store import Bar

HORIZONS = (5, 10)
COOLDOWN = 10  # trading days within which a repeat of the same pattern on a stock is dropped
MIN_EVENTS = 30  # fewer and a row is marked as too few
RESAMPLES = 1000
FIRST_TABLE = date(2017, 1, 1)
DEFAULT_ROOT = Path("data/local/pattern_stats")


@dataclass(frozen=True)
class Outcome:
    ticker: str
    pattern: str
    stage: str
    day: date  # the signal day
    horizon: int
    settled: date  # the close the label is known at
    excess: float  # log return minus SPY's, next open to the horizon's close


def _log_excess(bars: list[Bar], spy: dict[date, Bar], i: int, h: int) -> tuple[date, float] | None:
    """From the open after bar i to the close of bar i + h, minus SPY's; None without data."""
    if i + h >= len(bars):
        return None
    entry, exit_ = bars[i + 1], bars[i + h]
    s_entry, s_exit = spy.get(entry.day), spy.get(exit_.day)
    if s_entry is None or s_exit is None or min(entry.open, exit_.close) <= 0:
        return None
    own = math.log(exit_.close / entry.open)
    market = math.log(s_exit.close / s_entry.open)
    return exit_.day, own - market


def outcomes(
    ticker: str, bars: list[Bar], spy: dict[date, Bar], events: list[PatternEvent]
) -> list[Outcome]:
    """Every event's labels, dropping repeats of a pattern and stage within the cooldown."""
    index = {bar.day: i for i, bar in enumerate(bars)}
    last: dict[tuple[str, str], int] = {}
    out = []
    for event in sorted(events, key=lambda e: e.day):
        i = index.get(event.day)
        if i is None:
            continue
        key = (event.pattern, event.stage)
        if key in last and i - last[key] < COOLDOWN:
            continue
        last[key] = i
        for h in HORIZONS:
            labelled = _log_excess(bars, spy, i, h)
            if labelled is not None:
                settled, excess = labelled
                out.append(
                    Outcome(ticker, event.pattern, event.stage, event.day, h, settled, excess)
                )
    return out


@dataclass
class Baseline:
    """Sums of every stock-day's label by the month it settled in, to cumulate by table date."""

    sums: dict[tuple[int, str], float]
    counts: dict[tuple[int, str], int]

    @classmethod
    def empty(cls) -> "Baseline":
        return cls(defaultdict(float), defaultdict(int))

    def add(self, bars: list[Bar], spy: dict[date, Bar]) -> None:
        for i in range(len(bars)):
            if bars[i].volume <= 0:
                continue
            for h in HORIZONS:
                labelled = _log_excess(bars, spy, i, h)
                if labelled is not None:
                    settled, excess = labelled
                    month = settled.strftime("%Y-%m")
                    self.sums[(h, month)] += excess
                    self.counts[(h, month)] += 1

    def before(self, moment: date, h: int) -> tuple[float | None, int]:
        cut = moment.strftime("%Y-%m")
        keys = [k for k in self.counts if k[0] == h and k[1] < cut]
        n = sum(self.counts[k] for k in keys)
        return (sum(self.sums[k] for k in keys) / n if n else None), n


@dataclass(frozen=True)
class Row:
    pattern: str
    stage: str
    horizon: int
    n: int
    mean: float
    low: float  # 95% interval of the mean, by resampling signal days
    high: float
    hit: float  # share above zero
    baseline: float | None
    excess_over_baseline: float | None
    too_few: bool


def _resampled(values_by_day: dict[date, list[float]], seed: str) -> tuple[float, float]:
    """95% interval of the mean, resampling whole signal days (events of one day move together)."""
    sums = np.array([sum(v) for v in values_by_day.values()])
    counts = np.array([len(v) for v in values_by_day.values()])
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    picks = rng.integers(0, len(sums), size=(RESAMPLES, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def table(found: list[Outcome], baseline: Baseline, moment: date) -> list[Row]:
    """Every pattern, stage and horizon from what settled before `moment` (a month's first day)."""
    groups: dict[tuple[str, str, int], dict[date, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for o in found:
        if o.settled < moment:
            groups[(o.pattern, o.stage, o.horizon)][o.day].append(o.excess)
    rows = []
    for (pattern, stage, h), by_day in sorted(groups.items()):
        values = [v for day in by_day.values() for v in day]
        mean = sum(values) / len(values)
        low, high = _resampled(by_day, f"{pattern}|{stage}|{h}|{moment}")
        base, _ = baseline.before(moment, h)
        rows.append(
            Row(
                pattern=pattern, stage=stage, horizon=h, n=len(values), mean=mean,
                low=low, high=high, hit=sum(v > 0 for v in values) / len(values),
                baseline=base,
                excess_over_baseline=None if base is None else mean - base,
                too_few=len(values) < MIN_EVENTS,
            )
        )  # fmt: skip
    return rows


def month_starts(first: date, last: date) -> list[date]:
    months, day = [], date(first.year, first.month, 1)
    while day <= last:
        months.append(day)
        day = date(day.year + day.month // 12, day.month % 12 + 1, 1)
    return months


@dataclass(frozen=True)
class StoredStats:
    as_of: date
    config: str  # "<version>-<digest>" of the pattern rules
    universe: str
    rows: list[Row]


def save(stats: StoredStats, root: Path = DEFAULT_ROOT) -> Path:
    folder = root / stats.config
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stats.as_of}.json"
    record = {
        "as_of": stats.as_of.isoformat(), "config": stats.config, "universe": stats.universe,
        "rows": [asdict(r) for r in stats.rows],
    }  # fmt: skip
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return path


def load_all(config: str, root: Path = DEFAULT_ROOT) -> list[StoredStats]:
    folder = root / config
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        raw = json.loads(path.read_text())
        out.append(
            StoredStats(
                as_of=date.fromisoformat(raw["as_of"]), config=raw["config"],
                universe=raw["universe"], rows=[Row(**r) for r in raw["rows"]],
            )
        )  # fmt: skip
    return out


def latest_before(tables: list[StoredStats], moment: date) -> StoredStats | None:
    """The table fitted as of the latest month start not after `moment`."""
    usable = [t for t in tables if t.as_of <= moment]
    return max(usable, key=lambda t: t.as_of) if usable else None
