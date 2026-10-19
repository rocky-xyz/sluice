"""Decision quality of pipeline ratings, apart from any portfolio (docs/specs/05, tasks 6.1, 6.7).

The label of a decision is the stock's log excess return over SPY from the open
after the decision day through the close of the 10th trading day, the same as
the A/B of task 5.3. Rating scores are buy +2 .. sell -2.
- Hit rate: among ratings with a direction (not hold), the share whose sign
  matches the label's.
- Bullish / bearish share: of the successful runs.
- Calibration: the research manager's confidence in buckets against the hit
  rate in each; and each rating level against its mean label.
- Consistency: over cells whose every repeat succeeded, the share whose
  ratings all agree, and the mean standard deviation of the rating score.
"""

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise

from sluice.agents.prefetch import backtest_decision_time
from sluice.evaluation.decisions import Decision
from sluice.news.labels import excess_return_label
from sluice.prices.store import PriceStore

HORIZON = 10  # trading days: one decision period
SCORES = {"buy": 2, "overweight": 1, "hold": 0, "underweight": -1, "sell": -2}
CONFIDENCE_EDGES = (0.5, 0.6, 0.7)  # buckets: below 0.5, 0.5-0.6, 0.6-0.7, 0.7 and above


@dataclass(frozen=True)
class Labelled:
    decision: Decision
    label: float | None  # None until the window has closed, or without prices


def label_decisions(
    decisions: list[Decision], prices: PriceStore, now: datetime
) -> list[Labelled]:
    return [
        Labelled(
            d,
            excess_return_label(d.ticker, backtest_decision_time(d.day), HORIZON, prices, now),
        )
        for d in decisions
    ]


def _rank(values: list[float]) -> list[float]:
    """Average ranks, ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


@dataclass(frozen=True)
class Summary:
    runs: int
    failed: int
    labelled: int  # successful runs whose label is known
    bullish: float | None  # share of successful runs rated buy or overweight
    bearish: float | None
    directional: int
    hit_rate: float | None
    spearman: float | None  # rating score against label, hold included


def summary(rows: list[Labelled]) -> Summary:
    ok = [r for r in rows if r.decision.rating is not None]
    known = [r for r in ok if r.label is not None]
    pairs = [(SCORES[r.decision.rating], r.label) for r in known]
    directional = [(s, y) for s, y in pairs if s != 0]
    spearman = None
    if len(pairs) >= 3 and len({s for s, _ in pairs}) > 1 and len({y for _, y in pairs}) > 1:
        scores, labels = [s for s, _ in pairs], [y for _, y in pairs]
        spearman = statistics.correlation(_rank(scores), _rank(labels))
    return Summary(
        runs=len(rows),
        failed=len(rows) - len(ok),
        labelled=len(known),
        bullish=sum(SCORES[r.decision.rating] > 0 for r in ok) / len(ok) if ok else None,
        bearish=sum(SCORES[r.decision.rating] < 0 for r in ok) / len(ok) if ok else None,
        directional=len(directional),
        hit_rate=(
            sum((s > 0) == (y > 0) for s, y in directional) / len(directional)
            if directional
            else None
        ),
        spearman=spearman,
    )


@dataclass(frozen=True)
class Bucket:
    name: str
    n: int
    mean_confidence: float | None
    hit_rate: float | None


def confidence_calibration(rows: list[Labelled]) -> list[Bucket]:
    """Directional, labelled runs bucketed by stated confidence: is 0.7 right 70% of the time?"""
    names = [f"< {CONFIDENCE_EDGES[0]}"]
    names += [f"{a}–{b}" for a, b in pairwise(CONFIDENCE_EDGES)]
    names += [f"≥ {CONFIDENCE_EDGES[-1]}"]
    grouped: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for r in rows:
        d = r.decision
        if d.rating is None or d.confidence is None or r.label is None or SCORES[d.rating] == 0:
            continue
        index = sum(d.confidence >= edge for edge in CONFIDENCE_EDGES)
        grouped[index].append((d.confidence, (SCORES[d.rating] > 0) == (r.label > 0)))
    out = []
    for index, name in enumerate(names):
        items = grouped.get(index, [])
        out.append(
            Bucket(
                name=name,
                n=len(items),
                mean_confidence=statistics.fmean(c for c, _ in items) if items else None,
                hit_rate=sum(hit for _, hit in items) / len(items) if items else None,
            )
        )
    return out


def rating_calibration(rows: list[Labelled]) -> list[tuple[str, int, float | None]]:
    """Each rating level, in order: how many runs, and their mean label."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.decision.rating is not None and r.label is not None:
            grouped[r.decision.rating].append(r.label)
    return [
        (rating, len(labels), statistics.fmean(labels) if labels else None)
        for rating in SCORES
        for labels in [grouped[rating]]
    ]


@dataclass(frozen=True)
class Consistency:
    cells: int  # cells with more than one repeat, all successful
    agree: float | None  # share whose ratings are all the same
    mean_std: float | None  # mean, over those cells, of the rating score's standard deviation


def consistency(decisions: list[Decision]) -> Consistency:
    cells: dict[tuple[str, date], list[Decision]] = defaultdict(list)
    for d in decisions:
        cells[(d.ticker, d.day)].append(d)
    full = [c for c in cells.values() if len(c) > 1 and all(d.rating is not None for d in c)]
    if not full:
        return Consistency(cells=0, agree=None, mean_std=None)
    return Consistency(
        cells=len(full),
        agree=sum(len({d.rating for d in c}) == 1 for c in full) / len(full),
        mean_std=statistics.fmean(
            math.sqrt(statistics.pvariance([SCORES[d.rating] for d in c])) for c in full
        ),
    )
