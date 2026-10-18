"""Sample scored news for hand-labeling, and compare Jev's scores against the labels.

The review file never shows Jev's own scores (those live in a separate answer
file): labeling blind is what makes the comparison meaningful.
"""

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from sluice.news.models import StoredNews
from sluice.news.store import NewsStore
from sluice.scoring.config import ScoringConfig
from sluice.scoring.score import jobs_for_ticker
from sluice.scoring.similarity import most_similar
from sluice.scoring.store import ItemScores, ScoreStore

FIELDS = ("relevant", "stance", "novel", "rumour", "event_type")

_HEADER = """\
# Jev calibration review

Fill in every blank below from your own reading of the headline, summary and
(for the novelty question) the reference headlines. Do not open the answer
file next to this one until you are done labeling -- it holds Jev's own
scores, and knowing them while you label defeats the point of comparing.
"""


@dataclass(frozen=True)
class CalibrationCandidate:
    ticker: str
    item: StoredNews
    references: list[str]  # the novelty references Jev was shown
    scores: ItemScores  # Jev's actual answer, kept out of the review file


class HumanLabel(BaseModel):
    relevant: bool
    stance: Literal["bullish", "bearish", "neutral"]
    novel: bool
    rumour: bool
    event_type: str


def candidates(
    tickers: list[str], store: NewsStore, scores: ScoreStore, config: ScoringConfig
) -> list[CalibrationCandidate]:
    """Every already-scored item for `tickers`, with the novelty references it was shown.

    Mirrors `score_all`'s construction (one query per ticker, dedup across
    tickers) so the recomputed reference list matches what Jev actually saw.
    """
    now = datetime.now(UTC)
    seen: dict[int, CalibrationCandidate] = {}
    for ticker in tickers:
        for item, history in jobs_for_ticker(store.query(ticker, before=now, dedup=True)):
            if item.id in seen:
                continue
            item_scores = scores.get(item.id, config.version)
            if item_scores is None:
                continue  # not scored, or skipped as too large
            references = most_similar(item.title, history, config.novelty_reference_k) if history else []
            seen[item.id] = CalibrationCandidate(ticker, item, references, item_scores)
    return list(seen.values())


def stratified_sample(
    pool: list[CalibrationCandidate], n: int, seed: int = 0
) -> list[CalibrationCandidate]:
    """Up to `n` candidates, spread across Jev's event types as evenly as the pool allows.

    Round-robins over event-type buckets (each shuffled first), so a bucket
    running out early just lets the others fill the remaining budget.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[CalibrationCandidate]] = defaultdict(list)
    for c in pool:
        by_type[c.scores.event_type].append(c)
    for bucket in by_type.values():
        rng.shuffle(bucket)

    chosen: list[CalibrationCandidate] = []
    cursor = dict.fromkeys(by_type, 0)
    types = sorted(by_type)
    while len(chosen) < n and any(cursor[t] < len(by_type[t]) for t in types):
        for t in types:
            if len(chosen) >= n:
                break
            if cursor[t] < len(by_type[t]):
                chosen.append(by_type[t][cursor[t]])
                cursor[t] += 1
    return chosen


def build_review(chosen: list[CalibrationCandidate]) -> tuple[str, dict[int, ItemScores]]:
    """The blank review file's text, and the answer key kept out of it."""
    answers: dict[int, ItemScores] = {}
    blocks = []
    for i, c in enumerate(chosen, start=1):
        answers[c.item.id] = c.scores
        lines = [f"## [{i}] {c.ticker} — news_id={c.item.id}", f"Headline: {c.item.title}"]
        if c.item.summary:
            lines.append(f"Summary: {c.item.summary}")
        if c.references:
            lines.append("Recent headlines about the same company (for the novelty question):")
            lines.extend(f"- {h}" for h in c.references)
        lines += [
            "",
            "relevant (0 or 1):",
            "stance (bullish/bearish/neutral):",
            "novel (0 or 1):",
            "rumour (0 or 1):",
            "event_type (earnings/guidance/m_and_a/regulatory/contract_product/analyst_action/macro/other):",
        ]
        blocks.append("\n".join(lines))
    return _HEADER + "\n\n" + "\n\n".join(blocks) + "\n", answers


_BLOCK_HEADER_RE = re.compile(r"^## \[\d+\] (\S+) — news_id=(\d+)")
_FIELD_RES = {
    "relevant": re.compile(r"^relevant \(0 or 1\):\s*(\S*)", re.MULTILINE),
    "stance": re.compile(r"^stance \(bullish/bearish/neutral\):\s*(\S*)", re.MULTILINE),
    "novel": re.compile(r"^novel \(0 or 1\):\s*(\S*)", re.MULTILINE),
    "rumour": re.compile(r"^rumour \(0 or 1\):\s*(\S*)", re.MULTILINE),
    "event_type": re.compile(r"^event_type \([^)]*\):\s*(\S*)", re.MULTILINE),
}


def parse_review(text: str) -> dict[int, HumanLabel]:
    """The filled-in blanks, keyed by news_id. Raises ValueError naming the first blank left empty."""
    labels: dict[int, HumanLabel] = {}
    for block in re.split(r"\n(?=## \[)", text):
        header = _BLOCK_HEADER_RE.search(block)
        if not header:
            continue
        news_id = int(header.group(2))
        values: dict[str, str] = {}
        for field, pattern in _FIELD_RES.items():
            match = pattern.search(block)
            if not match or not match.group(1):
                raise ValueError(f"news_id={news_id}: '{field}' is still blank")
            values[field] = match.group(1)
        labels[news_id] = HumanLabel(
            relevant=values["relevant"] == "1",
            stance=values["stance"],  # type: ignore[arg-type]
            novel=values["novel"] == "1",
            rumour=values["rumour"] == "1",
            event_type=values["event_type"],
        )
    return labels


def noul_bins(pairs: list[tuple[float, bool]], bins: int = 5) -> list[tuple[float, float, int]]:
    """Reliability diagram data: (mean predicted probability, observed rate, count) per bin."""
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for predicted, observed in pairs:
        buckets[min(int(predicted * bins), bins - 1)].append((predicted, observed))
    return [
        (sum(p for p, _ in b) / len(b), sum(o for _, o in b) / len(b), len(b))
        for b in buckets
        if b
    ]


def choice_bins(
    triples: list[tuple[bool, float]], bins: int = 5
) -> list[tuple[float, float, int]]:
    """Reliability diagram data for a choice question: `triples` is (was Jev's pick correct?,
    Jev's confidence in that pick). Returns (mean confidence, accuracy, count) per bin.
    """
    buckets: list[list[tuple[bool, float]]] = [[] for _ in range(bins)]
    for correct, confidence in triples:
        buckets[min(int(confidence * bins), bins - 1)].append((correct, confidence))
    return [
        (sum(c for _, c in b) / len(b), sum(1 for ok, _ in b if ok) / len(b), len(b))
        for b in buckets
        if b
    ]
