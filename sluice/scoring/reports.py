"""Jev's stance on the pipeline's analyst reports (task 6.10, docs/specs/04-sizing.md).

The sizing formula's report score for a run is the mean, over its analyst
reports that had data, of P(bullish) - P(bearish). Scores are stored by run id,
report and spec version, so sizing reads them without asking Jev, and a rerun
only asks about what is missing. Like every reader of run states, this one
picks files by the decision day in their name and opens only those of the
requested segment.
"""

import json
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from sluice.evaluation.split import Segment, require_open, segment
from sluice.scoring.client import CachedJev, ChoiceAnswer, JevError, JevQuestion, JevRequest
from sluice.scoring.config import ReportScoringConfig

DEFAULT_PATH = Path("data/local/report_scores.db")
REPORT_NAMES = {
    "market": "market analyst",
    "news": "news analyst",
    "fundamentals": "fundamentals analyst",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS report_scores (
    run_id          TEXT NOT NULL,
    report          TEXT NOT NULL,  -- market / news / fundamentals
    config_version  TEXT NOT NULL,
    config_digest   TEXT NOT NULL,
    skipped         TEXT,           -- NULL when scored; else why it never will be
    stance          TEXT,
    probabilities   TEXT,           -- JSON {choice: probability}
    confidence      REAL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    scored_at       TEXT NOT NULL,
    PRIMARY KEY (run_id, report, config_version)
);
"""


class ReportText(BaseModel):
    run_id: str
    ticker: str
    day: date
    repeat: int
    report: str  # market / news / fundamentals
    text: str


def load_reports(folder: Path, requested: Segment, final: bool = False) -> list[ReportText]:
    """Every report with data in the runs of one config decided in `requested`."""
    require_open(requested, final)
    found = []
    for path in sorted(folder.glob("*/*.json")):
        day_text, repeat = path.stem.split("_r")
        day = date.fromisoformat(day_text)
        if segment(day) != requested:
            continue
        state = json.loads(path.read_text())
        for name, report in state["reports"].items():
            if report.get("no_data") or not report.get("text", "").strip():
                continue
            found.append(
                ReportText(
                    run_id=state["meta"]["run_id"],
                    ticker=path.parent.name,
                    day=day,
                    repeat=int(repeat),
                    report=name,
                    text=report["text"],
                )
            )
    return found


class ReportScoreStore:
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Scoring runs with worker threads sharing one store.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def done(self, run_id: str, report: str, version: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM report_scores"
                " WHERE run_id = ? AND report = ? AND config_version = ?",
                (run_id, report, version),
            ).fetchone()
        return row is not None

    def put(
        self,
        item: ReportText,
        config: ReportScoringConfig,
        answer: ChoiceAnswer | None,
        input_tokens: int = 0,
        skipped: str | None = None,
    ) -> None:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO report_scores (run_id, report, config_version,"
                " config_digest, skipped, stance, probabilities, confidence, input_tokens,"
                " scored_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.run_id,
                    item.report,
                    config.version,
                    config.digest,
                    skipped,
                    answer.choice if answer else None,
                    json.dumps(answer.probabilities) if answer else None,
                    answer.confidence if answer else None,
                    input_tokens,
                    now,
                ),
            )

    def stances(self, run_id: str, version: str) -> dict[str, dict[str, float]]:
        """Report -> its stance probabilities, for every scored report of the run."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT report, probabilities FROM report_scores"
                " WHERE run_id = ? AND config_version = ? AND skipped IS NULL",
                (run_id, version),
            ).fetchall()
        return {row["report"]: json.loads(row["probabilities"]) for row in rows}


def report_score(store: ReportScoreStore, run_id: str, version: str) -> float | None:
    """Mean of P(bullish) - P(bearish) over the run's scored reports; None if there are none."""
    stances = store.stances(run_id, version)
    if not stances:
        return None
    values = [p.get("bullish", 0.0) - p.get("bearish", 0.0) for p in stances.values()]
    return sum(values) / len(values)


def state_text(item: ReportText) -> str:
    return f"Stock: {item.ticker}\nReport by: {REPORT_NAMES[item.report]}\n\n{item.text.strip()}"


class ReportScoreResult(BaseModel):
    scored: int = 0
    done: int = 0  # already in the store
    skipped: int = 0  # too large for Jev; recorded, never asked again
    failed: int = 0  # not recorded; asked again next run
    input_tokens: int = 0


def score_reports(
    items: list[ReportText],
    store: ReportScoreStore,
    config: ReportScoringConfig,
    jev: CachedJev,
    workers: int = 4,
    on_step: Callable[[], None] = lambda: None,
) -> ReportScoreResult:
    questions = {
        name: JevQuestion(type=q.type, instructions=q.instructions, criteria=q.criteria)
        for name, q in config.questions.items()
    }

    def work(item: ReportText) -> tuple[str, int]:
        """The outcome for one report, and the input tokens it was charged for."""
        try:
            if store.done(item.run_id, item.report, config.version):
                return "done", 0
            request = JevRequest(model=config.model, state=state_text(item), questions=questions)
            try:
                response = jev.answer(request)
            except JevError as exc:
                if exc.too_large:
                    store.put(item, config, None, skipped="state_too_large")
                    return "skipped", 0
                return "failed", 0
            answer = response.answers["stance"]
            assert isinstance(answer, ChoiceAnswer)  # the spec's stance question is a choice
            store.put(item, config, answer, input_tokens=response.input_tokens)
            return "scored", 0 if response.cached else response.input_tokens
        finally:
            on_step()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outcomes = list(pool.map(work, items))
    kinds = [kind for kind, _ in outcomes]
    return ReportScoreResult(
        scored=kinds.count("scored"),
        done=kinds.count("done"),
        skipped=kinds.count("skipped"),
        failed=kinds.count("failed"),
        input_tokens=sum(tokens for _, tokens in outcomes),
    )
