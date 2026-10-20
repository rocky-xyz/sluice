"""The prompt and rule layer's experiment log, with two budgets (task 9.8).

docs/specs/08-agent-harness.md ("已确认（任务 9.8）"). `configs/pipeline/
experiments.jsonl`, one JSON line per proposal, appended and never rewritten,
apart from the sizing layer's log. A proposal is first screened on a fixed set
of cells (30 screens for the whole of step 9); only one that passed may get a
full evaluation (5 of those), against the same champion it was screened
against. An invalid proposal, refused before anything ran, is logged without a
number and uses neither budget. The checks guard against accidental edits.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from sluice.agents.config import CONFIG_ROOT
from sluice.evolution.experiments import Evidence, TrialBudgetExhausted

LOG_FILE = "experiments.jsonl"
BUDGETS = {"screen": 30, "full": 5}
Stage = Literal["screen", "full"]
STATUSES = {"screen": {"invalid", "passed", "failed"}, "full": {"invalid", "accepted", "rejected"}}


class ScreenEvidence(BaseModel):
    """Rank correlation of rating and 10-day excess return on the screening cells."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cells: int
    candidate_spearman: float | None
    champion_spearman: float | None
    candidate_failed: int  # runs without a rating among the cells
    champion_failed: int


class PipelineEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: Stage
    n: int | None  # number within its stage; None for an invalid proposal
    time: AwareDatetime
    commit: str
    source: Literal["llm", "manual"]
    parent: str  # the champion, "<version>-<digest>"
    layer: Literal["rules", "prompt"] | None
    changes: dict[str, Any]
    params_key: str | None
    rationale: str = ""
    status: Literal["invalid", "passed", "failed", "accepted", "rejected"]
    problems: list[str] = []
    evidence: ScreenEvidence | Evidence | None = None
    version: str | None = None  # written when accepted

    @model_validator(mode="after")
    def _consistent(self) -> "PipelineEntry":
        if self.status not in STATUSES[self.stage]:
            raise ValueError(f"a {self.stage} entry cannot be {self.status}")
        invalid = self.status == "invalid"
        if invalid != (self.n is None) or invalid != bool(self.problems):
            raise ValueError("an invalid proposal has problems and no number, and only it")
        wanted = ScreenEvidence if self.stage == "screen" else Evidence
        if invalid != (self.evidence is None) or not (invalid or isinstance(self.evidence, wanted)):
            raise ValueError(
                f"a {self.stage} result carries {wanted.__name__}, an invalid one none"
            )
        if (self.status == "accepted") != (self.version is not None):
            raise ValueError("an accepted proposal names its version, and only it")
        if not invalid and (self.params_key is None or self.layer is None):
            raise ValueError("a proposal that ran has its parameter key and layer")
        return self


def _check(entries: list[PipelineEntry]) -> None:
    for stage, budget in BUDGETS.items():
        counted = [e for e in entries if e.stage == stage and e.n is not None]
        for expected, entry in enumerate(counted, start=1):
            if entry.n != expected:
                raise ValueError(f"{stage} {entry.n} where {expected} was expected")
        keys = [(e.params_key, e.parent) for e in counted]
        if len(keys) != len(set(keys)):
            raise ValueError(f"the same parameters were run through {stage} twice")
        if len(counted) > budget:
            raise ValueError(f"{len(counted)} {stage} entries; at most {budget} are allowed")
    passed: set[tuple[str | None, str]] = set()
    for entry in entries:
        if entry.stage == "screen" and entry.status == "passed":
            passed.add((entry.params_key, entry.parent))
        unscreened = (entry.params_key, entry.parent) not in passed
        if entry.stage == "full" and entry.n is not None and unscreened:
            raise ValueError(f"full {entry.n} has no passed screen before it")


class PipelineLog:
    def __init__(self, root: Path = CONFIG_ROOT):
        self.path = root / "pipeline" / LOG_FILE

    def entries(self) -> list[PipelineEntry]:
        if not self.path.exists():
            return []
        entries = []
        for number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append(PipelineEntry.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"{self.path} line {number}: {error}") from error
        _check(entries)
        return entries

    def used(self, stage: Stage) -> int:
        return sum(e.stage == stage and e.n is not None for e in self.entries())

    def remaining(self, stage: Stage) -> int:
        return BUDGETS[stage] - self.used(stage)

    def next_number(self, stage: Stage) -> int:
        return self.used(stage) + 1

    def require_budget(self, stage: Stage) -> None:
        """Call before running anything: refuses once the stage's budget is used."""
        if self.remaining(stage) <= 0:
            raise TrialBudgetExhausted(f"all {BUDGETS[stage]} {stage} trials are used")

    def find(self, stage: Stage, params_key: str, parent: str) -> PipelineEntry | None:
        """The counted entry of this stage for these parameters against this champion."""
        return next(
            (
                e
                for e in self.entries()
                if e.stage == stage and e.n is not None
                and e.params_key == params_key and e.parent == parent
            ),
            None,
        )  # fmt: skip

    def append(self, entry: PipelineEntry) -> None:
        entries = self.entries()
        if entry.n is not None and self.remaining(entry.stage) <= 0:
            raise TrialBudgetExhausted(f"all {BUDGETS[entry.stage]} {entry.stage} trials are used")
        _check([*entries, entry])
        line = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False)
        with self.path.open("a") as file:
            file.write(line + "\n")
