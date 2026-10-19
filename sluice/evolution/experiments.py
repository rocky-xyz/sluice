"""The experiment log and the trial budget (task 7.3).

docs/specs/06-evolution.md ("已确认（任务 7.3：实验日志）"). One JSON line per
proposal in `configs/sizing/experiments.jsonl`, appended and never rewritten.
A proposal that looked at the validation period gets a number and uses one of
the `MAX_TRIALS`; one refused before anything was evaluated is logged as
`invalid`, without a number, so the proposer can see it. The checks on reading
guard against accidental edits, not against someone determined to get around them.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from sluice.sizing.config import CONFIG_ROOT

LOG_FILE = "experiments.jsonl"
MAX_TRIALS = 20  # for the whole of step 7; the gatekeeper's threshold is fixed from it

Status = Literal["seen", "invalid", "accepted", "rejected"]


class TrialBudgetExhausted(RuntimeError):
    """Every trial has been used; no further proposal may be evaluated."""


class Evidence(BaseModel):
    """What the gatekeeper decided on: Sharpe ratios annualised, z from the daily ones."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_sharpes: list[float]  # one per repeat
    champion_sharpes: list[float]
    candidate_sharpe: float  # of the repeats' daily returns averaged day by day
    champion_sharpe: float
    rho: float  # correlation of the two averaged series
    z: float  # Jobson-Korkie with Memmel's correction
    days: int


class Entry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int | None  # trial number; None for an invalid proposal, which uses no trial
    time: AwareDatetime
    commit: str
    source: Literal["6.12", "llm", "manual"]
    parent: str | None  # the champion compared against, "<version>-<digest>"; None if none
    changes: dict[str, Any]
    params_key: str | None  # None when the proposal could not even be built
    rationale: str = ""
    status: Status
    problems: list[str] = []  # why an invalid proposal was refused
    evidence: Evidence | None = None  # for accepted and rejected ones
    version: str | None = None  # the version written when accepted; the config seen

    @model_validator(mode="after")
    def _consistent(self) -> "Entry":
        if (self.status == "invalid") != (self.n is None):
            raise ValueError("an invalid proposal has no trial number, and only it")
        if (self.status == "invalid") != bool(self.problems):
            raise ValueError("problems are given for an invalid proposal, and only for it")
        if (self.status in ("accepted", "rejected")) != (self.evidence is not None):
            raise ValueError("evidence is given for an accepted or rejected proposal, and only it")
        if self.status in ("seen", "accepted") and self.version is None:
            raise ValueError(f"a {self.status} entry names its version")
        if self.status != "invalid" and self.params_key is None:
            raise ValueError("an evaluated proposal has a parameter key")
        return self


def _check(entries: list[Entry]) -> None:
    """Numbers run 1, 2, 3 …; a counted parameter key appears once; at most MAX_TRIALS."""
    counted = [entry for entry in entries if entry.n is not None]
    for expected, entry in enumerate(counted, start=1):
        if entry.n != expected:
            raise ValueError(f"trial {entry.n} where {expected} was expected")
    keys = [entry.params_key for entry in counted]
    for key in keys:
        if keys.count(key) > 1:
            raise ValueError(f"parameters {key} were evaluated more than once")
    if len(counted) > MAX_TRIALS:
        raise ValueError(f"{len(counted)} trials logged; at most {MAX_TRIALS} are allowed")


class ExperimentLog:
    def __init__(self, root: Path = CONFIG_ROOT):
        self.path = root / "sizing" / LOG_FILE

    def entries(self) -> list[Entry]:
        if not self.path.exists():
            return []
        entries = []
        for number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append(Entry.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"{self.path} line {number}: {error}") from error
        _check(entries)
        return entries

    def used(self) -> int:
        return sum(entry.n is not None for entry in self.entries())

    def remaining(self) -> int:
        return MAX_TRIALS - self.used()

    def next_number(self) -> int:
        return self.used() + 1

    def require_budget(self) -> None:
        """Call before evaluating anything: refuses once every trial is used."""
        if self.remaining() <= 0:
            raise TrialBudgetExhausted(f"all {MAX_TRIALS} trials are used")

    def find(self, params_key: str) -> Entry | None:
        """The counted entry with these parameters, if they have been evaluated before."""
        return next(
            (e for e in self.entries() if e.n is not None and e.params_key == params_key), None
        )

    def append(self, entry: Entry) -> None:
        entries = self.entries()
        if entry.n is not None:
            if sum(e.n is not None for e in entries) >= MAX_TRIALS:
                raise TrialBudgetExhausted(f"all {MAX_TRIALS} trials are used")
            _check([*entries, entry])
        line = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False)
        with self.path.open("a") as file:
            file.write(line + "\n")
