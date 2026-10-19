"""The gatekeeper: accept a sizing change only when it beats the champion beyond noise (task 7.5).

docs/specs/06-evolution.md ("已确认（任务 7.1）", "（任务 7.5）"). Both must hold:
the candidate's Sharpe is above the champion's on every one of the pipeline's
repeats (run-to-run noise), and on the repeats' daily returns averaged day by
day, the Jobson-Korkie/Memmel z of the difference is at least `THRESHOLD`
(sample noise). Only the validation period is read; the threshold was fixed
from the trial budget before any trial.

A proposal is checked before anything is evaluated: one outside the search
space, or with the parameters of an existing version (going back to one is a
rollback, a person's decision), is logged as invalid and uses no trial; one
evaluated before is logged as invalid too and returns its old entry. Otherwise the log entry is written
first, then the version file, then `CHAMPION`, so an interruption leaves an
accepted entry without its file, which the next run refuses to go past,
rather than an evaluation missing from the log.
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sluice.evaluation import metrics
from sluice.evolution.experiments import Entry, Evidence, ExperimentLog
from sluice.evolution.library import (
    Changes,
    candidate,
    check,
    load_champion,
    next_version,
    params,
    params_key,
    set_champion,
    version_with,
    write_version,
)
from sluice.sizing.config import CONFIG_ROOT, SizingConfig, load_sizing_config

THRESHOLD = 3.0  # z; from 20 trials at about 5% in all (docs/specs/06-evolution.md)
REPEATS = 3

# A sizing config -> the daily returns of its portfolio on the validation period, per repeat.
Backtest = Callable[[SizingConfig], dict[int, list[float]]]


@dataclass(frozen=True)
class Outcome:
    status: Literal["invalid", "evaluated before", "accepted", "rejected"]
    entry: Entry
    version: str | None = None  # the champion after this proposal, when it changed


def decide(
    candidate_returns: dict[int, list[float]], champion_returns: dict[int, list[float]]
) -> tuple[bool, Evidence]:
    """The two rules, on returns already worked out. Fewer than `REPEATS` repeats is an error."""
    repeats = sorted(champion_returns)
    if repeats != sorted(candidate_returns) or len(repeats) != REPEATS:
        raise ValueError(f"need the same {REPEATS} repeats on both sides, got {repeats}")
    candidate_sharpes = [metrics.sharpe(candidate_returns[r]) for r in repeats]
    champion_sharpes = [metrics.sharpe(champion_returns[r]) for r in repeats]
    if None in candidate_sharpes or None in champion_sharpes:
        raise ValueError("a portfolio with flat returns has no Sharpe ratio")

    def averaged(returns: dict[int, list[float]]) -> list[float]:
        return [statistics.fmean(day) for day in zip(*(returns[r] for r in repeats), strict=True)]

    mean_candidate, mean_champion = averaged(candidate_returns), averaged(champion_returns)
    test = metrics.sharpe_difference(mean_candidate, mean_champion)
    evidence = Evidence(
        candidate_sharpes=candidate_sharpes,
        champion_sharpes=champion_sharpes,
        candidate_sharpe=metrics.sharpe(mean_candidate),
        champion_sharpe=metrics.sharpe(mean_champion),
        rho=test.rho if test else 1.0,
        z=test.z if test else 0.0,
        days=len(mean_candidate),
    )
    every_repeat = all(c > h for c, h in zip(candidate_sharpes, champion_sharpes, strict=True))
    return every_repeat and evidence.z >= THRESHOLD, evidence


def _require_files_for_accepted(log: ExperimentLog, root: Path) -> None:
    for entry in log.entries():
        if entry.status == "accepted" and not (root / "sizing" / f"{entry.version}.toml").exists():
            raise RuntimeError(
                f"trial {entry.n} accepted {entry.version}, but its file is missing: an earlier "
                "run was interrupted; write it with library.write_version and set CHAMPION by hand"
            )


def propose(
    changes: Changes,
    source: Literal["6.12", "llm", "manual"],
    backtest: Backtest,
    commit: str,
    rationale: str = "",
    root: Path = CONFIG_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Outcome:
    """Run one proposal through the gate, against the current champion, and log it."""
    log = ExperimentLog(root)
    _require_files_for_accepted(log, root)
    champion = load_champion(root)
    common = {
        "time": now(),
        "commit": commit,
        "source": source,
        "parent": f"{champion.version}-{champion.digest}",
        "changes": changes,
        "rationale": rationale,
    }

    problems = check(champion, changes)
    key = None
    if not problems:
        key = candidate(champion, changes).digest
        if same := version_with(key, root):
            problems = [f"the parameters of {same}: going back to it is a rollback, not a trial"]
    if problems:
        entry = Entry(n=None, params_key=key, status="invalid", problems=problems, **common)
        log.append(entry)
        return Outcome("invalid", entry)

    if earlier := log.find(key):
        # Logged, so the log, and with it the proposer's next prompt, changes.
        problem = f"the parameters of trial {earlier.n}, evaluated before: {earlier.status}"
        log.append(Entry(n=None, params_key=key, status="invalid", problems=[problem], **common))
        return Outcome("evaluated before", earlier)

    log.require_budget()
    proposed = candidate(champion, changes)
    accepted, evidence = decide(backtest(proposed), backtest(champion))
    version = next_version(root) if accepted else None
    entry = Entry(
        n=log.next_number(),
        params_key=key,
        status="accepted" if accepted else "rejected",
        evidence=evidence,
        version=version,
        **common,
    )
    log.append(entry)
    if not accepted:
        return Outcome("rejected", entry)
    written = write_version(champion, changes, root)
    if written.version != version:
        raise RuntimeError(f"logged {version} but wrote {written.version}")
    set_champion(written.version, root)
    return Outcome("accepted", entry, written.version)


# The three sizing configs task 6.12 already looked at on the validation period,
# as parameters to apply to v1: the formula itself, k = 0, and exposure alone.
SEEN_IN_6_12: list[tuple[str, Changes]] = [
    ("6.12：波动率缩放 + 总仓位（k = 0，不看信号）", {"k": 0.0}),
    ("6.12：只做总仓位（等权 × E）", {"k": 0.0, "stocks.vol_scaling": False}),
]


def seed(
    backtest: Backtest,
    commit: str,
    root: Path = CONFIG_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[Outcome]:
    """Log what 6.12 saw: v1 as seen, then the two controls through the gate. Resumable."""
    log = ExperimentLog(root)
    v1 = load_champion(root) if not log.entries() else None
    if v1 is not None:
        if v1.version != "v1":
            raise RuntimeError(f"the log is empty but the champion is {v1.version}, not v1")
        log.append(
            Entry(
                n=1, time=now(), commit=commit, source="6.12", parent=None, changes={},
                params_key=params_key(v1), rationale="6.12：仓位公式 v1，初始冠军",
                status="seen", version="v1",
            )
        )  # fmt: skip
    first = log.entries()[0]
    if first.status != "seen" or first.version != "v1":
        raise RuntimeError("the log does not start with v1 as seen; it was not seeded")
    base = params(load_sizing_config("v1", root))
    outcomes = []
    for rationale, overrides in SEEN_IN_6_12:
        target = {**base, **overrides}
        if any(e.rationale == rationale for e in log.entries()):
            continue  # done on an earlier run
        champion = params(load_champion(root))
        changes = {k: v for k, v in target.items() if champion[k] != v}
        outcomes.append(propose(changes, "6.12", backtest, commit, rationale, root, now))
    return outcomes
