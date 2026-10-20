"""The two-stage gate for the prompt and rule layer (task 9.9).

docs/specs/08-agent-harness.md. A screen runs the candidate on 55 fixed
cells (validation decision days 1, 4, 7, 10 and 13, every stock, repeat 0) and
passes when the rank correlation of its ratings with the 10-day excess return
is above the champion's on the same cells; a failed run counts as hold. A full
evaluation, allowed only after a passed screen against the same champion,
runs every validation cell three times and applies the sizing layer's two
rules (`gate.decide`) to the portfolios the sizing champion builds from each
pipeline's ratings. Everything is logged; nothing about the rules lives here
that the evolution loop could change.

Running cells costs money, so it is a function handed in (`Runner`); so are
reading saved runs, labelling them and the portfolio backtest. A run that
broke off stops the gate before anything is logged: rerun it, finished cells
come from the cache.
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from sluice.agents.batch import Cell, CellResult
from sluice.agents.config import CONFIG_ROOT, PipelineConfig
from sluice.evaluation.decisions import Decision
from sluice.evaluation.quality import SCORES, Labelled, _rank
from sluice.evolution.gate import REPEATS, decide
from sluice.evolution.library import Changes, _next_version
from sluice.evolution.pipeline_log import PipelineEntry, PipelineLog, ScreenEvidence, Stage
from sluice.evolution.pipelines import (
    candidate,
    check,
    layer,
    load_champion,
    set_champion,
    version_with,
    write_version,
)

SCREEN_DAYS = (0, 3, 6, 9, 12)  # indexes into the validation decision days: 1st, 4th, ...
# Dollars per v3 run at peak list prices with nothing cached, measured on the 9.6 pilot
# (11 runs, $0.52); only for the estimate shown before anything is spent.
COST_PER_RUN = 0.047

Runner = Callable[[PipelineConfig, list[Cell]], list[CellResult]]
Decisions = Callable[[PipelineConfig], list[Decision]]  # a config's saved validation runs
Labeller = Callable[[list[Decision]], list[Labelled]]
Portfolio = Callable[[PipelineConfig], dict[int, list[float]]]  # daily returns per repeat


@dataclass(frozen=True)
class Outcome:
    status: Literal["invalid", "done before", "passed", "failed", "accepted", "rejected"]
    entry: PipelineEntry
    version: str | None = None


def screen_cells(days: list[date], tickers: list[str]) -> list[Cell]:
    return [Cell(ticker=t, day=days[i], repeat=0) for i in SCREEN_DAYS for t in tickers]


def full_cells(days: list[date], tickers: list[str]) -> list[Cell]:
    return [Cell(ticker=t, day=d, repeat=r) for d in days for t in tickers for r in range(REPEATS)]


def missing(decisions: list[Decision], cells: list[Cell]) -> list[Cell]:
    have = {(d.ticker, d.day, d.repeat) for d in decisions}
    return [c for c in cells if (c.ticker, c.day, c.repeat) not in have]


def screen_score(labelled: list[Labelled], cells: list[Cell]) -> tuple[float | None, int]:
    """Spearman of rating score and label over `cells`; a failed run scores 0. (value, failed)."""
    by_cell = {(r.decision.ticker, r.decision.day, r.decision.repeat): r for r in labelled}
    scores, labels, failed = [], [], 0
    for cell in cells:
        row = by_cell.get((cell.ticker, cell.day, cell.repeat))
        if row is None or row.label is None:
            continue
        rating = row.decision.rating
        failed += rating is None
        scores.append(SCORES[rating] if rating else 0)
        labels.append(row.label)
    if len(scores) < 3:
        return None, failed
    xs, ys = _rank(scores), _rank(labels)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None, failed
    return statistics.correlation(xs, ys), failed


def _common(champion: PipelineConfig, changes: Changes, source, commit, rationale, now) -> dict:
    return {
        "time": now(),
        "commit": commit,
        "source": source,
        "parent": f"{champion.version}-{champion.digest}",
        "changes": changes,
        "rationale": rationale,
    }


def _refusal(champion, changes, root) -> tuple[list[str], str | None]:
    problems = check(champion, changes)
    key = None
    if not problems:
        key = candidate(champion, changes).digest
        if same := version_with(key, root):
            problems = [f"the parameters of {same}: going back to it is a rollback, not a trial"]
    return problems, key


def _ran(run: Runner, config: PipelineConfig, cells: list[Cell]) -> None:
    results = run(config, cells)
    if broke := [r for r in results if r.status == "error"]:
        raise RuntimeError(f"{len(broke)} runs broke off; run again (finished ones are cached)")


def _invalid(log, stage: Stage, common, key, problems) -> Outcome:
    entry = PipelineEntry(
        stage=stage, n=None, layer=None, params_key=key, status="invalid", problems=problems,
        **common,
    )  # fmt: skip
    log.append(entry)
    return Outcome("invalid", entry)


def screen(
    changes: Changes,
    source: Literal["llm", "manual"],
    run: Runner,
    decisions: Decisions,
    label: Labeller,
    cells: list[Cell],
    commit: str,
    rationale: str = "",
    root: Path = CONFIG_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Outcome:
    log = PipelineLog(root)
    champion = load_champion(root)
    common = _common(champion, changes, source, commit, rationale, now)
    problems, key = _refusal(champion, changes, root)
    if problems:
        return _invalid(log, "screen", common, key, problems)
    if earlier := log.find("screen", key, common["parent"]):
        problem = f"screened before as screen {earlier.n}: {earlier.status}"
        _invalid(log, "screen", common, key, [problem])
        return Outcome("done before", earlier)
    log.require_budget("screen")
    held = decisions(champion)
    if gap := missing(held, cells):
        raise ValueError(f"the champion lacks {len(gap)} of the {len(cells)} screening runs")
    proposed = candidate(champion, changes)
    _ran(run, proposed, cells)
    candidate_value, candidate_failed = screen_score(label(decisions(proposed)), cells)
    champion_value, champion_failed = screen_score(label(held), cells)
    evidence = ScreenEvidence(
        cells=len(cells),
        candidate_spearman=candidate_value,
        champion_spearman=champion_value,
        candidate_failed=candidate_failed,
        champion_failed=champion_failed,
    )
    passed = None not in (candidate_value, champion_value) and candidate_value > champion_value
    entry = PipelineEntry(
        stage="screen", n=log.next_number("screen"), layer=layer(changes), params_key=key,
        status="passed" if passed else "failed", evidence=evidence, **common,
    )  # fmt: skip
    log.append(entry)
    return Outcome(entry.status, entry)


def full(
    changes: Changes,
    source: Literal["llm", "manual"],
    run: Runner,
    decisions: Decisions,
    portfolio: Portfolio,
    cells: list[Cell],
    commit: str,
    rationale: str = "",
    root: Path = CONFIG_ROOT,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Outcome:
    log = PipelineLog(root)
    champion = load_champion(root)
    common = _common(champion, changes, source, commit, rationale, now)
    problems, key = _refusal(champion, changes, root)
    parent = common["parent"]
    if not problems:
        screened = log.find("screen", key, parent)
        if screened is None or screened.status != "passed":
            problems = ["no passed screen of these parameters against the current champion"]
    if problems:
        return _invalid(log, "full", common, key, problems)
    if earlier := log.find("full", key, parent):
        return Outcome("done before", earlier)
    log.require_budget("full")
    if gap := missing(decisions(champion), cells):
        raise ValueError(f"the champion lacks {len(gap)} of the {len(cells)} validation runs")
    proposed = candidate(champion, changes)
    _ran(run, proposed, cells)
    accepted, evidence = decide(portfolio(proposed), portfolio(champion))
    version = _next_version(root / "pipeline") if accepted else None
    entry = PipelineEntry(
        stage="full", n=log.next_number("full"), layer=layer(changes), params_key=key,
        status="accepted" if accepted else "rejected", evidence=evidence, version=version,
        **common,
    )  # fmt: skip
    log.append(entry)
    if not accepted:
        return Outcome("rejected", entry)
    written = write_version(champion, changes, root)
    if written.version != version:
        raise RuntimeError(f"logged {version} but wrote {written.version}")
    set_champion(written.version, root)
    return Outcome("accepted", entry, written.version)
