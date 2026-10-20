import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from sluice.agents.batch import CellResult
from sluice.agents.config import load_config
from sluice.evaluation.decisions import Decision
from sluice.evaluation.quality import Labelled
from sluice.evolution.experiments import TrialBudgetExhausted
from sluice.evolution.pipeline_gate import (
    full,
    full_cells,
    screen,
    screen_cells,
    screen_score,
)
from sluice.evolution.pipeline_log import PipelineLog
from sluice.evolution.pipelines import champion, load_champion

CONFIGS = Path(__file__).parents[2] / "configs"
TIME = datetime(2026, 10, 21, tzinfo=UTC)
DAYS = [date(2026, 1, 2) + timedelta(days=14 * i) for i in range(15)]
TICKERS = ["AAA", "BBB", "CCC"]
BASE = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01] * 25


def label(ticker: str, day: date) -> float:
    return 0.02 if (DAYS.index(day) + TICKERS.index(ticker)) % 2 else -0.02


class World:
    """Saved runs per config, a runner that adds to them, and labels for every cell.

    A config with tools.max_calls == 4 rates every cell the right way; any other
    rates it the wrong way, except the champion, which is right half the time.
    """

    def __init__(self, root: Path, broken: int = 0):
        self.root, self.broken = root, broken
        self.saved: dict[str, list[Decision]] = {}
        self.ran: list[tuple[str, int]] = []
        v3 = load_config("v3", root)
        self.add(v3, full_cells(DAYS, TICKERS))

    def rating(self, config, ticker, day) -> str:
        good = "overweight" if label(ticker, day) > 0 else "underweight"
        bad = "underweight" if good == "overweight" else "overweight"
        if config.tools.max_calls == 4:
            return good
        if config.version == "v3":
            return good if TICKERS.index(ticker) == 0 else bad
        return bad

    def add(self, config, cells):
        key = f"{config.version}-{config.digest}"
        self.saved.setdefault(key, []).extend(
            Decision(
                ticker=c.ticker, day=c.day, repeat=c.repeat, config=key, run_id=None,
                rating=self.rating(config, c.ticker, c.day), confidence=0.5,
            )
            for c in cells
        )  # fmt: skip

    def run(self, config, cells):
        self.ran.append((config.digest, len(cells)))
        self.add(config, cells)
        status = ["error"] * self.broken + ["ok"] * (len(cells) - self.broken)
        return [CellResult(**c.model_dump(), status=s) for c, s in zip(cells, status, strict=True)]

    def decisions(self, config):
        return self.saved.get(f"{config.version}-{config.digest}", [])

    @staticmethod
    def label(decisions):
        return [Labelled(d, label(d.ticker, d.day)) for d in decisions]

    @staticmethod
    def portfolio(config):
        edge = 0.0005 if config.tools.max_calls == 4 else 0.0
        return {r: [x + edge for x in BASE] for r in range(3)}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "pipeline").mkdir()
    for name in ("v1.toml", "v3.toml", "CHAMPION"):
        shutil.copy(CONFIGS / "pipeline" / name, tmp_path / "pipeline" / name)
    return tmp_path


def do_screen(world, root, changes, **kw):
    cells = screen_cells(DAYS, TICKERS)
    return screen(
        changes, "manual", world.run, world.decisions, world.label, cells, "abc", "why", root,
        now=lambda: TIME, **kw,
    )  # fmt: skip


def do_full(world, root, changes):
    return full(
        changes, "manual", world.run, world.decisions, world.portfolio,
        full_cells(DAYS, TICKERS), "abc", "why", root, now=lambda: TIME,
    )  # fmt: skip


def test_screen_cells_are_days_1_4_7_10_13_at_repeat_0():
    cells = screen_cells(DAYS, TICKERS)
    assert sorted({c.day for c in cells}) == [DAYS[i] for i in (0, 3, 6, 9, 12)]
    assert {c.repeat for c in cells} == {0} and len(cells) == 15


def test_a_failed_run_counts_as_hold():
    cells = screen_cells(DAYS, TICKERS)
    ratings = ["overweight", None, "underweight", "hold", "overweight"] * 3

    def rows(fill):
        return [
            Labelled(
                Decision(ticker=c.ticker, day=c.day, repeat=0, config="x", run_id=None,
                         rating=r if r else fill, confidence=None),
                label(c.ticker, c.day),
            )
            for c, r in zip(cells, ratings, strict=True)
        ]  # fmt: skip

    failed_value, failed = screen_score(rows(None), cells)
    hold_value, none_failed = screen_score(rows("hold"), cells)
    assert (failed, none_failed) == (3, 0)
    assert failed_value == pytest.approx(hold_value)


def test_a_better_candidate_passes_the_screen_and_is_logged(root):
    world = World(root)
    outcome = do_screen(world, root, {"tools.max_calls": 4})
    assert outcome.status == "passed"
    ev = outcome.entry.evidence
    assert ev.candidate_spearman == pytest.approx(1.0) and ev.champion_spearman < 1
    assert world.ran == [(outcome.entry.params_key, 15)]
    assert PipelineLog(root).used("screen") == 1


def test_a_worse_candidate_fails_and_cannot_have_a_full_evaluation(root):
    world = World(root)
    assert do_screen(world, root, {"tools.max_calls": 8}).status == "failed"
    outcome = do_full(world, root, {"tools.max_calls": 8})
    assert outcome.status == "invalid" and "no passed screen" in outcome.entry.problems[0]
    assert len(world.ran) == 1  # the full evaluation never ran anything


def test_a_passed_screen_then_a_full_evaluation_makes_a_new_champion(root):
    world = World(root)
    do_screen(world, root, {"tools.max_calls": 4})
    outcome = do_full(world, root, {"tools.max_calls": 4})
    assert (outcome.status, outcome.version) == ("accepted", "v4")
    assert champion(root) == "v4" and load_champion(root).tools.max_calls == 4
    assert world.ran[-1][1] == len(DAYS) * len(TICKERS) * 3  # every cell, three repeats
    assert outcome.entry.evidence.z > 3
    log = PipelineLog(root)
    assert (log.used("screen"), log.used("full")) == (1, 1)


def test_an_invalid_proposal_runs_nothing_and_uses_no_budget(root):
    world = World(root)
    outcome = do_screen(world, root, {"temperature": 0.7})
    assert outcome.status == "invalid" and world.ran == []
    assert PipelineLog(root).used("screen") == 0


def test_a_screen_done_before_is_not_run_again(root):
    world = World(root)
    first = do_screen(world, root, {"tools.max_calls": 8})
    again = do_screen(world, root, {"tools.max_calls": 8})
    assert (again.status, again.entry) == ("done before", first.entry)
    assert len(world.ran) == 1
    assert PipelineLog(root).entries()[-1].status == "invalid"  # logged, so prompts move on


def test_broken_runs_stop_the_gate_before_anything_is_logged(root):
    world = World(root, broken=2)
    with pytest.raises(RuntimeError, match="2 runs broke off"):
        do_screen(world, root, {"tools.max_calls": 4})
    assert PipelineLog(root).entries() == []


def test_the_champion_must_have_its_runs_before_anything_is_spent(root):
    world = World(root)
    world.saved.clear()
    with pytest.raises(ValueError, match="champion lacks 15 of the 15"):
        do_screen(world, root, {"tools.max_calls": 4})
    assert world.ran == []


def test_the_screen_budget_runs_out(root):
    world = World(root)
    log = PipelineLog(root)
    for n in range(30):
        prompt = load_config("v3", root).prompts["bull"] + f"\nTrial {n}."
        do_screen(world, root, {"prompts.bull": prompt})
    assert log.remaining("screen") == 0
    with pytest.raises(TrialBudgetExhausted):
        do_screen(world, root, {"tools.max_calls": 4})
