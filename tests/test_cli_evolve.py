"""The step-9 evolve commands end to end, with the model and the runs faked (no API, no cost)."""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sluice import cli
from sluice.agents import llm as llm_module
from sluice.agents.batch import CellResult, run_dir
from sluice.agents.config import load_config
from sluice.agents.llm import LLMResponse
from sluice.evaluation.decisions import Decision
from sluice.evaluation.quality import Labelled
from sluice.evaluation.report import calendar
from sluice.evolution.pipeline_log import PipelineLog
from sluice.evolution.pipeline_proposer import unseen_days
from sluice.prices import yahoo
from sluice.universe import COMPANIES

REPO = Path(__file__).parents[1]
DAYS = calendar("validation").decisions


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A working copy with only the pipeline configs; the CLI's relative paths land here."""
    (tmp_path / "configs" / "pipeline").mkdir(parents=True)
    for name in ("v1.toml", "v3.toml", "CHAMPION"):
        shutil.copy(REPO / "configs" / "pipeline" / name, tmp_path / "configs" / "pipeline" / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_commit", lambda paths=("sluice",): "abc1234")
    monkeypatch.setattr(yahoo, "fetch_bars", lambda ticker, start: [])  # never the network
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    return tmp_path


def _good(ticker, day) -> str:
    return "overweight" if (DAYS.index(day) + COMPANIES.index(ticker)) % 2 else "underweight"


@contextmanager
def fake_gate(runs, workers, tickers):
    """Runs that rate every cell the right way for a candidate, half right for the champion."""
    saved: dict[str, list[Decision]] = {}
    v3 = load_config("v3")

    def add(config, cells, right):
        key = f"{config.version}-{config.digest}"
        saved.setdefault(key, []).extend(
            Decision(
                ticker=c.ticker, day=c.day, repeat=c.repeat, config=key, run_id=None,
                rating=_good(c.ticker, c.day) if right(c) else "hold", confidence=0.5,
            )
            for c in cells
        )  # fmt: skip

    from sluice.evolution.pipeline_gate import full_cells

    add(v3, full_cells(DAYS, tickers), lambda c: COMPANIES.index(c.ticker) % 2 == 0)

    def run(config, cells):
        add(config, cells, lambda c: True)
        return [CellResult(**c.model_dump(), status="ok") for c in cells]

    def decisions(config):
        return saved.get(f"{config.version}-{config.digest}", [])

    def label(found):
        sign = {"overweight": 0.02, "underweight": -0.02}
        return [Labelled(d, sign[_good(d.ticker, d.day)]) for d in found]

    yield run, decisions, label, lambda config: {}


def test_without_yes_a_screen_is_only_estimated(repo, monkeypatch):
    def never(*args):
        raise AssertionError("nothing may run without --yes")

    monkeypatch.setattr(cli, "_pipeline_gate", never)
    result = CliRunner().invoke(cli.app, ["evolve", "screen", "--set", "tools.max_calls=4"])
    assert result.exit_code == 0, result.output
    assert "55 runs, 0 already saved, 55 to run" in result.output
    assert "Not ready: the champion lacks 55 of these 55 runs." in result.output
    assert PipelineLog().entries() == []


def test_a_screen_with_yes_runs_through_the_gate_and_is_logged(repo, monkeypatch):
    monkeypatch.setattr(cli, "_pipeline_gate", fake_gate)
    result = CliRunner().invoke(
        cli.app, ["evolve", "screen", "--set", "tools.max_calls=4", "--rationale", "少调", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "passed (screen 1)" in result.output
    (entry,) = PipelineLog().entries()
    assert (entry.source, entry.rationale, entry.changes) == (
        "manual",
        "少调",
        {"tools.max_calls": 4},
    )
    status = CliRunner().invoke(cli.app, ["evolve", "pipeline-status"])
    assert "screens left 29, full evaluations left 5" in status.output


class FakeDeepSeek:
    answer = json.dumps({"changes": {"debate_rounds": 0}, "rationale": "辩论没有增加信息"})

    def __init__(self, key):
        pass

    def complete(self, request):
        assert "on decision days you are not shown" in request.messages[0].content
        return LLMResponse(text=self.answer, input_tokens=9000, output_tokens=500)


def seed_unseen_runs(repo: Path) -> None:
    v3 = load_config("v3")
    for day in unseen_days(DAYS)[:2]:
        for ticker in COMPANIES[:3]:
            path = run_dir(repo / "data/local/runs", v3) / ticker / f"{day}_r0.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "reports": {"news": {"text": "Strong [N1]."}},
                "decision": {"rating": "overweight", "confidence": 0.6, "rationale": "Up."},
                "retrieved": [],
            }))  # fmt: skip


def test_a_proposal_is_written_and_then_screened_as_the_models(repo, monkeypatch):
    seed_unseen_runs(repo)
    monkeypatch.setattr(llm_module, "DeepSeekClient", FakeDeepSeek)
    result = CliRunner().invoke(cli.app, ["evolve", "propose-pipeline"])
    assert result.exit_code == 0, result.output
    assert "material: 6 runs of v3-" in result.output
    assert "proposal: {'debate_rounds': 0}" in result.output
    (folder,) = (repo / "data/local/proposals").iterdir()
    assert json.loads((folder / "proposal.json").read_text())["rules"] == {"debate_rounds": 0}
    assert PipelineLog().entries() == []  # nothing is logged until it is screened

    monkeypatch.setattr(cli, "_pipeline_gate", fake_gate)
    screened = CliRunner().invoke(cli.app, ["evolve", "screen", "--proposal", str(folder), "--yes"])
    assert screened.exit_code == 0, screened.output
    (entry,) = PipelineLog().entries()
    assert (entry.source, entry.rationale) == ("llm", "辩论没有增加信息")


def test_proposing_nothing_is_logged_without_using_a_screen(repo, monkeypatch):
    seed_unseen_runs(repo)
    FakeNothing = type("FakeNothing", (FakeDeepSeek,), {})
    FakeNothing.answer = json.dumps({"changes": {}, "rationale": "没有值得试的改动"})
    monkeypatch.setattr(llm_module, "DeepSeekClient", FakeNothing)
    result = CliRunner().invoke(cli.app, ["evolve", "propose-pipeline"])
    assert result.exit_code == 0, result.output
    assert "logged as invalid (no change proposed); no screen used" in result.output
    (entry,) = PipelineLog().entries()
    assert (entry.status, entry.source, entry.n) == ("invalid", "llm", None)
    assert PipelineLog().remaining("screen") == 30


def test_the_real_gate_wiring_reads_saved_runs_and_needs_both_keys(repo, monkeypatch):
    import typer

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(typer.Exit), cli._pipeline_gate(repo / "data/local/runs", 1, COMPANIES):
        pass
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    seed_unseen_runs(repo)
    (repo / "configs" / "sizing").mkdir()
    for name in ("v1.toml", "CHAMPION"):
        shutil.copy(REPO / "configs" / "sizing" / name, repo / "configs" / "sizing" / name)
    with cli._pipeline_gate(repo / "data/local/runs", 1, COMPANIES) as (_, decisions, label, _p):
        found = decisions(load_config("v3"))
        assert len(found) == 6 and {d.rating for d in found} == {"overweight"}
        assert all(row.label is None for row in label(found))  # no prices stored here
        assert decisions(load_config("v1")) == []
