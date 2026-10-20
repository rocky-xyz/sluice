from datetime import UTC, datetime
from pathlib import Path

import pytest

from sluice.evolution.experiments import Evidence, TrialBudgetExhausted
from sluice.evolution.pipeline_log import BUDGETS, PipelineEntry, PipelineLog, ScreenEvidence

TIME = datetime(2026, 10, 21, tzinfo=UTC)
SCREEN = ScreenEvidence(
    cells=55, candidate_spearman=0.1, champion_spearman=0.05, candidate_failed=0, champion_failed=0
)
FULL = Evidence(
    candidate_sharpes=[1.7, 1.8, 1.6], champion_sharpes=[1.6, 1.7, 1.5],
    candidate_sharpe=1.7, champion_sharpe=1.6, rho=0.9, z=1.2, days=150,
)  # fmt: skip


def entry(stage, n, status, key="k1", parent="v3-49da8839", **extra) -> PipelineEntry:
    fields = {
        "stage": stage, "n": n, "time": TIME, "commit": "abc", "source": "llm", "parent": parent,
        "layer": "rules", "changes": {"tools.max_calls": 4}, "params_key": key, "status": status,
    }  # fmt: skip
    if status in ("passed", "failed"):
        fields["evidence"] = SCREEN
    if status in ("accepted", "rejected"):
        fields["evidence"] = FULL
    if status == "accepted":
        fields["version"] = "v4"
    if status == "invalid":
        fields |= {
            "problems": ["tools.max_calls: 11 outside 1..10"],
            "layer": None,
            "params_key": None,
        }
    return PipelineEntry(**(fields | extra))


@pytest.fixture
def log(tmp_path):
    (tmp_path / "pipeline").mkdir()
    return PipelineLog(tmp_path)


def test_the_two_stages_count_apart(log):
    log.append(entry("screen", 1, "passed"))
    log.append(entry("screen", None, "invalid"))
    log.append(entry("full", 1, "rejected"))
    log.append(entry("screen", 2, "failed", key="k2"))
    assert (log.used("screen"), log.used("full")) == (2, 1)
    assert (log.remaining("screen"), log.remaining("full")) == (28, 4)
    assert log.find("screen", "k1", "v3-49da8839").status == "passed"
    assert log.find("screen", "k1", "v4-xxxx") is None


def test_a_full_evaluation_needs_a_passed_screen_against_the_same_champion(log):
    log.append(entry("screen", 1, "failed"))
    with pytest.raises(ValueError, match="no passed screen"):
        log.append(entry("full", 1, "rejected"))
    log.append(entry("screen", 2, "passed", key="k2"))
    with pytest.raises(ValueError, match="no passed screen"):
        log.append(entry("full", 1, "rejected", key="k2", parent="v4-other"))
    log.append(entry("full", 1, "accepted", key="k2"))


def test_budgets_run_out_per_stage(log):
    for n in range(1, BUDGETS["full"] + 1):
        log.append(entry("screen", n, "passed", key=f"k{n}"))
        log.append(entry("full", n, "rejected", key=f"k{n}"))
    with pytest.raises(TrialBudgetExhausted):
        log.require_budget("full")
    log.require_budget("screen")
    log.append(entry("screen", 6, "passed", key="k6"))
    with pytest.raises(TrialBudgetExhausted):
        log.append(entry("full", 6, "rejected", key="k6"))


def test_the_same_parameters_are_not_run_twice_in_a_stage(log):
    log.append(entry("screen", 1, "failed"))
    with pytest.raises(ValueError, match="twice"):
        log.append(entry("screen", 2, "failed"))
    log.append(entry("screen", 2, "failed", parent="v4-new"))  # a new champion may screen it again


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"stage": "screen", "n": 1, "status": "accepted", "evidence": FULL, "version": "v4"},
            "cannot be",
        ),
        ({"stage": "full", "n": 1, "status": "passed", "evidence": SCREEN}, "cannot be"),
        ({"stage": "screen", "n": 1, "status": "passed", "evidence": FULL}, "ScreenEvidence"),
        ({"stage": "full", "n": 1, "status": "rejected", "evidence": SCREEN}, "Evidence"),
        ({"stage": "full", "n": 1, "status": "accepted", "evidence": FULL}, "names its version"),
        ({"stage": "screen", "n": None, "status": "failed", "evidence": SCREEN}, "no number"),
    ],
)
def test_inconsistent_entries_are_refused(fields, message):
    base = {
        "time": TIME, "commit": "abc", "source": "llm", "parent": "v3-x", "layer": "rules",
        "changes": {}, "params_key": "k",
    }  # fmt: skip
    with pytest.raises(ValueError, match=message):
        PipelineEntry(**(base | fields))


def test_an_edited_log_is_refused(log):
    log.append(entry("screen", 1, "failed"))
    log.append(entry("screen", 2, "failed", key="k2"))
    log.path.write_text(log.path.read_text().splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="screen 2 where 1 was expected"):
        log.entries()


def test_the_shipped_log_is_valid():
    assert PipelineLog(Path(__file__).parents[2] / "configs").used("screen") >= 0
