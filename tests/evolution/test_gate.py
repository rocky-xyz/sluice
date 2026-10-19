import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from sluice.evaluation.decisions import Decision
from sluice.evaluation.report import Calendar
from sluice.evolution.experiments import ExperimentLog, TrialBudgetExhausted
from sluice.evolution.gate import THRESHOLD, decide, propose, seed
from sluice.evolution.library import champion, load_champion
from sluice.evolution.validation import require_complete

CONFIGS = Path(__file__).parents[2] / "configs"
TIME = datetime(2026, 10, 20, tzinfo=UTC)
BASE = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01] * 25  # 150 days


@pytest.fixture
def root(tmp_path):
    (tmp_path / "sizing").mkdir()
    for name in ("v1.toml", "CHAMPION"):
        shutil.copy(CONFIGS / "sizing" / name, tmp_path / "sizing" / name)
    return tmp_path


class Fake:
    """Returns per repeat from a config: `edge[k]` added to every day; repeat 2 can be spoilt."""

    def __init__(self, edge: dict[float, float], spoil: dict[float, float] | None = None):
        self.edge, self.spoil, self.calls = edge, spoil or {}, []

    def __call__(self, config):
        self.calls.append(config.k)
        bonus = self.edge.get(config.k, 0.0)
        out = {r: [x + bonus for x in BASE] for r in (0, 1)}
        out[2] = [x + bonus + self.spoil.get(config.k, 0.0) for x in BASE]
        return out


def run(changes, backtest, root, source="manual"):
    return propose(changes, source, backtest, "abc1234", "why", root, now=lambda: TIME)


def test_decide_needs_every_repeat_better_and_z_over_the_threshold():
    up = {r: [x + 0.0005 for x in BASE] for r in range(3)}
    flat = {r: list(BASE) for r in range(3)}
    accepted, evidence = decide(up, flat)
    assert accepted and evidence.z >= THRESHOLD and evidence.days == 150
    worse_once = {**up, 2: [x - 0.0005 for x in BASE]}
    accepted, evidence = decide(worse_once, flat)
    assert not accepted
    assert evidence.candidate_sharpes[2] < evidence.champion_sharpes[2]
    with pytest.raises(ValueError, match="same 3 repeats"):
        decide({0: BASE, 1: BASE}, {0: BASE, 1: BASE})


def test_an_accepted_proposal_is_logged_then_written_then_made_champion(root):
    outcome = run({"k": 0.3}, Fake({0.3: 0.0005}), root)
    assert (outcome.status, outcome.version) == ("accepted", "v2")
    assert (outcome.entry.n, outcome.entry.parent) == (1, "v1-1bedf5b4")
    assert champion(root) == "v2" and load_champion(root).k == 0.3
    assert ExperimentLog(root).entries() == [outcome.entry]


def test_a_rejected_proposal_uses_a_trial_and_changes_nothing_else(root):
    outcome = run({"k": 0.3}, Fake({0.3: -0.0005}), root)
    assert outcome.status == "rejected" and outcome.entry.n == 1
    assert champion(root) == "v1" and not (root / "sizing" / "v2.toml").exists()


def test_an_invalid_proposal_is_logged_without_a_trial_or_a_backtest(root):
    fake = Fake({})
    outcome = run({"k": 5.0}, fake, root)
    assert outcome.status == "invalid" and "outside" in outcome.entry.problems[0]
    assert fake.calls == [] and ExperimentLog(root).used() == 0


def test_the_parameters_of_an_existing_version_call_for_a_rollback(root):
    run({"k": 0.3}, Fake({0.3: 0.0005}), root)  # v2 is champion now
    fake = Fake({})
    outcome = run({"k": 0.5}, fake, root)  # back to v1's parameters
    assert outcome.status == "invalid" and "rollback" in outcome.entry.problems[0]
    assert fake.calls == []


def test_a_proposal_evaluated_before_returns_its_entry(root):
    first = run({"k": 0.3}, Fake({0.3: -0.0005}), root)
    fake = Fake({})
    again = run({"k": 0.3}, fake, root)
    assert (again.status, again.entry) == ("evaluated before", first.entry)
    assert fake.calls == [] and ExperimentLog(root).used() == 1
    logged = ExperimentLog(root).entries()[-1]  # so the proposer's next prompt differs
    assert logged.status == "invalid" and "trial 1, evaluated before" in logged.problems[0]


def test_no_evaluation_once_the_budget_is_used(root):
    for i in range(20):
        run({"k": round(0.6 + 0.05 * i, 2)}, Fake({}), root)
    fake = Fake({})
    with pytest.raises(TrialBudgetExhausted):
        run({"k": 0.1}, fake, root)
    assert fake.calls == []


def test_an_interrupted_acceptance_is_refused_until_mended(root):
    run({"k": 0.3}, Fake({0.3: 0.0005}), root)
    (root / "sizing" / "v2.toml").unlink()
    with pytest.raises(RuntimeError, match="file is missing"):
        run({"k": 0.2}, Fake({}), root)


def test_seed_logs_v1_then_the_two_controls_and_can_be_rerun(root):
    fake = Fake({})  # neither control beats v1
    outcomes = seed(fake, "abc1234", root, now=lambda: TIME)
    assert [o.status for o in outcomes] == ["rejected", "rejected"]
    entries = ExperimentLog(root).entries()
    assert [(e.n, e.status, e.source) for e in entries] == [
        (1, "seen", "6.12"), (2, "rejected", "6.12"), (3, "rejected", "6.12"),
    ]  # fmt: skip
    assert entries[1].changes == {"k": 0.0}
    assert entries[2].changes == {"k": 0.0, "stocks.vol_scaling": False}
    assert seed(fake, "abc1234", root, now=lambda: TIME) == []
    assert ExperimentLog(root).used() == 3


def test_seed_measures_the_second_control_against_a_new_champion(root):
    seed(Fake({0.0: 0.0005}), "abc1234", root, now=lambda: TIME)  # k = 0 wins
    entries = ExperimentLog(root).entries()
    assert (entries[1].status, entries[1].version) == ("accepted", "v2")
    assert entries[2].parent.startswith("v2-")
    assert entries[2].changes == {"stocks.vol_scaling": False}


def test_seed_refuses_an_empty_log_with_another_champion(root):
    run({"k": 0.3}, Fake({0.3: 0.0005}), root)
    (root / "sizing" / "experiments.jsonl").unlink()
    with pytest.raises(RuntimeError, match="not v1"):
        seed(Fake({}), "abc1234", root)


def test_the_backtest_waits_for_every_repeat():
    days = [date(2026, 1, 2), date(2026, 1, 16)]
    cal = Calendar(decisions=days, rebalances=days, days=days, end=days[-1])
    full = [
        Decision(ticker=t, day=d, repeat=r, config="v1-x", run_id=None, rating="hold",
                 confidence=None)
        for t in ("AMD", "KO") for d in days for r in range(3)
    ]  # fmt: skip
    require_complete(full, ["AMD", "KO"], cal)
    with pytest.raises(ValueError, match="repeat 2 has 3 of 4"):
        require_complete(full[:-1], ["AMD", "KO"], cal)
