from datetime import UTC, datetime
from pathlib import Path

import pytest

from sluice.evolution.experiments import (
    MAX_TRIALS,
    Entry,
    Evidence,
    ExperimentLog,
    TrialBudgetExhausted,
)

TIME = datetime(2026, 10, 20, tzinfo=UTC)
EVIDENCE = Evidence(
    candidate_sharpes=[1.6, 1.7, 1.5], champion_sharpes=[1.5, 1.6, 1.4],
    candidate_sharpe=1.62, champion_sharpe=1.52, rho=0.97, z=1.1, days=150,
)  # fmt: skip


def entry(n: int | None, status: str = "rejected", key: str | None = None, **extra) -> Entry:
    fields = {
        "n": n, "time": TIME, "commit": "abc1234", "source": "llm", "parent": "v1-1bedf5b4",
        "changes": {"k": 0.1 * (n or 0)}, "params_key": key or f"key{n}", "status": status,
    }  # fmt: skip
    if status in ("accepted", "rejected"):
        fields["evidence"] = EVIDENCE
    if status in ("accepted", "seen"):
        fields["version"] = "v2"
    if status == "invalid":
        fields["problems"] = ["k: 9.0 outside 0..2"]
    return Entry(**(fields | extra))


@pytest.fixture
def log(tmp_path):
    (tmp_path / "sizing").mkdir()
    return ExperimentLog(tmp_path)


def test_an_empty_or_missing_log_has_every_trial_left(log):
    assert (log.entries(), log.used(), log.remaining(), log.next_number()) == ([], 0, 20, 1)
    log.require_budget()
    log.path.write_text("")
    assert log.entries() == []


def test_entries_round_trip_and_only_numbered_ones_count(log):
    first = entry(1, "seen", parent=None, changes={}, source="6.12", version="v1")
    invalid = entry(None, "invalid", key=None)
    second = entry(2, "accepted")
    for e in (first, invalid, second):
        log.append(e)
    assert log.entries() == [first, invalid, second]
    assert (log.used(), log.remaining(), log.next_number()) == (2, 18, 3)
    assert log.find("key2") == second
    assert log.find("nope") is None


def test_appending_never_rewrites_earlier_lines(log):
    log.append(entry(1))
    before = log.path.read_text()
    log.append(entry(2))
    assert log.path.read_text().startswith(before)


def test_the_budget_runs_out(log):
    for n in range(1, MAX_TRIALS + 1):
        log.append(entry(n))
    with pytest.raises(TrialBudgetExhausted):
        log.require_budget()
    with pytest.raises(TrialBudgetExhausted):
        log.append(entry(MAX_TRIALS + 1))
    log.append(entry(None, "invalid", key=None))  # a refused proposal is still logged


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (entry(3), "trial 3 where 2 was expected"),
        (entry(2, key="key1"), "evaluated more than once"),
    ],
)
def test_append_refuses_a_broken_sequence(log, bad, message):
    log.append(entry(1))
    with pytest.raises(ValueError, match=message):
        log.append(bad)
    assert log.used() == 1


def test_reading_refuses_an_edited_log(log):
    log.append(entry(1))
    log.append(entry(2))
    lines = log.path.read_text().splitlines()
    log.path.write_text(lines[1] + "\n")  # the first line deleted by hand
    with pytest.raises(ValueError, match="trial 2 where 1 was expected"):
        log.entries()
    log.path.write_text(lines[0] + "\n" + lines[1].replace('"z": 1.1', '"zz": 1.1') + "\n")
    with pytest.raises(ValueError, match="line 2"):
        log.entries()


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"n": None, "status": "rejected"}, "no trial number"),
        ({"n": 1, "status": "invalid", "problems": ["x"]}, "no trial number"),
        ({"n": None, "status": "invalid"}, "problems are given"),
        ({"n": 1, "status": "rejected"}, "evidence is given"),
        ({"n": 1, "status": "seen", "evidence": EVIDENCE}, "evidence is given"),
        ({"n": 1, "status": "accepted", "evidence": EVIDENCE}, "names its version"),
        ({"n": 1, "status": "rejected", "evidence": EVIDENCE, "params_key": None}, "parameter key"),
    ],
)
def test_an_inconsistent_entry_is_refused(fields, message):
    base = {
        "time": TIME, "commit": "abc1234", "source": "llm", "parent": None, "changes": {},
        "params_key": "key",
    }  # fmt: skip
    with pytest.raises(ValueError, match=message):
        Entry(**(base | fields))


def test_the_shipped_log_is_valid_and_within_budget():
    log = ExperimentLog(Path(__file__).parents[2] / "configs")
    assert log.used() <= MAX_TRIALS
