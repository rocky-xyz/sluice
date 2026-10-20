from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from sluice.evaluation.decisions import Decision
from sluice.evaluation.final import NOT_RUN, evaluate_once, missing_runs, sealed_config, too_early
from sluice.evaluation.report import Calendar, Result, render
from sluice.evaluation.seal import AlreadyOpened, opened
from sluice.sizing.config import load_sizing_config

DAYS = [date(2026, 8, 10), date(2026, 8, 24)]
CAL = Calendar(decisions=DAYS, rebalances=DAYS, days=DAYS, end=date(2026, 10, 20))


def decision(ticker: str, day: date, repeat: int, rating: str | None = "hold") -> Decision:
    return Decision(
        ticker=ticker, day=day, repeat=repeat, config="v1-x", run_id=None, rating=rating,
        confidence=None,
    )  # fmt: skip


def test_missing_runs_counts_files_not_successes():
    full = [decision(t, d, r) for t in ("AMD", "KO") for d in DAYS for r in range(3)]
    assert missing_runs("v1", full, ["AMD", "KO"], CAL, 3) == []
    failed = [*full[:-1], decision("KO", DAYS[1], 2, rating=None)]  # a failed run is a file
    assert missing_runs("v1", failed, ["AMD", "KO"], CAL, 3) == []
    assert missing_runs("v1", full[:-1], ["AMD", "KO"], CAL, 3) == [
        "v1 repeat 2: 1 of 4 runs missing"
    ]
    assert missing_runs("ta", [], ["AMD", "KO"], CAL, 1) == ["ta repeat 0: 4 of 4 runs missing"]


def test_too_early_until_the_last_close():
    assert too_early(CAL, datetime(2026, 10, 20, 19, 0, tzinfo=UTC)) is not None  # 15:00 New York
    assert too_early(CAL, datetime(2026, 10, 20, 21, 0, tzinfo=UTC)) is None  # after the close


def test_the_sealed_config_names_pipeline_and_sizing():
    v1 = load_sizing_config("v1", Path(__file__).parents[2] / "configs")
    assert sealed_config("v1-97b47162", v1) == "v1-97b47162 + sizing v1-1bedf5b4"


def test_a_failed_computation_leaves_the_segment_sealed(tmp_path):
    path = tmp_path / "seals.json"

    def broken() -> str:
        raise ValueError("no prices for KO")

    with pytest.raises(ValueError, match="no prices"):
        evaluate_once("test_1", broken, "cfg", "abc", path)
    assert opened(path) == []
    seal, report = evaluate_once("test_1", lambda: "the report", "cfg", "abc", path)
    assert (seal.segment, seal.config, seal.commit, report) == (
        "test_1",
        "cfg",
        "abc",
        "the report",
    )
    assert [s.segment for s in opened(path)] == ["test_1"]


def test_a_second_evaluation_is_refused_before_anything_is_computed(tmp_path):
    path = tmp_path / "seals.json"
    evaluate_once("test_1", lambda: "first", "cfg", "abc", path)
    calls = []
    with pytest.raises(AlreadyOpened):
        evaluate_once("test_1", lambda: calls.append(1) or "again", "cfg", "abc", path)
    assert calls == []


def test_the_missing_baseline_row_only_when_it_is_missing():
    result = Result("x", 0.001, 1.0, 1.0, 0.1, 0.1, None, 0.001)
    assert "| 原版 TradingAgents | 缺（推迟到 7.7" in render([result], CAL, "validation")
    assert "| 原版 TradingAgents | 缺（为控制花费" in render(
        [result], CAL, "test_1", missing_baseline=NOT_RUN
    )
    assert "原版 TradingAgents" not in render([result], CAL, "test_1", missing_baseline=None)
