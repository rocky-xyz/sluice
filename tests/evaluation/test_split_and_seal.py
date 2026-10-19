import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from sluice.agents.batch import FIRST_DAY, decision_days
from sluice.evaluation.decisions import load_decisions
from sluice.evaluation.seal import AlreadyOpened, open_seal, opened
from sluice.evaluation.split import (
    TEST_1,
    TEST_2_FROM,
    VALIDATION,
    SealedError,
    require_open,
    segment,
)

NOW = datetime(2026, 12, 1, tzinfo=UTC)


# --- the split ---


def test_segments_are_in_time_order_and_do_not_overlap():
    assert VALIDATION[0] <= VALIDATION[1] < TEST_1[0] <= TEST_1[1] < TEST_2_FROM


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2026, 1, 1), None),
        (date(2026, 1, 2), "validation"),
        (date(2026, 7, 27), "validation"),
        (date(2026, 7, 28), None),
        (date(2026, 8, 10), "test_1"),
        (date(2026, 10, 6), "test_1"),
        (date(2026, 10, 19), None),
        (date(2026, 10, 20), "test_2"),
        (date(2027, 3, 1), "test_2"),
    ],
)
def test_boundaries(day, expected):
    assert segment(day) == expected


def test_the_decision_grid_falls_15_validation_5_test_1_then_test_2():
    days = decision_days(FIRST_DAY, date(2026, 11, 30))
    by_segment = [segment(d) for d in days]
    assert by_segment.count("validation") == 15
    assert by_segment.count("test_1") == 5
    assert None not in by_segment  # every grid day belongs to a segment
    assert days[20] == TEST_2_FROM  # the first decision after test one opens test two


def test_validation_is_always_open_and_tests_only_when_final():
    require_open("validation")
    for sealed in ("test_1", "test_2"):
        with pytest.raises(SealedError):
            require_open(sealed)
        require_open(sealed, final=True)


# --- opening the seals ---


def test_each_test_segment_opens_once(tmp_path):
    path = tmp_path / "seals.json"
    open_seal("test_1", "v2-ed2e4c63", "abc123", path, now=lambda: NOW)
    with pytest.raises(AlreadyOpened):
        open_seal("test_1", "v2-ed2e4c63", "abc123", path, now=lambda: NOW)
    [seal] = opened(path)
    assert (seal.segment, seal.config, seal.commit, seal.opened_at) == (
        "test_1", "v2-ed2e4c63", "abc123", NOW,
    )  # fmt: skip


def test_test_two_needs_test_one_first_and_the_same_config(tmp_path):
    path = tmp_path / "seals.json"
    with pytest.raises(SealedError, match="after test_1"):
        open_seal("test_2", "v2-ed2e4c63", "abc", path)
    open_seal("test_1", "v2-ed2e4c63", "abc", path)
    with pytest.raises(SealedError, match="frozen at test_1"):
        open_seal("test_2", "v3-11111111", "def", path)
    open_seal("test_2", "v2-ed2e4c63", "def", path)
    assert [s.segment for s in opened(path)] == ["test_1", "test_2"]


def test_validation_has_no_seal_to_open(tmp_path):
    with pytest.raises(ValueError):
        open_seal("validation", "v1-97b47162", "abc", tmp_path / "seals.json")


# --- reading decisions ---


def _state(folder: Path, ticker: str, day: str, repeat: int, rating: str | None) -> None:
    path = folder / ticker / f"{day}_r{repeat}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = None if rating is None else {"rating": rating, "confidence": 0.6}
    path.write_text(json.dumps({"decision": decision}))


def test_validation_decisions_are_read_and_test_files_never_opened(tmp_path):
    folder = tmp_path / "v1-97b47162"
    _state(folder, "NVDA", "2026-03-03", 0, "overweight")
    _state(folder, "AMD", "2026-01-02", 0, None)
    (folder / "NVDA" / "2026-08-10_r0.json").write_text("opening this would fail")
    (folder / "NVDA" / "2026-10-20_r0.json").write_text("so would this")

    decisions = load_decisions(folder, "validation")
    assert [(d.ticker, str(d.day), d.rating, d.confidence) for d in decisions] == [
        ("AMD", "2026-01-02", None, None),
        ("NVDA", "2026-03-03", "overweight", 0.6),
    ]
    assert {d.config for d in decisions} == {"v1-97b47162"}


def test_test_decisions_need_final(tmp_path):
    folder = tmp_path / "v1-97b47162"
    _state(folder, "NVDA", "2026-08-10", 0, "hold")
    with pytest.raises(SealedError):
        load_decisions(folder, "test_1")
    assert [d.rating for d in load_decisions(folder, "test_1", final=True)] == ["hold"]


# --- the evolution layer ---


def test_the_evolution_layer_never_touches_the_split_or_the_seals():
    root = Path(__file__).parents[2] / "sluice" / "evolution"
    for source in root.rglob("*.py"):
        text = source.read_text()
        assert "evaluation.split" not in text and "evaluation.seal" not in text, source
