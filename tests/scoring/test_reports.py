import json
from datetime import date
from pathlib import Path

import pytest

from sluice.evaluation.split import SealedError
from sluice.scoring.client import JevError, JevResponse
from sluice.scoring.config import load_report_scoring_config
from sluice.scoring.reports import (
    ReportScoreStore,
    load_reports,
    report_score,
    score_reports,
    state_text,
)

ROOT = Path(__file__).parents[2] / "configs"
CONFIG = load_report_scoring_config("reports-v1", ROOT)


def stance(bullish: float, bearish: float) -> dict:
    neutral = round(1 - bullish - bearish, 6)
    probabilities = {"bullish": bullish, "bearish": bearish, "neutral": neutral}
    return {
        "stance": {
            "type": "choice",
            "choice": max(probabilities, key=probabilities.get),
            "probabilities": probabilities,
            "confidence": 0.8,
        }
    }


class FakeJev:
    """Answers by report: the state's 'Report by' line picks the answer."""

    def __init__(self, answers: dict[str, object]):
        self.answers = answers
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        for name, answer in self.answers.items():
            if f"Report by: {name} analyst" in request.state:
                if isinstance(answer, Exception):
                    raise answer
                return JevResponse(answers=answer, input_tokens=100)
        raise AssertionError(request.state)


def save_run(folder: Path, ticker: str, day: str, reports: dict[str, str | None]) -> None:
    path = folder / ticker / f"{day}_r0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "meta": {"run_id": f"{ticker}_{day}_r0"},
        "reports": {
            name: {"text": text or "No data.", "no_data": text is None}
            for name, text in reports.items()
        },
    }
    path.write_text(json.dumps(state))


def test_the_shipped_report_spec_loads():
    assert CONFIG.version == "reports-v1"
    assert set(CONFIG.questions["stance"].criteria) == {"bullish", "bearish", "neutral"}


def test_only_validation_reports_with_data_are_read(tmp_path):
    save_run(tmp_path, "NVDA", "2026-03-03", {"market": "Up [P:ret_5d].", "news": None})
    (tmp_path / "NVDA" / "2026-08-10_r0.json").write_text("a test-period file, never opened")
    reports = load_reports(tmp_path, "validation")
    assert [(r.ticker, r.day, r.report) for r in reports] == [("NVDA", date(2026, 3, 3), "market")]
    with pytest.raises(SealedError):
        load_reports(tmp_path, "test_1")


def test_the_state_names_the_stock_and_the_analyst(tmp_path):
    save_run(tmp_path, "NVDA", "2026-03-03", {"news": "Demand is strong [N1]."})
    [item] = load_reports(tmp_path, "validation")
    assert state_text(item) == "Stock: NVDA\nReport by: news analyst\n\nDemand is strong [N1]."


def test_reports_are_scored_once_and_averaged(tmp_path):
    save_run(tmp_path, "NVDA", "2026-03-03", {"market": "a", "news": "b", "fundamentals": None})
    items = load_reports(tmp_path, "validation")
    jev = FakeJev({"market": stance(0.7, 0.1), "news": stance(0.2, 0.6)})
    with ReportScoreStore(tmp_path / "scores.db") as store:
        result = score_reports(items, store, CONFIG, jev)
        assert (result.scored, result.done, result.input_tokens) == (2, 0, 200)
        assert report_score(store, "NVDA_2026-03-03_r0", CONFIG.version) == pytest.approx(
            ((0.7 - 0.1) + (0.2 - 0.6)) / 2
        )
        again = score_reports(items, store, CONFIG, jev)
        assert (again.scored, again.done) == (0, 2)
        assert len(jev.requests) == 2


def test_failures_are_retried_later_and_too_large_is_remembered(tmp_path):
    save_run(tmp_path, "NVDA", "2026-03-03", {"market": "a", "news": "b"})
    items = load_reports(tmp_path, "validation")
    jev = FakeJev(
        {"market": JevError("busy", retryable=True), "news": JevError("422", too_large=True)}
    )
    with ReportScoreStore(tmp_path / "scores.db") as store:
        result = score_reports(items, store, CONFIG, jev)
        assert (result.failed, result.skipped) == (1, 1)
        assert report_score(store, "NVDA_2026-03-03_r0", CONFIG.version) is None
        later = FakeJev({"market": stance(0.5, 0.3), "news": AssertionError("asked again")})
        assert score_reports(items, store, CONFIG, later).scored == 1


def test_a_run_without_scored_reports_has_no_score(tmp_path):
    with ReportScoreStore(tmp_path / "scores.db") as store:
        assert report_score(store, "nothing", CONFIG.version) is None
