import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from sluice.agents.cache import CallCache
from sluice.agents.llm import CachedLLM, LLMError, LLMResponse
from sluice.evaluation.backtest import PriceBar
from sluice.evaluation.report import Calendar
from sluice.evolution.experiments import Entry, Evidence
from sluice.evolution.proposer import SYSTEM, ask, build_request
from sluice.evolution.validation import Period, Validation
from sluice.sizing.config import load_sizing_config

CONFIGS = Path(__file__).parents[2] / "configs"
V1 = load_sizing_config("v1", CONFIGS)
TEXT = (CONFIGS / "sizing" / "v1.toml").read_text()
TICKERS = ["AMD", "KO"]
TIME = datetime(2026, 10, 20, tzinfo=UTC)


def period(day: int, amd: float, ko: float, spy: float) -> Period:
    weights = {"AMD": 0.4, "KO": 0.5}
    contributions = {"AMD": 0.4 * amd, "KO": 0.5 * ko}
    return Period(
        start=date(2026, 1, day), exposure=0.9, weights=weights, contributions=contributions,
        portfolio=sum(contributions.values()), benchmark=spy, equal=(amd + ko) / 2,
    )  # fmt: skip


PERIODS = [period(5, 0.10, 0.0, 0.02), period(20, -0.10, 0.01, 0.01), period(30, 0.0, 0.0, 0.03)]
RETURNS = {r: [0.01, -0.005, 0.002] * 50 for r in range(3)}
LOG = [
    Entry(
        n=1, time=TIME, commit="abc", source="6.12", parent=None, changes={},
        params_key="aaa", status="seen", version="v1", rationale="6.12：初始冠军",
    ),
    Entry(
        n=2, time=TIME, commit="abc", source="6.12", parent="v1-1bedf5b4", changes={"k": 0.0},
        params_key="bbb", status="rejected",
        evidence=Evidence(
            candidate_sharpes=[1.64, 1.64, 1.64], champion_sharpes=[1.53, 1.6, 1.58],
            candidate_sharpe=1.64, champion_sharpe=1.57, rho=0.98, z=0.9, days=150,
        ),
    ),
    Entry(
        n=None, time=TIME, commit="abc", source="llm", parent="v1-1bedf5b4",
        changes={"k": 9}, params_key=None, status="invalid", problems=["k: 9 outside 0..2"],
    ),
]  # fmt: skip


def request():
    return build_request(V1, TEXT, 17, LOG, PERIODS, RETURNS, TICKERS)


def test_the_prompt_holds_the_rules_the_config_the_log_and_the_material():
    system, user = (m.content for m in request().messages)
    assert "z >= 3" in system and "every one of the\n  3 repeats" in system
    assert "20 trials" in system and "empty `changes`" in system
    assert "Trials left: 17 of 20" in user
    assert "k = 0.5  # tilt strength" in user  # the config file, comments and all
    assert "- stocks.vol_lookback: int from 10 to 120" in user
    assert "- stocks.vol_scaling: true or false" in user
    assert "#2 rejected, by 6.12" in user and "z +0.90" in user
    assert "refused: k: 9 outside 0..2" in user
    assert "| 2026-01-20 | 0.90 | -3.5% | +1.0% | -4.5% | -4.5% |" in user
    assert "| total | +0 | +50 |" in user
    assert "- 2026-01-20 (-4.5% vs SPY): AMD -400bp, KO +50bp" in user
    assert request().json_mode and request().model == "deepseek-v4-pro"


def test_the_prompt_gives_no_conclusions():
    text = SYSTEM + request().messages[1].content
    for hint in ("no skill", "6.12 report", "signals do not", "set k to 0"):
        assert hint not in text


class Script:
    def __init__(self, *texts: str):
        self.texts, self.calls = list(texts), 0

    def complete(self, request):
        self.calls += 1
        return LLMResponse(text=self.texts.pop(0), input_tokens=100, output_tokens=10)


@pytest.fixture
def cache(tmp_path):
    with CallCache(tmp_path / "calls.db") as c:
        yield c


def test_a_proposal_is_parsed_and_cached(cache):
    answer = json.dumps({"changes": {"k": 0.2}, "rationale": "降低倾斜"}, ensure_ascii=False)
    client = Script(answer)
    proposal, _ = ask(CachedLLM(cache, client), request())
    assert (proposal.changes, proposal.rationale) == ({"k": 0.2}, "降低倾斜")
    again, replayed = ask(CachedLLM(cache, None), request())  # a dry run, then the real one
    assert again == proposal and replayed[0].cached and client.calls == 1


def test_a_broken_answer_is_retried_and_rule_breaking_ones_pass_through(cache):
    rule_breaking = json.dumps({"changes": {"exposure.benchmark": "QQQ"}, "rationale": "x"})
    client = Script("not json", rule_breaking)
    proposal, answers = ask(CachedLLM(cache, client, sleep=lambda s: None), request())
    assert proposal.changes == {"exposure.benchmark": "QQQ"} and len(answers) == 2


def test_proposing_nothing_is_allowed_but_three_broken_answers_are_not(cache):
    none = json.dumps({"changes": {}, "rationale": "没有值得试的改动"}, ensure_ascii=False)
    proposal, _ = ask(CachedLLM(cache, Script(none)), request())
    assert proposal.changes == {}
    other = build_request(V1, TEXT, 16, LOG, PERIODS, RETURNS, TICKERS)
    with pytest.raises(LLMError, match="no acceptable answer"):
        ask(CachedLLM(cache, Script("{}", "[]", "nope")), other)


def test_periods_average_the_repeats_and_measure_open_to_open():
    days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    validation = Validation.__new__(Validation)  # no stores: the plan is given below
    validation.tickers = TICKERS
    validation.cal = Calendar(decisions=days[:2], rebalances=days[:2], days=days, end=days[2])
    bar = PriceBar
    validation.bars = {
        "AMD": {days[0]: bar(100, 101), days[1]: bar(110, 111), days[2]: bar(120, 99)},
        "KO": {days[0]: bar(50, 50), days[1]: bar(50, 51), days[2]: bar(52, 55)},
        "SPY": {days[0]: bar(400, 401), days[1]: bar(404, 404), days[2]: bar(410, 412)},
    }

    class M:
        exposure = 0.8

    schedules = {
        0: {days[0]: {"AMD": 0.6, "KO": 0.2}, days[1]: {"AMD": 0.4}},
        1: {days[0]: {"AMD": 0.2, "KO": 0.6}, days[1]: {"KO": 0.4}},
    }
    validation._plan = lambda config: ({d: M for d in days[:2]}, schedules)
    first, second = validation.periods(V1)
    assert first.weights == pytest.approx({"AMD": 0.4, "KO": 0.4})
    assert first.contributions["AMD"] == pytest.approx(0.4 * 0.10)  # open 100 -> next open 110
    assert first.benchmark == pytest.approx(0.01) and first.equal == pytest.approx(0.05)
    assert second.contributions["AMD"] == pytest.approx(0.2 * (99 / 110 - 1))  # to the last close
    assert second.exposure == 0.8
