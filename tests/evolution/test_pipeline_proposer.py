import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from sluice.agents.cache import CallCache
from sluice.agents.config import load_config
from sluice.agents.llm import CachedLLM, LLMError, LLMResponse
from sluice.evolution.pipeline_log import PipelineEntry, ScreenEvidence
from sluice.evolution.pipeline_proposer import (
    PROPOSABLE,
    SYSTEM,
    Call,
    Proposal,
    ask,
    build_request,
    load,
    save,
    unseen_days,
)

V3 = load_config("v3", Path(__file__).parents[2] / "configs")
DAYS = [date(2026, 1, 2) + timedelta(days=14 * i) for i in range(15)]
TIME = datetime(2026, 10, 21, tzinfo=UTC)


def state(rating: str | None, news: str = "Demand is strong [N1].") -> dict:
    return {
        "reports": {"market": {"text": "Up [P:ret_5d]."}, "news": {"text": news}},
        "decision": None if rating is None else {
            "rating": rating, "confidence": 0.6, "rationale": f"Rated {rating}.",
        },
        "retrieved": [
            {"role": "market_analyst", "round": 0, "tool": "get_indicators", "arguments": {}},
            {"role": "news_analyst", "round": 0, "tool": "search_news",
             "arguments": {"query": "export controls"}},
        ],
    }  # fmt: skip


CALLS = [
    Call("AAA", DAYS[1], state("overweight", "Big miss [N7]."), -0.08),  # worst
    Call("BBB", DAYS[2], state("underweight"), 0.03),
    Call("CCC", DAYS[4], state("overweight"), 0.02),  # a hit
    Call("DDD", DAYS[5], state("hold"), -0.05),  # hold: never a miss
    Call("EEE", DAYS[7], state(None), 0.01),  # failed run
]
LOG = [
    PipelineEntry(
        stage="screen", n=1, time=TIME, commit="abc", source="llm", parent="v3-49da8839",
        layer="rules", changes={"tools.max_calls": 4}, params_key="k1", status="failed",
        rationale="少调几次", evidence=ScreenEvidence(
            cells=55, candidate_spearman=0.01, champion_spearman=0.05,
            candidate_failed=0, champion_failed=1,
        ),
    )
]  # fmt: skip


def request():
    return build_request(V3, {"screen": 29, "full": 5}, LOG, CALLS)


def test_the_material_comes_only_from_days_the_screen_does_not_use():
    unseen = unseen_days(DAYS)
    assert len(unseen) == 10
    assert not {DAYS[i] for i in (0, 3, 6, 9, 12)} & set(unseen)


def test_the_request_holds_rules_prompts_log_and_the_worst_calls():
    system, user = (m.content for m in request().messages)
    assert "on decision days you are not shown" in system and "z >= 3" in system
    assert "Screens left: 29 of 30; full evaluations left: 5 of 5" in user
    assert "- tools.max_calls: now 6; int 1..10" in user
    assert "- tools.pass_reasoning: now False; true or false" in user
    for role in PROPOSABLE:
        assert f"## {role} (" in user
    assert "## bull" not in user
    assert "screen 1 failed against v3-49da8839: tools.max_calls=4" in user
    assert "Spearman 0.01 vs 0.05" in user and "rationale: 少调几次" in user
    assert "directional ratings whose sign matched" in user and "1 of 3 (33%)" in user
    assert "'export controls'" in user and "news_analyst: 1.0 tool calls per report" in user
    # worst first; a hold and a failed run are never shown as misses
    assert user.index("### AAA") < user.index("### BBB")
    assert "### DDD" not in user and "### EEE" not in user and "### CCC" not in user
    assert "Big miss [N7]." in user and "manager's rationale: Rated overweight." in user


def test_the_request_gives_no_conclusions():
    text = SYSTEM + request().messages[1].content
    for hint in ("no skill", "6.12", "set k to 0", "signals do not"):
        assert hint not in text


def test_only_the_four_roles_may_be_proposed():
    Proposal(changes={"prompts.news_analyst": "New."}, rationale="x")
    Proposal(changes={}, rationale="不值得试")
    with pytest.raises(ValueError, match="may not be proposed"):
        Proposal(changes={"prompts.bull": "Argue."}, rationale="x")


class Script:
    def __init__(self, *texts):
        self.texts = list(texts)

    def complete(self, request):
        return LLMResponse(text=self.texts.pop(0))


def test_a_proposal_for_a_forbidden_role_is_asked_again(tmp_path):
    bad = json.dumps({"changes": {"prompts.bear": "x"}, "rationale": "x"})
    good = json.dumps({"changes": {"debate_rounds": 0}, "rationale": "省掉辩论"})
    with CallCache(tmp_path / "c.db") as cache:
        proposal, answers = ask(
            CachedLLM(cache, Script(bad, good), sleep=lambda s: None), request()
        )
    assert proposal.changes == {"debate_rounds": 0} and len(answers) == 2
    with CallCache(tmp_path / "d.db") as cache, pytest.raises(LLMError):
        ask(CachedLLM(cache, Script(bad, bad, bad), sleep=lambda s: None), request())


def test_a_saved_proposal_loads_back_and_an_edit_is_noticed(tmp_path):
    text = 'New prompt with "quotes" and a \\ backslash.\n'
    proposal = Proposal(changes={"prompts.news_analyst": text}, rationale="更聚焦")
    folder = save(proposal, V3, tmp_path)
    changes, why, edited = load(folder)
    assert (changes, why, edited) == ({"prompts.news_analyst": text}, "更聚焦", False)
    (folder / "news_analyst.txt").write_text(text + "One more rule.\n")
    assert load(folder)[2] is True
    rules = save(Proposal(changes={"tools.max_calls": 4}, rationale="r"), V3, tmp_path / "b")
    assert load(rules) == ({"tools.max_calls": 4}, "r", False)
