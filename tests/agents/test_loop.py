import json
from datetime import timedelta
from typing import ClassVar

import pytest

from sluice.agents.cache import CallCache
from sluice.agents.config import ToolsConfig
from sluice.agents.graph import run
from sluice.agents.llm import CachedLLM, LLMError, LLMRequest, LLMResponse, ToolCall
from sluice.agents.loop import LIMIT_REACHED, NO_DATA, answer_followups, tool_analyst
from sluice.agents.state import Meta, Question, Report, cited_refs, dump_state
from sluice.agents.tools import ToolContext
from sluice.fundamentals.store import FundamentalsStore
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore
from tests.agents.test_pipeline import (
    CLOSE,
    CONFIG,
    ROLE_OF_PROMPT,
    FakeModel,
    article,
    pipeline,
    verdict,
)

TOOLED = CONFIG.model_copy(update={"tools": ToolsConfig()})


@pytest.fixture
def stocked(tmp_path):
    """Two stories before the close, one at the close, one after (as in test_pipeline)."""
    with NewsStore(tmp_path / "news.db") as store:
        store.add(
            [
                article(1, CLOSE - timedelta(days=2)),
                article(2, CLOSE - timedelta(seconds=1)),
                article(3, CLOSE),
                article(4, CLOSE + timedelta(hours=1)),
            ]
        )
        yield store


def ask(*calls: tuple[str, dict]) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=tuple(
            ToolCall(id=f"call_{i}", name=name, arguments=json.dumps(args))
            for i, (name, args) in enumerate(calls)
        ),
        finish_reason="tool_calls",
        input_tokens=100,
        output_tokens=10,
    )


def say(text: str) -> LLMResponse:
    return LLMResponse(text=text, finish_reason="stop", input_tokens=100, output_tokens=20)


def returned(request: LLMRequest) -> list[str]:
    """Ids in the tool results so far, in order."""
    return [
        ref for m in request.messages if m.role == "tool" for ref in sorted(cited_refs(m.content))
    ]


def tool_results(request: LLMRequest) -> int:
    return sum(m.role == "tool" for m in request.messages)


class Agents:
    """Analysts that call one tool, then write a report citing what came back."""

    TOOL: ClassVar[dict[str, tuple[str, dict]]] = {
        "market_analyst": ("get_indicators", {}),
        "news_analyst": ("search_news", {"query": "chipmaker"}),
        "fundamentals_analyst": ("get_fundamentals", {"quarters": 2}),
    }

    def __init__(self, **overrides):
        self.overrides = overrides  # role -> function of the request
        self.requests: list[tuple[str, LLMRequest]] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        role = ROLE_OF_PROMPT[request.messages[0].content]
        self.requests.append((role, request))
        if role in self.overrides:
            return self.overrides[role](request)
        if role == "research_manager":
            return say(verdict(*sorted(cited_refs(request.messages[1].content))[:1]))
        if not tool_results(request):
            return ask(self.TOOL[role])
        ids = returned(request)
        return say(f"Seen [{ids[0]}]." if ids else "Nothing came back.")

    def asked(self, role: str) -> list[LLMRequest]:
        return [r for asked, r in self.requests if asked == role]


def go(graph, config=TOOLED, repeat=0):
    return run(graph, config, "NVDA", CLOSE, repeat=repeat)


def test_each_analyst_fetches_with_its_own_tools_and_cites_what_came_back(stocked, tmp_path):
    model = Agents()
    state = go(pipeline(stocked, tmp_path, model, config=TOOLED))
    assert state["meta"].status == "ok" and state["decision"] is not None
    assert state["inputs"].prices == [] and state["inputs"].news == []  # nothing prefetched
    used = {(u.role, u.tool) for u in state["retrieved"]}
    assert used == {
        ("market_analyst", "get_indicators"),
        ("news_analyst", "search_news"),
        ("fundamentals_analyst", "get_fundamentals"),
    }
    first = model.asked("market_analyst")[0]
    assert [spec.name for spec in first.tools] == ["get_indicators", "get_price_history"]
    assert [spec.name for spec in model.asked("news_analyst")[0].tools] == ["search_news"]
    assert "Decision time: 2026-09-30 16:00 ET" in first.messages[1].content
    # the news search never sees the item published at the close
    news = state["retrieved"][[u.role for u in state["retrieved"]].index("news_analyst")]
    assert all(item.published_at < CLOSE for item in news.items)
    assert sum(c.role == "market_analyst" for c in state["usage"]) == 2  # call, then report


def test_the_manager_may_cite_what_the_reports_cite_from_tools(stocked, tmp_path):
    state = go(pipeline(stocked, tmp_path, Agents(), config=TOOLED))
    cited = {ref for e in state["decision"].evidence for ref in e.refs}
    assert cited and cited <= {ref for u in state["retrieved"] for ref in u.refs}


def test_past_the_call_limit_the_tools_are_switched_off(stocked, tmp_path):
    config = CONFIG.model_copy(update={"tools": ToolsConfig(max_calls=2)})

    def greedy(request):
        if request.tool_choice == "none":
            return say(f"Done [{returned(request)[0]}].")
        return ask(*[("get_price_history", {"days": d}) for d in (5, 10, 20)])  # three at once

    model = Agents(market_analyst=greedy)
    state = go(pipeline(stocked, tmp_path, model, config=config), config)
    calls = [u for u in state["retrieved"] if u.role == "market_analyst"]
    assert [u.arguments["days"] for u in calls] == [5, 10]
    second = model.asked("market_analyst")[1]
    assert second.tool_choice == "none"
    assert second.messages[-1].content == LIMIT_REACHED  # the third call still gets an answer
    assert state["reports"]["market"].text.startswith("Done")


def test_a_report_citing_what_was_not_returned_is_sent_back_with_the_reason(stocked, tmp_path):
    def sloppy(request):
        if not tool_results(request):
            return ask(("get_indicators", {}))
        if request.tool_choice == "none":
            return say("Fixed [P:rsi_14].")
        return say("Momentum [P:rsi_14] and news [N999999].")

    model = Agents(market_analyst=sloppy)
    state = go(pipeline(stocked, tmp_path, model, config=TOOLED))
    fix = model.asked("market_analyst")[2]
    assert fix.tool_choice == "none"
    assert "ids the tools did not return: N999999" in fix.messages[-1].content
    assert state["reports"]["market"].text == "Fixed [P:rsi_14]."


def test_a_report_that_is_never_fixed_fails_the_run(stocked, tmp_path):
    def stubborn(request):
        return ask(("get_indicators", {})) if not tool_results(request) else say("No ids.")

    state = go(pipeline(stocked, tmp_path, Agents(market_analyst=stubborn), config=TOOLED))
    assert state["meta"].status == "failed"
    assert "market_analyst: no acceptable report after 3 tries" in state["meta"].failure


def test_a_rerun_replays_every_step_from_the_cache(stocked, tmp_path):
    go(pipeline(stocked, tmp_path, Agents(), config=TOOLED, checkpoints=False))
    again = go(pipeline(stocked, tmp_path, Agents(), config=TOOLED, checkpoints=False))
    assert again["usage"] and all(call.cached for call in again["usage"])


def test_tool_calls_are_saved_with_the_run_and_v1_states_stay_as_they_were(stocked, tmp_path):
    tooled = dump_state(go(pipeline(stocked, tmp_path, Agents(), config=TOOLED)))
    assert {u["tool"] for u in tooled["retrieved"]} == {
        "get_indicators", "search_news", "get_fundamentals"
    }  # fmt: skip
    plain = dump_state(run(pipeline(stocked, tmp_path / "v1", Agents()), CONFIG, "NVDA", CLOSE))
    assert "retrieved" not in plain


def test_an_analyst_whose_tools_return_nothing_writes_no_data(tmp_path):
    with NewsStore(tmp_path / "n.db") as news, PriceStore(tmp_path / "p.db") as prices:
        ctx = ToolContext("NVDA", CLOSE, news, prices, FundamentalsStore(tmp_path / "s"), TOOLED)
        meta = Meta(
            run_id="x", ticker="NVDA", decision_time=CLOSE, mode="backtest",
            config_version="t1", config_digest="abc", repeat=0,
        )  # fmt: skip
        model = Agents()
        with CallCache(tmp_path / "c.db") as cache:
            out = tool_analyst({"meta": meta}, CachedLLM(cache, model), TOOLED, "news_analyst", ctx)
    assert out["reports"]["news"].no_data and out["reports"]["news"].text == NO_DATA["news_analyst"]


@pytest.mark.parametrize("flag", [False, True])
def test_thinking_is_sent_back_only_when_the_rules_say_so(stocked, tmp_path, flag):
    config = CONFIG.model_copy(update={"tools": ToolsConfig(pass_reasoning=flag)})

    def thinking(request):
        if not tool_results(request):
            first = ask(("get_indicators", {}))
            return first.model_copy(update={"reasoning": "I should look at momentum."})
        return say(f"Seen [{returned(request)[0]}].")

    model = Agents(market_analyst=thinking)
    go(pipeline(stocked, tmp_path, model, config=config), config)
    turn = model.asked("market_analyst")[1].messages[2]
    assert turn.reasoning_content == ("I should look at momentum." if flag else None)


# --- follow-up questions (task 9.5) ---


def manager_asking(*questions: tuple[str, str], again: bool = False):
    """A manager that asks once, then decides citing what it may; or asks every time."""

    def answer(request):
        prompt = request.messages[1].content
        cited = sorted(cited_refs(prompt))
        draft = json.loads(verdict(cited[0]))
        if "No more questions can be asked" in prompt and not again:
            return say(verdict(*cited[-1:], rating="hold"))
        if questions and "you may send each analyst" in prompt or again:
            draft["followups"] = [{"analyst": a, "question": q} for a, q in questions]
        return say(json.dumps(draft))

    return answer


def news_answering(request):
    text = request.messages[1].content
    if "The research manager asks" in text and not tool_results(request):
        return ask(("search_news", {"query": "headline", "days": 3}))
    if "The research manager asks" in text:
        return say(f"Checked again [{returned(request)[-1]}].")
    if not tool_results(request):
        return ask(("search_news", {"query": "chipmaker"}))
    return say(f"Seen [{returned(request)[0]}].")


def test_the_manager_asks_an_analyst_answers_with_its_tools_and_the_manager_decides(
    stocked, tmp_path
):
    model = Agents(
        research_manager=manager_asking(("news", "Is the latest story confirmed?")),
        news_analyst=news_answering,
    )
    state = go(pipeline(stocked, tmp_path, model, config=TOOLED))
    assert state["meta"].status == "ok"
    first, second = model.asked("research_manager")
    assert "you may send each analyst at most one question" in first.messages[1].content
    assert "at most 3 more calls each" in first.messages[1].content
    assert "## follow-up, round 1, news analyst" in second.messages[1].content
    assert "No more questions can be asked" in second.messages[1].content
    (followup,) = state["followups"]
    assert (followup.round, followup.analyst, followup.answered) == (1, "news", True)
    asked = [
        r for r in model.asked("news_analyst") if "research manager asks" in r.messages[1].content
    ]
    assert "Is the latest story confirmed?" in asked[0].messages[1].content
    assert "Your report:\nSeen [" in asked[0].messages[1].content
    assert [s.name for s in asked[0].tools] == ["search_news"]
    extra = [u for u in state["retrieved"] if u.round == 1]
    assert [(u.role, u.step) for u in extra] == [("news_analyst", 2)]
    # the preliminary view is kept; the decision comes after the answer
    assert state["preliminary"].rating == "overweight" and state["decision"].rating == "hold"
    saved = dump_state(state)
    assert saved["followups"][0]["question"] == "Is the latest story confirmed?"
    assert saved["preliminary"]["rating"] == "overweight"


def test_without_questions_the_manager_decides_at_once(stocked, tmp_path):
    model = Agents(research_manager=manager_asking())
    state = go(pipeline(stocked, tmp_path, model, config=TOOLED))
    assert len(model.asked("research_manager")) == 1
    assert "followups" not in dump_state(state) and state["decision"] is not None


def test_a_second_round_of_questions_is_refused(stocked, tmp_path):
    asks_forever = manager_asking(("news", "And now?"), again=True)
    model = Agents(research_manager=asks_forever, news_analyst=news_answering)
    state = go(pipeline(stocked, tmp_path, model, config=TOOLED))
    assert state["meta"].status == "failed"
    assert "no more follow-up questions can be asked" in state["meta"].failure
    assert len(state["followups"]) == 1


def test_an_unanswerable_question_is_noted_and_the_run_goes_on(stocked, tmp_path):
    def mute(request):
        if "The research manager asks" in request.messages[1].content:
            return say("I cannot tell.")  # never cites anything
        return news_answering(request)

    model = Agents(research_manager=manager_asking(("news", "Why?")), news_analyst=mute)
    state = go(pipeline(stocked, tmp_path, model, config=TOOLED))
    (followup,) = state["followups"]
    assert not followup.answered and followup.answer.startswith("The analyst could not answer")
    assert state["meta"].status == "ok" and state["decision"] is not None
    second = model.asked("research_manager")[1].messages[1].content
    assert "news analyst (not answered)" in second


def test_two_questions_to_one_analyst_are_asked_again(stocked, tmp_path):
    def twice(request):
        draft = json.loads(verdict(min(cited_refs(request.messages[1].content))))
        draft["followups"] = [
            {"analyst": "news", "question": "A?"},
            {"analyst": "news", "question": "B?"},
        ]
        return say(json.dumps(draft))

    model = Agents(research_manager=twice)
    state = go(pipeline(stocked, tmp_path, model, config=TOOLED))
    assert state["meta"].status == "failed" and "one question per analyst" in state["meta"].failure


def test_no_follow_ups_without_rounds_or_without_tools(stocked, tmp_path):
    config = CONFIG.model_copy(update={"tools": ToolsConfig(followup_rounds=0)})
    model = Agents(research_manager=manager_asking())
    go(pipeline(stocked, tmp_path, model, config=config), config)
    assert "you may send" not in model.asked("research_manager")[0].messages[1].content
    v1 = FakeModel()  # prefetching analysts, as in test_pipeline
    run(pipeline(stocked, tmp_path / "v1", v1), CONFIG, "NVDA", CLOSE)
    prompt = v1.asked("research_manager")[0].messages[1].content
    assert "you may send" not in prompt and "No more questions" not in prompt


def test_the_shipped_v3_uses_tools_and_keeps_v1s_debate_prompts():
    from sluice.agents.config import load_config

    v1, v3 = load_config("v1"), load_config("v3")
    assert v3.tools == ToolsConfig(max_calls=6, followup_rounds=1, followup_calls=3)
    assert (v1.digest, v1.tools) == ("97b47162", None)
    assert v3.models == v1.models and v3.debate_rounds == v1.debate_rounds
    assert v3.prompts["bull"] == v1.prompts["bull"] and v3.prompts["bear"] == v1.prompts["bear"]
    for role in ("market_analyst", "news_analyst", "fundamentals_analyst"):
        assert "limited number of tool calls" in v3.prompts[role]


# --- paths that need a failing model (added after the step-9 coverage check) ---


def test_a_call_that_finally_fails_ends_the_run_and_keeps_what_was_fetched(stocked, tmp_path):
    def breaks(request):
        if not tool_results(request):
            return ask(("get_indicators", {}))
        raise LLMError("HTTP 400: bad request")  # not retryable

    state = go(pipeline(stocked, tmp_path, Agents(market_analyst=breaks), config=TOOLED))
    assert state["meta"].status == "failed"
    assert state["meta"].failure == "market_analyst: HTTP 400: bad request"
    fetched = [u for u in state["retrieved"] if u.role == "market_analyst"]
    assert [u.tool for u in fetched] == ["get_indicators"]
    assert [c.role for c in state["usage"]].count("market_analyst") == 1  # the paid answer


def test_a_follow_up_whose_call_fails_is_noted_and_the_run_goes_on(stocked, tmp_path):
    def breaks_on_follow_up(request):
        if "The research manager asks" in request.messages[1].content:
            raise LLMError("HTTP 400: bad request")
        return news_answering(request)

    model = Agents(
        research_manager=manager_asking(("news", "Why?")), news_analyst=breaks_on_follow_up
    )
    state = go(pipeline(stocked, tmp_path, model, config=TOOLED))
    (followup,) = state["followups"]
    assert not followup.answered and "HTTP 400" in followup.answer
    assert state["meta"].status == "ok" and state["decision"] is not None


def test_a_follow_up_with_nothing_to_go_on_says_so(tmp_path):
    with NewsStore(tmp_path / "n.db") as news, PriceStore(tmp_path / "p.db") as prices:
        ctx = ToolContext("NVDA", CLOSE, news, prices, FundamentalsStore(tmp_path / "s"), TOOLED)
        meta = Meta(
            run_id="x", ticker="NVDA", decision_time=CLOSE, mode="backtest",
            config_version="t1", config_digest="abc", repeat=0,
        )  # fmt: skip
        state = {
            "meta": meta,
            "reports": {"news": Report(text="No news was available.", no_data=True)},
            "pending": [Question(analyst="news", question="Anything new?")],
        }
        with CallCache(tmp_path / "c.db") as cache:
            out = answer_followups(state, CachedLLM(cache, Agents()), TOOLED, lambda: ctx)
    (followup,) = out["followups"]
    assert (followup.answered, followup.answer) == (False, "No data was available to answer.")
    assert out["pending"] == []
