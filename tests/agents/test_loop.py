import json
from datetime import timedelta
from typing import ClassVar

import pytest

from sluice.agents.config import ToolsConfig
from sluice.agents.graph import run
from sluice.agents.llm import LLMRequest, LLMResponse, ToolCall
from sluice.agents.loop import LIMIT_REACHED, NO_DATA, tool_analyst
from sluice.agents.state import Meta, cited_refs, dump_state
from sluice.agents.tools import ToolContext
from sluice.fundamentals.store import FundamentalsStore
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore
from tests.agents.test_pipeline import CLOSE, CONFIG, ROLE_OF_PROMPT, article, pipeline, verdict

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
        from sluice.agents.cache import CallCache
        from sluice.agents.llm import CachedLLM

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
