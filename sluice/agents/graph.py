import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sluice.agents import loop, roles
from sluice.agents.config import PipelineConfig
from sluice.agents.llm import CachedLLM
from sluice.agents.prefetch import Weighting, prefetch
from sluice.agents.state import (
    STATE_MODELS,
    Gap,
    Inputs,
    Meta,
    PipelineState,
    RunMode,
    make_run_id,
)
from sluice.agents.tools import ToolContext
from sluice.fundamentals.store import FundamentalsStore
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore
from sluice.universe import canonical

DEFAULT_CHECKPOINTS = Path("data/local/checkpoints.db")
FOLLOWUP = "followup"  # the analysts answering the research manager's questions (task 9.5)


@dataclass(frozen=True)
class Deps:
    """What the nodes need besides the state.

    The two callables are only used by an ad-hoc run, before it pre-fetches:
    they bring in what the stores lack for a ticker nobody collected, and
    return the reason when they could not. `weighting` is only used, and then
    needed, when the config weights news.
    """

    store: NewsStore
    prices: PriceStore
    statements: FundamentalsStore
    llm: CachedLLM
    config: PipelineConfig
    collect_news: Callable[[str], str | None] | None = None
    fetch_statements: Callable[[str], str | None] | None = None
    weighting: Weighting | None = None


def build_graph(
    deps: Deps, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """prefetch -> three analysts side by side -> bull/bear debate -> research manager.

    With a [tools] section in the config, prefetch only notes what an ad-hoc run
    could not bring in, each analyst fetches its own data in a tool loop, and the
    manager may send questions back once before deciding.

    The analysts do not depend on each other. `analysts_done` waits for all
    three, so what follows runs once and only when none of them failed. The
    debate is a loop: bull, bear, and back to bull until the configured number
    of rounds is done. Any role that fails ends the run. The manager validates
    its own answer, because a rejected answer has to be asked again and the
    retry lives with the call.

    Backtests and ad-hoc runs share this graph. An ad-hoc run only does more
    before pre-fetching: it brings its ticker's news and statements up to date.
    """
    rounds = deps.config.debate_rounds

    def prefetch_node(state: PipelineState) -> dict:
        meta = state["meta"]
        failures = []
        if meta.mode == "adhoc":
            if deps.collect_news and (error := deps.collect_news(meta.ticker)):
                failures.append(Gap(kind="news", reason=f"live collection failed: {error}"))
            if (
                deps.fetch_statements
                and deps.statements.quarters(meta.ticker) is None
                and (error := deps.fetch_statements(meta.ticker))
            ):
                failures.append(Gap(kind="fundamentals", reason=f"fetch failed: {error}"))
        if deps.config.tools is not None:  # the analysts fetch for themselves (task 9.4)
            return {"inputs": Inputs(gaps=failures)}
        inputs = prefetch(
            deps.store,
            deps.prices,
            deps.statements,
            meta.ticker,
            meta.decision_time,
            deps.config,
            deps.weighting,
        )
        return {"inputs": inputs.model_copy(update={"gaps": [*failures, *inputs.gaps]})}

    def with_tools(state: PipelineState, role: str) -> dict:
        return loop.tool_analyst(state, deps.llm, deps.config, role, context(state))

    def market_analyst_node(state: PipelineState) -> dict:
        if deps.config.tools is not None:
            return with_tools(state, roles.MARKET_ANALYST)
        return roles.market_analyst(state, deps.llm, deps.config)

    def news_analyst_node(state: PipelineState) -> dict:
        if deps.config.tools is not None:
            return with_tools(state, roles.NEWS_ANALYST)
        return roles.news_analyst(state, deps.llm, deps.config)

    def fundamentals_analyst_node(state: PipelineState) -> dict:
        if deps.config.tools is not None:
            return with_tools(state, roles.FUNDAMENTALS_ANALYST)
        return roles.fundamentals_analyst(state, deps.llm, deps.config)

    def analysts_done_node(state: PipelineState) -> dict:
        return {}

    def bull_node(state: PipelineState) -> dict:
        return roles.debater(state, deps.llm, deps.config, roles.BULL)

    def bear_node(state: PipelineState) -> dict:
        return roles.debater(state, deps.llm, deps.config, roles.BEAR)

    def research_manager_node(state: PipelineState) -> dict:
        return roles.research_manager(state, deps.llm, deps.config)

    def context(state: PipelineState) -> ToolContext:
        meta = state["meta"]
        return ToolContext(
            meta.ticker, meta.decision_time, deps.store, deps.prices, deps.statements, deps.config
        )

    def followup_node(state: PipelineState) -> dict:
        return loop.answer_followups(state, deps.llm, deps.config, lambda: context(state))

    def after_manager(state: PipelineState) -> str:
        if state["meta"].status == "failed" or not state.get("pending"):
            return END
        return FOLLOWUP

    def after_analysts(state: PipelineState) -> str:
        if state["meta"].status == "failed":
            return END
        # Nothing to argue about without rounds or without a single report with data.
        if rounds == 0 or not roles.has_data(state["reports"]):
            return roles.RESEARCH_MANAGER
        return roles.BULL

    def after_bull(state: PipelineState) -> str:
        return END if state["meta"].status == "failed" else roles.BEAR

    def after_bear(state: PipelineState) -> str:
        if state["meta"].status == "failed":
            return END
        finished = sum(turn.side == roles.BEAR for turn in state["debate"])
        return roles.RESEARCH_MANAGER if finished >= rounds else roles.BULL

    analysts = {
        roles.MARKET_ANALYST: market_analyst_node,
        roles.NEWS_ANALYST: news_analyst_node,
        roles.FUNDAMENTALS_ANALYST: fundamentals_analyst_node,
    }
    graph = StateGraph(PipelineState)
    graph.add_node("prefetch", prefetch_node)
    graph.add_node("analysts_done", analysts_done_node)
    graph.add_node(roles.BULL, bull_node)
    graph.add_node(roles.BEAR, bear_node)
    graph.add_node(roles.RESEARCH_MANAGER, research_manager_node)
    graph.add_edge(START, "prefetch")
    for name, node in analysts.items():
        graph.add_node(name, node)
        graph.add_edge("prefetch", name)
    graph.add_edge(list(analysts), "analysts_done")
    graph.add_conditional_edges(
        "analysts_done", after_analysts, [roles.BULL, roles.RESEARCH_MANAGER, END]
    )
    graph.add_conditional_edges(roles.BULL, after_bull, [roles.BEAR, END])
    graph.add_conditional_edges(roles.BEAR, after_bear, [roles.BULL, roles.RESEARCH_MANAGER, END])
    graph.add_node(FOLLOWUP, followup_node)
    graph.add_conditional_edges(roles.RESEARCH_MANAGER, after_manager, [FOLLOWUP, END])
    graph.add_edge(FOLLOWUP, roles.RESEARCH_MANAGER)
    return graph.compile(checkpointer=checkpointer)


def open_checkpointer(path: str | Path = DEFAULT_CHECKPOINTS) -> SqliteSaver:
    """SQLite checkpoints. The caller closes `.conn`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LangGraph runs nodes on worker threads.
    conn = sqlite3.connect(path, check_same_thread=False)
    allowed = [(model.__module__, model.__name__) for model in STATE_MODELS]
    return SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))


def run(
    graph: CompiledStateGraph,
    config: PipelineConfig,
    ticker: str,
    decision_time: datetime,
    repeat: int = 0,
    mode: RunMode = "backtest",
) -> PipelineState:
    """Run one ticker at one decision time, picking up whatever an earlier attempt left.

    A finished run is returned as stored. An interrupted run continues after its
    last completed node. A failed run starts over: answers it already received
    come from the LLM cache, so only what actually failed is asked again.
    """
    ticker = canonical(ticker)
    run_id = make_run_id(ticker, decision_time, config.version, config.digest, repeat)
    meta = Meta(
        run_id=run_id,
        ticker=ticker,
        decision_time=decision_time,
        mode=mode,
        config_version=config.version,
        config_digest=config.digest,
        repeat=repeat,
    )
    thread = {"configurable": {"thread_id": run_id}}
    if graph.checkpointer is None:
        return graph.invoke({"meta": meta}, thread)

    snapshot = graph.get_state(thread)
    if not snapshot.values:
        return graph.invoke({"meta": meta}, thread)
    if snapshot.next:
        return graph.invoke(None, thread)
    if snapshot.values["meta"].status == "ok":
        return snapshot.values
    # Channels with a merge rule would pile new values onto the failed run's.
    graph.checkpointer.delete_thread(run_id)
    return graph.invoke({"meta": meta}, thread)
