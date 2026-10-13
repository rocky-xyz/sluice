import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sluice.agents import roles
from sluice.agents.config import PipelineConfig
from sluice.agents.llm import CachedLLM
from sluice.agents.prefetch import prefetch
from sluice.agents.state import STATE_MODELS, Meta, PipelineState, RunMode, make_run_id
from sluice.news.store import NewsStore
from sluice.universe import canonical

DEFAULT_CHECKPOINTS = Path("data/local/checkpoints.db")


@dataclass(frozen=True)
class Deps:
    """What the nodes need besides the state."""

    store: NewsStore
    llm: CachedLLM
    config: PipelineConfig


def build_graph(
    deps: Deps, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """prefetch -> news analyst -> research manager; a failed analyst ends the run.

    The research manager validates its own answer, because a rejected answer has
    to be asked again and the retry lives with the call.
    """

    def prefetch_node(state: PipelineState) -> dict:
        meta = state["meta"]
        inputs = prefetch(deps.store, meta.ticker, meta.decision_time, deps.config.news)
        return {"inputs": inputs}

    def news_analyst_node(state: PipelineState) -> dict:
        return roles.news_analyst(state, deps.llm, deps.config)

    def research_manager_node(state: PipelineState) -> dict:
        return roles.research_manager(state, deps.llm, deps.config)

    def after_analysts(state: PipelineState) -> str:
        return END if state["meta"].status == "failed" else roles.RESEARCH_MANAGER

    graph = StateGraph(PipelineState)
    graph.add_node("prefetch", prefetch_node)
    graph.add_node(roles.NEWS_ANALYST, news_analyst_node)
    graph.add_node(roles.RESEARCH_MANAGER, research_manager_node)
    graph.add_edge(START, "prefetch")
    graph.add_edge("prefetch", roles.NEWS_ANALYST)
    graph.add_conditional_edges(roles.NEWS_ANALYST, after_analysts, [roles.RESEARCH_MANAGER, END])
    graph.add_edge(roles.RESEARCH_MANAGER, END)
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
