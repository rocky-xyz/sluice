"""One run of a terminal session: the live panel while it goes, the report after (docs/specs/12).

`listener` picks how events are shown: the panel, or one line each when the
terminal is too small for it. `execute` runs the graph, telling the panel
first what a checkpoint already holds. `finish` prints the report, saves the
state and says what was spent.
"""

from datetime import datetime
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.live import Live

from sluice.agents.config import PipelineConfig
from sluice.agents.events import Event, OnEvent
from sluice.agents.graph import run
from sluice.agents.state import PipelineState, RunMode, make_run_id
from sluice.costs import spend
from sluice.session.panel import MIN_SIZE, Board
from sluice.session.report import SESSIONS, full_report, save
from sluice.universe import canonical


def fits(console: Console) -> bool:
    width, height = console.size
    return width >= MIN_SIZE[0] and height >= MIN_SIZE[1]


def listener(board: Board, console: Console, live: bool) -> OnEvent:
    if live:
        return board

    def print_lines(event: Event) -> None:
        board(event)
        if (line := board.line(event)) is not None:
            console.print(line, markup=False)

    return print_lines


def execute(
    graph: CompiledStateGraph,
    config: PipelineConfig,
    ticker: str,
    decision_time: datetime,
    mode: RunMode,
    board: Board,
    console: Console,
    live: bool,
) -> PipelineState:
    """Run once; a run the checkpoint holds shows its finished steps as stored."""
    if graph.checkpointer is not None:
        run_id = make_run_id(canonical(ticker), decision_time, config.version, config.digest, 0)
        snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
        if snapshot.values and snapshot.values["meta"].status == "ok":  # a failed one restarts
            board.mark_stored(snapshot.values)
            board.note = (
                "Continuing from the checkpoint."
                if snapshot.next
                else "Finished before: shown from the checkpoint, nothing run now."
            )
            if not live:
                console.print(board.note, markup=False)
    if not live:
        return run(graph, config, ticker, decision_time, mode=mode)
    with Live(get_renderable=board.render, console=console, refresh_per_second=4, screen=False):
        return run(graph, config, ticker, decision_time, mode=mode)


def finish(state: PipelineState, board: Board, console: Console, root: Path = SESSIONS) -> bool:
    """Print the report and save the state; True when the run gave a rating."""
    for part in full_report(state):
        console.print(part)
    path = save(state, root)
    ended = datetime.now().astimezone()
    console.print(f"Spent {spend(board.tally, board.started_at, ended).describe()}.")
    console.print(f"Full state saved to {path}", markup=False)
    return state.get("decision") is not None
