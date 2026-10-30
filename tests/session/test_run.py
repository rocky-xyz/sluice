from rich.console import Console

from sluice.costs import Tally
from sluice.session import run as session
from sluice.session.panel import Board
from tests.agents.test_pipeline import CLOSE, CONFIG, FakeModel, pipeline, stocked, store  # noqa: F401


def board():
    return Board("sluice · NVDA", CONFIG, Tally())


def test_the_panel_needs_room():
    assert session.fits(Console(width=80, height=24))
    assert not session.fits(Console(width=79, height=40))
    assert not session.fits(Console(width=120, height=23))


def test_a_small_terminal_gets_a_line_per_event(stocked, tmp_path):  # noqa: F811
    b, console = board(), Console(record=True, width=100)
    on_event = session.listener(b, console, live=False)
    graph = pipeline(stocked, tmp_path, FakeModel(), on_event=on_event)
    state = session.execute(graph, CONFIG, "NVDA", CLOSE, "backtest", b, console, live=False)
    assert state["meta"].status == "ok"
    text = console.export_text()
    assert "Gather data started" in text and "Research manager done" in text
    assert all(step.status == "done" for step in b.steps)
    assert b.tally.snapshot()[0]["fast-model"].new_calls == 3


def test_a_finished_run_is_shown_from_its_checkpoint(stocked, tmp_path):  # noqa: F811
    first = board()
    graph = pipeline(stocked, tmp_path, FakeModel(), on_event=first)
    session.execute(graph, CONFIG, "NVDA", CLOSE, "backtest", first, Console(), live=False)

    again, console = board(), Console(record=True, width=100)
    graph = pipeline(stocked, tmp_path, FakeModel(), on_event=again)
    session.execute(graph, CONFIG, "NVDA", CLOSE, "backtest", again, console, live=False)
    assert "Finished before" in console.export_text()
    assert all(step.stored for step in again.steps)
    assert again.tally.snapshot() == ({}, again.tally.snapshot()[1])  # nothing asked


def test_the_live_panel_is_the_board_itself():
    b = board()
    assert session.listener(b, Console(), live=True) is b


def test_finish_prints_the_report_and_saves_the_state(stocked, tmp_path):  # noqa: F811
    b, console = board(), Console(record=True, width=100)
    graph = pipeline(stocked, tmp_path, FakeModel(), on_event=b)
    state = session.execute(graph, CONFIG, "NVDA", CLOSE, "backtest", b, Console(), live=False)
    assert session.finish(state, b, console, tmp_path / "sessions")
    text = console.export_text()
    assert "IV. Research manager" in text and "Spent about $0.00" in text
    saved = tmp_path / "sessions" / f"{state['meta'].run_id}.json"
    assert saved.exists() and str(saved) in text.replace("\n", "")
