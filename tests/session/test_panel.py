import threading
from datetime import UTC, datetime, timedelta

import pytest
from rich.console import Console

from sluice.agents.events import JevCalled, LLMCalled, NodeFinished, NodeStarted
from sluice.agents.state import Decision, Evidence, Gap, Inputs, Meta, Report, Turn
from sluice.costs import Tally
from sluice.session.panel import Board, planned_steps
from tests.agents.test_pipeline import CONFIG

RUNNING = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)  # a Saturday: off-peak


def config(rounds=1, analysts=None):
    return CONFIG.model_copy(update={"debate_rounds": rounds, "analysts": analysts})


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def board(rounds=1, analysts=None):
    clock = Clock()
    made = Board(
        "sluice · NVDA",
        config(rounds, analysts),
        Tally(),
        clock=clock,
        wall=lambda: RUNNING + timedelta(seconds=clock.now),
    )
    return made, clock


def decision(rating="hold"):
    evidence = [Evidence(claim="Demand is strong.", direction="bullish", refs=["N1"])]
    return Decision(rating=rating, confidence=0.6, evidence=evidence, rationale="Two-way risk.")


def meta(status="ok", failure=None):
    return Meta(
        run_id="NVDA_x", ticker="NVDA", decision_time=RUNNING, mode="backtest",
        config_version="t1", config_digest="abc12345", repeat=0, status=status, failure=failure,
    )  # fmt: skip


def statuses(b):
    return [(s.label, s.status) for s in b.steps]


def test_the_plan_follows_the_config():
    labels = [s.label for s in planned_steps(config(rounds=2))]
    assert labels == [
        "Gather data", "Market analyst", "News analyst", "Fundamentals analyst",
        "Bull, round 1", "Bear, round 1", "Bull, round 2", "Bear, round 2", "Research manager",
    ]  # fmt: skip
    labels = [s.label for s in planned_steps(config(rounds=0, analysts=("news",)))]
    assert labels == ["Gather data", "News analyst", "Research manager"]


def test_steps_run_and_finish_with_their_time():
    b, clock = board()
    b(NodeStarted("prefetch"))
    clock.now = 2
    b(NodeFinished("prefetch", {"inputs": Inputs(gaps=[Gap(kind="news", reason="none")])}))
    b(NodeStarted("news_analyst"))
    clock.now = 30
    assert statuses(b)[:3] == [
        ("Gather data", "done"), ("Market analyst", "waiting"), ("News analyst", "running"),
    ]  # fmt: skip
    assert b.steps[0].seconds(clock.now) == 2 and b.steps[2].seconds(clock.now) == 28
    assert b.latest == ("Gather data", "0 price indicators, 0 news items, 0 fundamental figures, 1 gaps")


def test_debate_turns_fill_their_rounds_in_order_and_the_join_is_hidden():
    b, _ = board(rounds=2)
    b(NodeStarted("analysts_done"))
    b(NodeFinished("analysts_done", {}))
    for side in ("bull", "bear", "bull"):
        b(NodeStarted(side))
        b(NodeFinished(side, {"debate": [Turn(round=1, side=side, text=f"{side} says")]}))
    assert [s.status for s in b.steps if s.name in ("bull", "bear")] == [
        "done", "done", "done", "waiting",
    ]  # fmt: skip
    assert b.latest == ("Bull, round 2", "bull says")
    assert "analysts_done" not in {s.name for s in b.steps}


def test_follow_ups_and_a_second_verdict_are_added_when_they_happen():
    b, _ = board(rounds=0)
    for name in ("research_manager", "followup", "research_manager"):
        b(NodeStarted(name))
        b(NodeFinished(name, {}))
    assert [s.label for s in b.steps][-3:] == [
        "Research manager", "Follow-up answers", "Research manager (again, 2)",
    ]  # fmt: skip
    assert all(s.status == "done" for s in b.steps[-3:])


def test_a_failure_shows_whether_raised_or_written_to_the_state():
    b, _ = board()
    b(NodeStarted("market_analyst"))
    b(NodeFinished("market_analyst", failed=True))
    b(NodeStarted("research_manager"))
    b(NodeFinished("research_manager", {"meta": meta("failed", "bad answer")}))
    assert dict(statuses(b))["Market analyst"] == "failed"
    assert dict(statuses(b))["Research manager"] == "failed"


def test_the_decision_shows_rating_and_rationale():
    b, _ = board()
    b(NodeStarted("research_manager"))
    b(NodeFinished("research_manager", {"decision": decision("overweight")}))
    assert b.latest == ("Research manager", "overweight, confidence 0.60\nTwo-way risk.")


def test_calls_are_listed_and_counted():
    b, _ = board()
    b(LLMCalled("news_analyst", "deepseek-flash", 4100, 700, cached=False, tools=("get_news",)))
    b(LLMCalled("bull", "deepseek-flash", 2000, 300, cached=True))
    b(JevCalled(500, 0, cached=False))
    assert [(c.node, c.cached, c.tools) for c in b.calls] == [
        ("news_analyst", False, ("get_news",)), ("bull", True, ()),
    ]  # fmt: skip
    llm, jev = b.tally.snapshot()
    assert (llm["deepseek-flash"].new_calls, llm["deepseek-flash"].cached_calls) == (1, 1)
    assert jev.new_calls == 1


def test_a_stored_state_marks_what_is_already_done():
    b, _ = board(rounds=2)
    b.mark_stored({
        "inputs": Inputs(),
        "reports": {"market": Report(text="up"), "news": Report(text="quiet")},
        "debate": [Turn(round=1, side="bull", text="b"), Turn(round=1, side="bear", text="r")],
        "decision": None,
    })  # fmt: skip
    done = [s.label for s in b.steps if s.stored]
    assert done == ["Gather data", "Market analyst", "News analyst", "Bull, round 1", "Bear, round 1"]
    assert all(s.status == "done" for s in b.steps if s.stored)


@pytest.mark.parametrize("width, height", [(80, 24), (140, 40)])
def test_the_panel_draws_every_part(width, height):
    b, clock = board()
    b(NodeStarted("news_analyst"))
    b(LLMCalled("news_analyst", "deepseek-flash", 4100, 700, cached=False))
    b(NodeFinished("news_analyst", {"reports": {"news": Report(text="Demand is strong.")}}))
    clock.now = 75
    console = Console(width=width, height=height, record=True)
    console.print(b.render(), height=height)
    text = console.export_text()
    for part in ("sluice · NVDA", "Progress", "News analyst", "done", "4.1k→700",
                 "Latest: News analyst", "Demand is strong.", "1 new, 0 cached",
                 "Spent about $0.00 (off-peak)", "elapsed 1:15"):  # fmt: skip
        assert part in text, part


def test_a_note_is_shown_above_the_footer():
    b, _ = board()
    b.note = "From the checkpoint: nothing was run now."
    console = Console(width=100, height=30, record=True)
    console.print(b.render(), height=30)
    assert "From the checkpoint" in console.export_text()


def test_lines_for_a_small_terminal():
    b, _ = board()
    assert b.line(NodeStarted("news_analyst")) == "12:00:00 News analyst started"
    assert b.line(NodeFinished("bull", failed=True)) == "12:00:00 Bull failed"
    assert b.line(LLMCalled("bull", "deepseek-v4-pro", 1200, 30, cached=True)) == (
        "12:00:00   pro: 1,200 in, 30 out, cached"
    )
    assert b.line(NodeStarted("analysts_done")) is None
    assert b.line(JevCalled(1, 0, cached=True)) is None


def test_events_from_many_threads_are_all_kept():
    b, _ = board()
    workers = [
        threading.Thread(
            target=lambda: [
                b(LLMCalled("bull", "deepseek-flash", 1, 1, cached=False)) for _ in range(500)
            ]
        )
        for _ in range(4)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert b.tally.snapshot()[0]["deepseek-flash"].new_calls == 2000
