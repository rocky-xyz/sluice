import json

from rich.console import Console

from sluice.agents.state import Gap, Inputs, Report, Turn
from sluice.session.report import full_report, save
from tests.session.test_panel import decision, meta


def text_of(parts) -> str:
    console = Console(width=100, record=True)
    for part in parts:
        console.print(part)
    return console.export_text()


def state(**overrides):
    base = {
        "meta": meta(),
        "inputs": Inputs(gaps=[Gap(kind="fundamentals", reason="no statements")]),
        "reports": {"news": Report(text="**Key events**\n1. Demand is strong [N1]."),
                    "market": Report(text="Trend is up [P:ret_5d].")},
        "debate": [Turn(round=1, side="bull", text="It will rise."),
                   Turn(round=1, side="bear", text="It will fall.")],
        "decision": decision("overweight"),
        "usage": [],
    }
    return base | overrides


def test_the_report_has_its_four_sections_in_order():
    text = text_of(full_report(state()))
    order = ["I. Inputs", "Gap in fundamentals", "II. Analyst reports", "Market analyst",
             "News analyst", "Key events", "III. Debate", "Bull, round 1", "Bear, round 1",
             "IV. Research manager", "overweight", "confidence 0.60", "Demand is strong.",
             "N1", "Two-way risk."]  # fmt: skip
    at = 0
    for part in order:  # each one after the one before
        at = text.index(part, at)
    assert "**" not in text  # Markdown is rendered, not shown raw


def test_a_run_without_a_rating_says_why():
    text = text_of(full_report(state(decision=None, meta=meta("failed", "bear gave up"))))
    assert "No rating: bear gave up" in text


def test_a_run_without_debate_leaves_its_section_out():
    assert "III. Debate" not in text_of(full_report(state(debate=[])))


def test_the_whole_state_is_saved_under_its_run_id(tmp_path):
    path = save(state(), tmp_path)
    assert path == tmp_path / "NVDA_x.json"
    saved = json.loads(path.read_text())
    assert saved["decision"]["rating"] == "overweight"
    assert saved["meta"]["run_id"] == "NVDA_x"
