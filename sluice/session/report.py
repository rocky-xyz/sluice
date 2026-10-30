"""The full report printed after a terminal session's run, and its saved state (docs/specs/12).

Four sections, each a set of panels: what went in, the analysts' reports, the
debate, and the research manager's decision. The whole state is written as
JSON next to it, as `sluice analyze --out` used to.
"""

import json
from pathlib import Path

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sluice.agents import roles
from sluice.agents.state import PipelineState, dump_state
from sluice.session.panel import LABELS

SESSIONS = Path("data/local/sessions")
RATING_STYLES = {
    "buy": "bold green",
    "overweight": "green",
    "hold": "yellow",
    "underweight": "red",
    "sell": "bold red",
}


def save(state: PipelineState, root: Path = SESSIONS) -> Path:
    path = root / f"{state['meta'].run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump_state(state), ensure_ascii=False, indent=2) + "\n")
    return path


def _section(title: str, style: str) -> Panel:
    return Panel(Text(title, style="bold"), border_style=style)


def inputs_section(state: PipelineState) -> list[RenderableType]:
    meta, inputs = state["meta"], state.get("inputs")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Run", meta.run_id)
    if inputs is not None:
        as_of = f" (as of {inputs.prices_as_of})" if inputs.prices_as_of else ""
        table.add_row("Price indicators", f"{len(inputs.prices)}{as_of}")
        table.add_row("News items", str(len(inputs.news)))
        table.add_row("Fundamental figures", str(len(inputs.fundamentals)))
        for signal in inputs.signals:
            table.add_row(
                signal.ref,
                f"{signal.value:+.2e} over {signal.n_items} scored items "
                f"(weights as of {signal.weights_as_of:%Y-%m-%d})",
            )
        for gap in inputs.gaps:
            table.add_row(Text(f"Gap in {gap.kind}", style="yellow"), gap.reason)
    return [_section("I. Inputs", "cyan"), Panel(table, border_style="blue")]


def reports_section(state: PipelineState) -> list[RenderableType]:
    reports = state.get("reports") or {}
    if not reports:
        return []
    out: list[RenderableType] = [_section("II. Analyst reports", "cyan")]
    for role in (roles.MARKET_ANALYST, roles.NEWS_ANALYST, roles.FUNDAMENTALS_ANALYST):
        channel = roles.ANALYSTS[role]
        if (report := reports.get(channel)) is not None:
            style = "dim" if report.no_data else "blue"
            out.append(Panel(Markdown(report.text), title=LABELS[role], border_style=style))
    return out


def debate_section(state: PipelineState) -> list[RenderableType]:
    turns = state.get("debate") or []
    if not turns:
        return []
    out: list[RenderableType] = [_section("III. Debate", "magenta")]
    for turn in turns:
        title = f"{LABELS[turn.side]}, round {turn.round}"
        style = "green" if turn.side == roles.BULL else "red"
        out.append(Panel(Markdown(turn.text), title=title, border_style=style))
    return out


def decision_section(state: PipelineState) -> list[RenderableType]:
    decision, meta = state.get("decision"), state["meta"]
    out: list[RenderableType] = [_section("IV. Research manager", "green")]
    if decision is None:
        out.append(Panel(Text(f"No rating: {meta.failure}", style="bold red"), border_style="red"))
        return out
    head = Text.assemble(
        (decision.rating, RATING_STYLES.get(decision.rating, "bold")),
        f"   confidence {decision.confidence:.2f}",
    )
    evidence = Table(show_header=False, box=None, padding=(0, 1))
    for item in decision.evidence:
        mark = Text("+", style="green") if item.direction == "bullish" else Text("-", style="red")
        evidence.add_row(mark, item.claim, Text(" ".join(item.refs), style="dim"))
    parts: list[RenderableType] = [head, Text(""), evidence, Text(""), Markdown(decision.rationale)]
    parts += [Text(f"Note: {note}", style="yellow") for note in decision.notes]
    out.append(Panel(Group(*parts), border_style="green"))
    return out


def full_report(state: PipelineState) -> list[RenderableType]:
    return [
        *inputs_section(state),
        *reports_section(state),
        *debate_section(state),
        *decision_section(state),
    ]
