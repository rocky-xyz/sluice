"""The live panel of a terminal session (docs/specs/12, task 13.4).

A `Board` listens to a run's events (sluice.agents.events) from any thread and
keeps what the panel shows: each step's status and time, the latest calls,
the latest finished output, and the tally behind the footer. `render` draws
it with rich; `line` gives one line per event for a terminal too small for
the layout.
"""

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sluice.agents import roles
from sluice.agents.config import PipelineConfig
from sluice.agents.events import Event, JevCalled, LLMCalled, NodeFinished, NodeStarted
from sluice.agents.graph import FOLLOWUP
from sluice.costs import Tally, spend

HIDDEN = {"analysts_done"}  # a join of the graph, not a step anyone waits on
LABELS = {
    "prefetch": "Gather data",
    roles.MARKET_ANALYST: "Market analyst",
    roles.NEWS_ANALYST: "News analyst",
    roles.FUNDAMENTALS_ANALYST: "Fundamentals analyst",
    roles.BULL: "Bull",
    roles.BEAR: "Bear",
    roles.RESEARCH_MANAGER: "Research manager",
    FOLLOWUP: "Follow-up answers",
}
SHORT = {  # for the narrow call list
    "prefetch": "Data",
    roles.MARKET_ANALYST: "Market",
    roles.NEWS_ANALYST: "News",
    roles.FUNDAMENTALS_ANALYST: "Fundmtl",
    roles.BULL: "Bull",
    roles.BEAR: "Bear",
    roles.RESEARCH_MANAGER: "Manager",
    FOLLOWUP: "Follow",
}
CALLS_KEPT = 50
MIN_SIZE = (80, 24)  # columns, rows: smaller and the session prints lines instead


@dataclass
class Step:
    name: str
    label: str
    status: str = "waiting"  # running, done, failed
    started: float | None = None
    finished: float | None = None
    stored: bool = False  # taken from the checkpoint, not run now

    def seconds(self, now: float) -> float | None:
        if self.started is None:
            return None
        return (self.finished or now) - self.started


@dataclass(frozen=True)
class CallLine:
    at: datetime
    node: str | None
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool
    tools: tuple[str, ...]


def planned_steps(config: PipelineConfig) -> list[Step]:
    """What a run of `config` will go through, in order; follow-ups are added when they happen."""
    steps = [Step("prefetch", LABELS["prefetch"])]
    for role in (roles.MARKET_ANALYST, roles.NEWS_ANALYST, roles.FUNDAMENTALS_ANALYST):
        if roles.ANALYSTS[role] in config.selected_analysts:
            steps.append(Step(role, LABELS[role]))
    for k in range(1, config.debate_rounds + 1):
        steps.append(Step(roles.BULL, f"{LABELS[roles.BULL]}, round {k}"))
        steps.append(Step(roles.BEAR, f"{LABELS[roles.BEAR]}, round {k}"))
    steps.append(Step(roles.RESEARCH_MANAGER, LABELS[roles.RESEARCH_MANAGER]))
    return steps


def _tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def _model(name: str) -> str:
    return {"deepseek-flash": "flash", "deepseek-v4-pro": "pro"}.get(name, name)


def _excerpt(text: str, lines: int) -> str:
    kept = [line for line in text.splitlines() if line.strip()][:lines]
    return "\n".join(kept)


class Board:
    """What the panel shows, updated by events from any thread."""

    def __init__(
        self,
        title: str,
        config: PipelineConfig,
        tally: Tally,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.title = title
        self.tally = tally
        self.steps = planned_steps(config)
        self.calls: deque[CallLine] = deque(maxlen=CALLS_KEPT)
        self.latest: tuple[str, str] | None = None  # (heading, text)
        self.note: str | None = None  # shown above the footer, e.g. "from the checkpoint"
        self._clock, self._wall = clock, wall
        self.started = clock()
        self.started_at = wall().astimezone()
        self._lock = threading.Lock()

    # --- events ---

    def __call__(self, event: Event) -> None:
        self.tally(event)
        with self._lock:
            if isinstance(event, NodeStarted) and event.name not in HIDDEN:
                step = self._step_for(event.name, "waiting")
                step.status, step.started = "running", self._clock()
            elif isinstance(event, NodeFinished) and event.name not in HIDDEN:
                step = self._step_for(event.name, "running")
                step.finished = self._clock()
                failed = event.failed or self._run_failed(event.update)
                step.status = "failed" if failed else "done"
                self._show(event.name, step.label, event.update)
            elif isinstance(event, LLMCalled):
                self.calls.append(
                    CallLine(
                        self._wall(),
                        event.node,
                        event.model,
                        event.input_tokens,
                        event.output_tokens,
                        event.cached,
                        event.tools,
                    )
                )

    def _step_for(self, name: str, status: str) -> Step:
        """The first step of `name` in `status`; a new one at the end when there is none."""
        for step in self.steps:
            if step.name == name and step.status == status:
                return step
        # A follow-up round, or the research manager again after one.
        again = sum(s.name == name for s in self.steps)
        label = LABELS.get(name, name) + (f" (again, {again + 1})" if again else "")
        step = Step(name, label, status=status)
        self.steps.append(step)
        return step

    @staticmethod
    def _run_failed(update: dict) -> bool:
        meta = update.get("meta")
        return meta is not None and getattr(meta, "status", "ok") == "failed"

    def _show(self, name: str, label: str, update: dict) -> None:
        if decision := update.get("decision"):
            head = f"{decision.rating}, confidence {decision.confidence:.2f}"
            self.latest = (label, f"{head}\n{decision.rationale}")
        elif turns := update.get("debate"):
            self.latest = (label, turns[-1].text)
        elif reports := update.get("reports"):
            self.latest = (label, next(iter(reports.values())).text)
        elif name == "prefetch" and (inputs := update.get("inputs")) is not None:
            self.latest = (label, self._inputs_line(inputs))

    @staticmethod
    def _inputs_line(inputs) -> str:
        parts = [
            f"{len(inputs.prices)} price indicators",
            f"{len(inputs.news)} news items",
            f"{len(inputs.fundamentals)} fundamental figures",
        ]
        if inputs.gaps:
            parts.append(f"{len(inputs.gaps)} gaps")
        return ", ".join(parts)

    def mark_stored(self, state: dict) -> None:
        """Steps whose output is already in a checkpointed state, shown as done from it."""
        reports = state.get("reports") or {}
        turns = list(state.get("debate") or [])
        with self._lock:
            for step in self.steps:
                if step.name == "prefetch":
                    done = state.get("inputs") is not None
                elif step.name in roles.ANALYSTS:  # reports are kept by channel
                    done = roles.ANALYSTS[step.name] in reports
                elif step.name in (roles.BULL, roles.BEAR):
                    done = any(t.side == step.name for t in turns)
                    if done:
                        turns.remove(next(t for t in turns if t.side == step.name))
                elif step.name == roles.RESEARCH_MANAGER:
                    done = state.get("decision") is not None
                else:
                    done = False
                if done and step.status == "waiting":
                    step.status, step.stored = "done", True

    # --- drawing ---

    def render(self) -> Layout:
        with self._lock:
            now = self._clock()
            layout = Layout()
            layout.split_column(
                Layout(Panel(Text(self.title, style="bold"), border_style="cyan"), size=3),
                Layout(name="upper", ratio=3),
                Layout(self._latest_panel(), ratio=2),
                Layout(Panel(self._footer(now), border_style="grey50"), size=4 + bool(self.note)),
            )
            layout["upper"].split_row(
                Layout(self._progress_panel(now), ratio=1),
                Layout(self._calls_panel(), ratio=1),
            )
            return layout

    def _progress_panel(self, now: float) -> Panel:
        table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        table.add_column("step", no_wrap=True, overflow="ellipsis", ratio=1)
        table.add_column("status", no_wrap=True, min_width=7)
        table.add_column("time", justify="right", no_wrap=True, min_width=4)
        styles = {"waiting": "dim", "running": "yellow", "done": "green", "failed": "bold red"}
        for step in self.steps:
            status = "stored" if step.stored else step.status
            seconds = step.seconds(now)
            table.add_row(
                Text(step.label, style="dim" if step.status == "waiting" else ""),
                Text(status, style=styles[step.status]),
                "" if seconds is None else f"{seconds:.0f}s",
            )
        return Panel(table, title="Progress", border_style="cyan")

    def _calls_panel(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
        table.add_column("at", no_wrap=True, min_width=8)
        table.add_column("who", no_wrap=True, overflow="ellipsis")
        table.add_column("model", no_wrap=True, min_width=5)
        table.add_column("tokens", no_wrap=True, min_width=9)
        table.add_column("note", no_wrap=True, overflow="ellipsis")
        for call in list(self.calls)[-12:]:
            note = "cached" if call.cached else ""
            if call.tools:
                note = (note + " " if note else "") + "-> " + ", ".join(call.tools)
            who = SHORT.get(call.node or "", call.node or "-")
            table.add_row(
                f"{call.at:%H:%M:%S}",
                who,
                _model(call.model),
                f"{_tokens(call.input_tokens)}→{_tokens(call.output_tokens)}",
                Text(note, style="dim"),
            )
        return Panel(
            table, title="LLM calls (tokens in → out)", border_style="cyan", padding=(0, 0)
        )

    def _latest_panel(self) -> Panel:
        if self.latest is None:
            return Panel(Text("Nothing finished yet.", style="dim"), title="Latest")
        heading, text = self.latest
        return Panel(Text(_excerpt(text, 8)), title=f"Latest: {heading}", border_style="blue")

    def _footer(self, now: float) -> Group:
        llm, jev = self.tally.snapshot()
        new = sum(u.new_calls for u in llm.values())
        cached = sum(u.cached_calls for u in llm.values())
        tokens_in = sum(u.all_in for u in llm.values())
        tokens_out = sum(u.all_out for u in llm.values())
        cost = spend(self.tally, self.started_at, self._wall().astimezone()).describe()
        usage = [
            f"LLM calls {new + cached} ({new} new, {cached} cached)",
            f"tokens {_tokens(tokens_in)} in / {_tokens(tokens_out)} out",
            f"Jev {jev.new_calls} new",
        ]
        lines = [
            Text(" · ".join(usage)),
            Text(f"Spent {cost} · elapsed {_clock(now - self.started)}"),
        ]
        if self.note:
            lines.insert(0, Text(self.note, style="yellow"))
        return Group(*lines)

    # --- a terminal too small for the layout ---

    def line(self, event: Event) -> str | None:
        """One line for an event, or None for one not worth a line."""
        if isinstance(event, NodeStarted) and event.name not in HIDDEN:
            return f"{self._wall():%H:%M:%S} {LABELS.get(event.name, event.name)} started"
        if isinstance(event, NodeFinished) and event.name not in HIDDEN:
            outcome = "failed" if event.failed or self._run_failed(event.update) else "done"
            return f"{self._wall():%H:%M:%S} {LABELS.get(event.name, event.name)} {outcome}"
        if isinstance(event, LLMCalled):
            cached = ", cached" if event.cached else ""
            return (
                f"{self._wall():%H:%M:%S}   {_model(event.model)}: {event.input_tokens:,} in, "
                f"{event.output_tokens:,} out{cached}"
            )
        if isinstance(event, JevCalled) and not event.cached:
            return f"{self._wall():%H:%M:%S}   Jev scored one item"
        return None
