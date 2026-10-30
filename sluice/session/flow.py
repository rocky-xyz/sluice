"""The welcome page and the questions of a terminal session (docs/specs/12, task 13.5).

What can be checked without a terminal (a ticker, a backtest day, a config's
one-line summary, the remembered answers) is a plain function here; `ask`
only puts the questions.
"""

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sluice import __version__
from sluice.agents import configure as cfg
from sluice.agents.config import CONFIG_ROOT, PipelineConfig, load_config
from sluice.market_calendar import close_of, last_closed_session
from sluice.universe import UNIVERSE, canonical

LAST = Path("data/local/session-last.json")
NEW = "__new__"  # the "New..." entry of the config list
TICKER = re.compile(r"[A-Z][A-Z0-9.\-]{0,9}")
BANNER = r"""
     _       _
 ___| |_   _(_) ___ ___
/ __| | | | | |/ __/ _ \
\__ \ | |_| | | (_|  __/
|___/_|\__,_|_|\___\___|
"""


@dataclass(frozen=True)
class Plan:
    ticker: str
    mode: Literal["adhoc", "backtest"]
    day: date | None  # the decision day of a backtest
    config: str


def welcome(console: Console) -> None:
    body = Text(BANNER.strip("\n"), style="bold cyan")
    body.append(f"\n\nsluice {__version__} · multi-agent equity research\n", style="bold")
    body.append("Gather data → analysts → bull/bear debate → research manager\n")
    body.append("A research aid, not trading advice.", style="dim")
    console.print(Panel(body, border_style="cyan", padding=(1, 4)))


def ticker_problem(text: str) -> str | None:
    if not TICKER.fullmatch(canonical(text)):
        return "a ticker such as NVDA or BRK.B"
    return None


def outside_universe(ticker: str) -> bool:
    return canonical(ticker) not in UNIVERSE


def date_problem(text: str, now: datetime) -> str | None:
    """Why `text` cannot be a backtest's decision day as of `now`, or None when it can."""
    try:
        day = date.fromisoformat(text.strip())
    except ValueError:
        return "a date as YYYY-MM-DD"
    if close_of(day) is None:
        return f"{day} is not a trading day"
    if day > last_closed_session(now):
        return f"{day} has not closed yet"
    return None


def config_line(config: PipelineConfig) -> str:
    """One line on what a config runs, for the list to pick from."""
    analysts = "+".join(config.selected_analysts)
    parts = [analysts]
    if "market" in config.selected_analysts:
        parts.append(f"{len(config.prices.selected_indicators)} indicators")
    rounds = config.debate_rounds
    parts.append("no debate" if rounds == 0 else f"{rounds} round{'s' if rounds > 1 else ''}")
    if config.tools is not None:
        parts.append("tools")
    if config.news.weighted:
        parts.append("weighted news")
    return " · ".join(parts)


def config_names(root: Path = CONFIG_ROOT) -> list[str]:
    """Every pipeline config, v1 first."""
    names = cfg.versions(root)
    return sorted(names, key=lambda name: (name != "v1", name))


def load_last(path: Path = LAST, root: Path = CONFIG_ROOT) -> dict[str, Any]:
    """The last session's answers, without any that are no longer valid."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    last: dict[str, Any] = {}
    if isinstance(raw.get("ticker"), str) and not ticker_problem(raw["ticker"]):
        last["ticker"] = raw["ticker"]
    if raw.get("mode") in ("adhoc", "backtest"):
        last["mode"] = raw["mode"]
    if isinstance(raw.get("day"), str):
        try:
            last["day"] = date.fromisoformat(raw["day"]).isoformat()
        except ValueError:
            pass
    if raw.get("config") in cfg.versions(root):
        last["config"] = raw["config"]
    return last


def save_last(plan: Plan, path: Path = LAST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(plan) | {"day": plan.day.isoformat() if plan.day else None}
    path.write_text(json.dumps(data, indent=2) + "\n")


def summary(plan: Plan, config: PipelineConfig) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Ticker", plan.ticker)
    when = "now (ad hoc)" if plan.mode == "adhoc" else f"backtest, the close of {plan.day}"
    table.add_row("Decides", when)
    table.add_row("Config", f"{config.version}-{config.digest}")
    table.add_row("", config_line(config))
    return table


def new_config(console: Console, root: Path = CONFIG_ROOT) -> str | None:
    """The new-config questions (once `sluice configure`); the name written, or None."""
    picked = cfg.interview(root)
    if picked is None:
        return None
    base, choices, name = picked
    source = (root / "pipeline" / f"{base.version}.toml").read_text()
    try:
        text = cfg.build_config_text(source, base, choices, name, cfg.now_utc())
    except ValueError as exc:
        console.print(f"Not written: {exc}", markup=False)
        return None
    console.print(f"New config {name}, from {base.version}-{base.digest}:", markup=False)
    for line in cfg.changes(base, choices) or ["no changes"]:
        console.print(f"  {line}", markup=False)
    if not cfg.confirm("Write it?"):
        return None
    path = cfg.write_config(name, text, root)
    cfg.save_last(base.version, choices)
    console.print(f"Wrote {path}.", markup=False)
    return name


def ask(console: Console, now: datetime, root: Path = CONFIG_ROOT) -> Plan | None:
    """The questions, until the user confirms a plan; None when they cancel."""
    import questionary  # only the interactive session needs it

    def asked(question: questionary.Question) -> Any:
        answer = question.ask()  # None on Ctrl-C or Esc
        if answer is None:
            raise KeyboardInterrupt
        return answer

    last = load_last(root=root)
    try:
        while True:
            ticker = canonical(asked(questionary.text(
                "Ticker:", default=last.get("ticker", "NVDA"),
                validate=lambda text: ticker_problem(text) or True,
            )))  # fmt: skip
            if outside_universe(ticker):
                console.print(f"{ticker} is outside the study's universe: the rating will say so.")
            mode = asked(questionary.select(
                "Decide when?",
                choices=[
                    questionary.Choice("Now (ad hoc)", value="adhoc"),
                    questionary.Choice("At a past close (backtest)", value="backtest"),
                ],
                default=last.get("mode", "backtest"),
            ))  # fmt: skip
            day = None
            if mode == "backtest":
                day = date.fromisoformat(asked(questionary.text(
                    "Decision day (YYYY-MM-DD):",
                    default=last.get("day", last_closed_session(now).isoformat()),
                    validate=lambda text: date_problem(text, now) or True,
                )).strip())  # fmt: skip
            while True:
                names = config_names(root)
                listed = [
                    questionary.Choice(f"{n:<14} {config_line(load_config(n, root))}", value=n)
                    for n in names
                ]
                listed.append(questionary.Choice("New...  pick analysts, indicators, depth", NEW))
                name = asked(questionary.select(
                    "Which config?", choices=listed, default=last.get("config", "v1"),
                ))  # fmt: skip
                if name == NEW:
                    name = new_config(console, root)
                    if name is None:
                        continue
                config = load_config(name, root)
                problems = cfg.weighting_problems(config) if config.news.weighted else []
                for problem in problems:
                    console.print(f"Cannot run {name}: {problem}", markup=False)
                if not problems:
                    break
            plan = Plan(ticker, mode, day, name)
            console.print(Panel(summary(plan, config), title="About to run", border_style="cyan"))
            if asked(questionary.confirm("Run it?", default=True)):
                return plan
            last = {"ticker": ticker, "mode": mode, "config": name} | (
                {"day": day.isoformat()} if day else {}
            )
    except KeyboardInterrupt:
        return None
