"""`sluice configure`: a new pipeline config from an existing one and a few choices.

docs/specs/10-configurable-components.md, task 11.5. The choices become a file under
`configs/pipeline/`, so a run of it has its own digest and run ids like any version.
Everything here except `interview` works without a terminal; `interview` only asks.

The new file is the base file with as little changed as possible: its comments and
prompt layout stay, the keys that change are rewritten in place or added, and the
result is read back and compared with what was asked for before anything is written.
"""

import json
import os
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from sluice.agents import indicators
from sluice.agents.config import ANALYST_KEYS, CONFIG_ROOT, PipelineConfig, load_config
from sluice.news import weight_tables

LAST_CHOICES = Path("data/local/configure-last.json")
# One checkbox each, in the order of indicators.ALL.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Returns over 5, 20 and 60 days", ("ret_5d", "ret_20d", "ret_60d")),
    ("Returns against the benchmark over 5, 20 and 60 days",
     ("excess_5d", "excess_20d", "excess_60d")),
    ("20-day volatility", ("vol_20d",)),
    ("Gap to the 50- and 200-day averages", ("vs_sma50", "vs_sma200")),
    ("RSI 14", ("rsi_14",)),
    ("Volume against its 20-day average", ("volume_ratio_20d",)),
    ("MACD histogram", ("macd_hist_pct",)),
    ("Bollinger %B", ("bb_pctb_20",)),
    ("Bollinger band width", ("bb_width_20",)),
    ("ATR 14", ("atr_14_pct",)),
    ("Gap to the 10-day EMA", ("vs_ema10",)),
    ("Position in the 250-day range", ("range_250d",)),
    ("KDJ (K and D)", ("kdj_k", "kdj_d")),
)  # fmt: skip
# What a new config starts with ticked: the original eleven and five of the new ones.
SUGGESTED = (
    *indicators.NAMES, "macd_hist_pct", "bb_pctb_20", "bb_width_20", "atr_14_pct", "range_250d",
)  # fmt: skip
DEPTHS = {
    0: "No debate (0 rounds)",
    1: "Shallow (1 round)",
    3: "Medium (3 rounds)",
    5: "Deep (5 rounds)",
}
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
EVOLVED = re.compile(r"v\d+")  # the evolution loop numbers its own versions so

# The market analyst's prompt when the indicators differ from the base config's: it
# names none of them, so that one left out cannot be asked about. As written in TOML.
GENERIC_MARKET_PROMPT_TOML = """
You are the market analyst of an equity research team. You receive price and volume indicators \\
for one stock, computed from daily data up to the decision time. Each line starts with its \\
reference id in square brackets, for example [P:ret_20d], and says what the indicator measures. \\
All of them are ratios or positions; you are not given price levels.

Write a report for colleagues who will not see the indicators themselves. Cover, as far as the \\
indicators you receive allow:
1. The trend: its direction and strength, and whether shorter and longer measures agree.
2. Momentum, and how the stock has done against the market.
3. Risk: volatility, and whether the move looks stretched.
4. What this suggests about the stock over the next two weeks, and how strongly.

Rules:
- Use only the indicators given, and do not discuss any you were not given. Do not add prices, \\
levels or events from memory.
- After every claim, cite the reference id of each indicator it rests on, in square brackets.
- Quote a number only as it is written in the indicators.
- Nothing after the decision time exists for you.
- Do not recommend a position size.
- At most 250 words.
"""
GENERIC_MARKET_PROMPT = tomllib.loads(f'p = """{GENERIC_MARKET_PROMPT_TOML}"""')["p"]

_SECTION = re.compile(r"\[(?P<name>\w+)\]")
_ASSIGNMENT = re.compile(r"(?P<key>\w+) = (?P<value>.*?)(?P<comment>\s+#.*)?")
_COMMENTS = {
    "analysts": "the analysts that run (docs/specs/10)",
    "indicators": "shown in the registry's order (docs/specs/10)",
    "weighted": "filter by Jev relevance, label sources and types, add S:news_signal",
}


class Choices(BaseModel):
    """What the user picked. `indicators` is empty when the market analyst is left out."""

    model_config = ConfigDict(frozen=True)

    analysts: tuple[str, ...]
    indicators: tuple[str, ...]
    debate_rounds: int
    weighted: bool


def versions(root: Path = CONFIG_ROOT) -> list[str]:
    return sorted(path.stem for path in (root / "pipeline").glob("*.toml"))


def name_problem(name: str, root: Path = CONFIG_ROOT) -> str | None:
    """Why `name` cannot be a new config's name, or None when it can."""
    if not NAME.fullmatch(name):
        return "use lower-case letters, digits and single hyphens, e.g. market-news-deep"
    if EVOLVED.fullmatch(name):
        return "names like v4 are kept for the versions the evolution loop writes"
    if (root / "pipeline" / f"{name}.toml").exists():
        return f"configs/pipeline/{name}.toml exists; a version that may have run is not changed"
    return None


def changes(base: PipelineConfig, choices: Choices) -> list[str]:
    """What `choices` change in `base`, one line each, for the summary and the file's header."""
    lines = []
    if choices.analysts != base.selected_analysts:
        lines.append(f"analysts: {', '.join(choices.analysts)}")
    if "market" in choices.analysts:
        before, after = set(base.prices.selected_indicators), set(choices.indicators)
        if added := [name for name in indicators.ALL if name in after - before]:
            lines.append(f"indicators added: {', '.join(added)}")
        if removed := [name for name in indicators.ALL if name in before - after]:
            lines.append(f"indicators removed: {', '.join(removed)}")
    if choices.debate_rounds != base.debate_rounds:
        lines.append(f"debate rounds: {base.debate_rounds} -> {choices.debate_rounds}")
    if choices.weighted != base.news.weighted:
        lines.append(f"weighted news: {'on' if choices.weighted else 'off'}")
    return lines


def _new_prompt(base: PipelineConfig, choices: Choices) -> bool:
    return (
        "market" in choices.analysts
        and set(choices.indicators) != set(base.prices.selected_indicators)
    )


def expected(base: PipelineConfig, choices: Choices) -> dict[str, Any]:
    """Every parameter the new config must have: the base's, with the choices applied."""
    tree = base.model_dump(exclude={"version", "digest"})
    chosen = tuple(key for key in ANALYST_KEYS if key in choices.analysts)
    tree["analysts"] = None if chosen == ANALYST_KEYS else chosen  # all three: leave it out
    if "market" not in choices.analysts:
        tree["prices"]["indicators"] = None
    elif _new_prompt(base, choices):
        tree["prices"]["indicators"] = tuple(n for n in indicators.ALL if n in choices.indicators)
        tree["prompts"]["market_analyst"] = GENERIC_MARKET_PROMPT
    tree["debate_rounds"] = choices.debate_rounds
    tree["news"]["weighted"] = choices.weighted and "news" in choices.analysts
    return tree


def _array(names: tuple[str, ...]) -> list[str]:
    """A TOML array of strings, wrapped: the opening line, the items, the closing line."""
    items, line = [], "   "
    for name in names:
        word = f' "{name}",'
        if len(line) + len(word) > 96:
            items.append(line)
            line = "   "
        line += word
    return ["[", *items, line, "]"]


def _extent(lines: list[str], at: int) -> int:
    """The index after the assignment starting at `at`; an array may span lines."""
    value = _ASSIGNMENT.fullmatch(lines[at])["value"]
    if not value.startswith("[") or value.endswith("]"):
        return at + 1
    return next(i for i in range(at + 1, len(lines)) if lines[i].strip() == "]") + 1


def _set(lines: list[str], section: str, key: str, value: list[str] | None) -> list[str]:
    """`lines` with `key` in `section` ("" for the top level) set to `value`, or removed.

    `value` is the value's lines (one, or an array's several). An existing assignment
    keeps its comment; a new one goes after the section's last key.
    """
    current, found, last_key, end = "", None, None, len(lines)
    i = 0
    while i < len(lines):
        line = lines[i]
        if match := _SECTION.fullmatch(line.split("#")[0].strip()):
            if current == section:
                end = i
                break
            current = match["name"]
            if current == section:
                last_key = i
            i += 1
            continue
        if current == section and (match := _ASSIGNMENT.fullmatch(line)):
            after = _extent(lines, i)
            last_key = after - 1
            if match["key"] == key:
                found = (i, after)
            i = after
            continue
        i += 1
    if found is None and value is None:
        return lines
    if found is not None:
        start, after = found
        comment = _ASSIGNMENT.fullmatch(lines[start])["comment"] or ""
    else:
        start = after = (last_key + 1) if last_key is not None else end
        comment = f"  # {_COMMENTS[key]}"
    if value is None:
        return lines[:start] + lines[after:]
    written = [f"{key} = {value[0]}{comment}", *value[1:]]
    return [*lines[:start], *written, *lines[after:]]


def _replace_prompt(lines: list[str], role: str, toml_body: str) -> list[str]:
    opening = next(
        i for i, line in enumerate(lines) if re.fullmatch(rf"{role} = (\"\"\"|''')\s*", line)
    )
    quote = lines[opening].split("=", 1)[1].strip()
    closing = next(i for i in range(opening + 1, len(lines)) if lines[i].strip() == quote)
    body = toml_body.strip("\n").splitlines()
    return [*lines[:opening], f'{role} = """', *body, '"""', *lines[closing + 1 :]]


def build_config_text(
    source: str, base: PipelineConfig, choices: Choices, name: str, now: datetime
) -> str:
    """The new config's file: `source` (the base's file) with `choices` applied.

    Raises ValueError when the choices do not fit together (as loading would), or when
    the text read back is not what was asked for: then nothing should be written.
    """
    lines = source.splitlines()
    start = 0
    while start < len(lines) and lines[start].startswith("#"):
        start += 1
    header = [
        f"# Pipeline config {name}: written by `sluice configure` on {now:%Y-%m-%d %H:%M} UTC,",
        f"# from {base.version}-{base.digest}. Changes from it:",
        *(f"#   {line}" for line in changes(base, choices) or ["none"]),
        "# A change to a version that has been run is a new file.",
    ]
    lines = lines[start:]
    split = lines.index("[prompts]")  # keys are only edited above it, prompts below it
    head, prompts = lines[:split], lines[split:]

    tree = expected(base, choices)
    analysts = tree["analysts"]
    head = _set(head, "", "analysts", [json.dumps(list(analysts))] if analysts else None)
    head = _set(head, "", "debate_rounds", [str(choices.debate_rounds)])
    picked = tree["prices"]["indicators"]
    if picked != base.prices.indicators:
        head = _set(head, "prices", "indicators", _array(picked) if picked else None)
    head = _set(head, "news", "weighted", ["true"] if tree["news"]["weighted"] else None)
    if _new_prompt(base, choices):
        prompts = _replace_prompt(prompts, "market_analyst", GENERIC_MARKET_PROMPT_TOML)

    text = "\n".join([*header, *head, *prompts]) + "\n"
    written = PipelineConfig(version=name, digest="", **tomllib.loads(text))
    if written.model_dump(exclude={"version", "digest"}) != tree:
        raise ValueError("the written config would not hold the choices; nothing written")
    return text


def write_config(name: str, text: str, root: Path = CONFIG_ROOT) -> Path:
    """Write the new file; fails rather than replace one that exists."""
    path = root / "pipeline" / f"{name}.toml"
    with path.open("x") as file:
        file.write(text)
    load_config(name, root)  # loads as any version does
    return path


def weighting_problems(base: PipelineConfig) -> list[str]:
    """What a weighted run of `base` would lack: shown as a warning, not a refusal."""
    problems = []
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        problems.append("TYPESAFE_API_KEY is not set: news without scores cannot be scored")
    scoring = base.news.scoring_config
    if not weight_tables.load_all(weight_tables.DEFAULT_ROOT, scoring):
        problems.append(f"no weight tables for scoring config {scoring} (sluice news fit-weights)")
    return problems


def load_last(path: Path = LAST_CHOICES, root: Path = CONFIG_ROOT) -> dict[str, Any]:
    """The last run's choices, keeping only values that are still on offer."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    last: dict[str, Any] = {}
    if raw.get("base") in versions(root):
        last["base"] = raw["base"]
    for key, offered in (("analysts", ANALYST_KEYS), ("indicators", indicators.ALL)):
        values = raw.get(key)
        if isinstance(values, list) and (kept := [value for value in offered if value in values]):
            last[key] = kept
    if type(raw.get("debate_rounds")) is int and raw["debate_rounds"] in DEPTHS:  # not a bool
        last["debate_rounds"] = raw["debate_rounds"]
    if isinstance(raw.get("weighted"), bool):
        last["weighted"] = raw["weighted"]
    return last


def save_last(base: str, choices: Choices, path: Path = LAST_CHOICES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"base": base, **choices.model_dump(mode="json")}, indent=2) + "\n")


def interview(root: Path = CONFIG_ROOT) -> tuple[PipelineConfig, Choices, str] | None:
    """Ask for the base config, the choices and a name. None when the user cancels."""
    import questionary  # only the interactive command needs it

    def asked(question: questionary.Question) -> Any:
        answer = question.ask()  # None on Ctrl-C or Esc
        if answer is None:
            raise KeyboardInterrupt
        return answer

    last = load_last(root=root)
    try:
        names = versions(root)
        base_name = asked(questionary.select(
            "Start from which config?", choices=names, default=last.get("base", "v1"),
        ))  # fmt: skip
        base = load_config(base_name, root)

        if base.tools is not None:
            print(f"{base_name} lets the analysts call tools: all three run, with its indicators.")
            analysts, picked = base.selected_analysts, base.prices.selected_indicators
        else:
            ticked = set(last.get("analysts", base.selected_analysts))
            analysts = tuple(asked(questionary.checkbox(
                "Which analysts?",
                choices=[questionary.Choice(key, checked=key in ticked) for key in ANALYST_KEYS],
                validate=lambda keys: bool(keys) or "Pick at least one analyst.",
            )))  # fmt: skip
            picked = ()
            if "market" in analysts:
                ticked = set(last.get("indicators", SUGGESTED))
                groups = asked(questionary.checkbox(
                    "Which indicators does the market analyst see?",
                    choices=[
                        questionary.Choice(label, value=names, checked=bool(ticked & set(names)))
                        for label, names in GROUPS
                    ],
                    validate=lambda ticks: bool(ticks) or "Pick at least one indicator.",
                ))  # fmt: skip
                chosen = {name for names in groups for name in names}
                picked = tuple(name for name in indicators.ALL if name in chosen)

        rounds = last.get("debate_rounds", base.debate_rounds)
        depth = asked(questionary.select(
            "How deep a debate?",
            choices=[questionary.Choice(label, value=n) for n, label in DEPTHS.items()],
            default=rounds if rounds in DEPTHS else 1,  # a value, not a label
        ))  # fmt: skip

        weighted = False
        if "news" in analysts:
            weighted = asked(questionary.confirm(
                "Weight the news (Jev relevance filter, source and type labels, news signal)?",
                default=last.get("weighted", base.news.weighted),
            ))  # fmt: skip
            for problem in weighting_problems(base) if weighted else []:
                print(f"Warning: {problem}")

        choices = Choices(
            analysts=analysts, indicators=picked, debate_rounds=depth, weighted=weighted
        )
        name = asked(questionary.text(
            "Name of the new config:",
            validate=lambda text: name_problem(text, root) or True,
        )).strip()  # fmt: skip
    except KeyboardInterrupt:
        return None
    return base, choices, name


def confirm(message: str) -> bool:
    import questionary

    return bool(questionary.confirm(message, default=True).ask())


def now_utc() -> datetime:
    return datetime.now(UTC)
