"""Pipeline config library: rules and prompts the evolution loop may change (task 9.8).

docs/specs/08-agent-harness.md ("已确认（任务 9.8）"). The sizing library's
counterpart for `configs/pipeline/`: a proposal changes either rules (keys in
`RULES`, within their ranges) or one role's whole prompt, never both. It
becomes a candidate in memory; only an accepted one is written, as the next
`vN.toml`, copied from the champion's text with just the changed parts
replaced. A new prompt is written as a literal string ('''...'''), so that
nothing in it is read as an escape. `CHAMPION` names the version in use.
"""

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from sluice.agents.config import CONFIG_ROOT, PipelineConfig, load_config
from sluice.evolution.library import (
    ASSIGNMENT,
    SECTION,
    Bound,
    Changes,
    _flatten,
    _literal,
    _next_version,
    _right_type,
    _unflatten,
)

CHAMPION_FILE = "CHAMPION"
PROMPT = "prompts."
PROMPT_GROWTH = 1.5  # a new prompt may be at most this much longer than the one it replaces

# Rules a proposal may change, and how far. Left out on purpose: the models and the
# temperature (cost and repeatability), fundamentals.lag_days (it keeps unpublished
# quarters out), the price indicators' definitions, and fundamentals.quarters.
RULES: dict[str, Bound] = {
    "tools.max_calls": Bound(int, 1, 10),
    "tools.followup_rounds": Bound(int, 0, 2),
    "tools.followup_calls": Bound(int, 0, 5),
    "tools.search_max_limit": Bound(int, 5, 30),
    "tools.history_max_days": Bound(int, 20, 250),
    "tools.pass_reasoning": Bound(bool),
    "news.lookback_days": Bound(int, 7, 28),
    "debate_rounds": Bound(int, 0, 2),
}
Layer = Literal["rules", "prompt"]


def normalise(prompt: str) -> str:
    """How a prompt is stored: no surrounding blank space, one final newline."""
    return prompt.strip() + "\n"


def params(config: PipelineConfig) -> dict[str, Any]:
    """Every parameter as "section.key" -> value, prompts included; no version or digest."""
    return _flatten(config.model_dump(exclude={"version", "digest"}))


def params_key(config: PipelineConfig) -> str:
    text = json.dumps(params(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def layer(changes: Changes) -> Layer | None:
    prompts = [k for k in changes if k.startswith(PROMPT)]
    if prompts and len(prompts) == len(changes):
        return "prompt"
    return "rules" if not prompts else None


def check(champion: PipelineConfig, changes: Changes) -> list[str]:
    """What is wrong with a proposal before anything is run; empty when nothing is."""
    if not changes:
        return ["no change proposed"]
    prompts = [k for k in changes if k.startswith(PROMPT)]
    if prompts and len(prompts) != len(changes):
        return ["rules and a prompt in one proposal: change one layer at a time"]
    if len(prompts) > 1:
        return ["more than one prompt: change one role's prompt at a time"]
    current = params(champion)
    problems = []
    for key, value in changes.items():
        if key.startswith(PROMPT):
            role = key.removeprefix(PROMPT)
            if role not in champion.prompts:
                problems.append(f"{key}: there is no role {role!r}")
            elif not isinstance(value, str) or not value.strip():
                problems.append(f"{key}: a prompt must be non-empty text")
            elif "'''" in value:
                problems.append(f"{key}: a prompt may not contain '''")
            elif len(normalise(value)) > PROMPT_GROWTH * len(champion.prompts[role]):
                limit = int(PROMPT_GROWTH * len(champion.prompts[role]))
                problems.append(f"{key}: {len(normalise(value))} characters, at most {limit}")
            elif normalise(value) == champion.prompts[role]:
                problems.append(f"{key}: unchanged")
            continue
        bound = RULES.get(key)
        if bound is None:
            problems.append(f"{key}: not a rule the loop may change")
        elif key not in current:
            problems.append(f"{key}: the champion has no such setting")
        elif not _right_type(value, bound.kind):
            problems.append(f"{key}: expected {bound.kind.__name__}, got {value!r}")
        elif bound.kind is not bool and not bound.low <= value <= bound.high:
            problems.append(f"{key}: {value} outside {bound.low}..{bound.high}")
        elif value == current[key]:
            problems.append(f"{key}: already {value!r}")
    return problems


def _applied(changes: Changes) -> Changes:
    return {k: normalise(v) if k.startswith(PROMPT) else v for k, v in changes.items()}


def candidate(champion: PipelineConfig, changes: Changes) -> PipelineConfig:
    """The champion with `changes` applied, in memory only. Its digest is its `params_key`."""
    if problems := check(champion, changes):
        raise ValueError("; ".join(problems))
    tree = _unflatten({**params(champion), **_applied(changes)})
    config = PipelineConfig(version="candidate", digest="", **tree)
    return config.model_copy(update={"digest": params_key(config)})


_PROMPT_OPEN = re.compile(r"^(?P<role>[A-Za-z_]\w*) = (?P<quote>\"\"\"|''')\s*$")


def render(source: str, changes: Changes, header: list[str]) -> str:
    """`source` with its leading comments replaced by `header` and only `changes` rewritten."""
    lines = source.splitlines()
    start = 0
    while start < len(lines) and lines[start].startswith("#"):
        start += 1
    out = [f"# {line}" for line in header]
    pending = _applied(changes)
    section, i = "", start
    while i < len(lines):
        line = lines[i]
        if match := SECTION.match(line):
            section = match["name"]
        elif section == "prompts" and (match := _PROMPT_OPEN.match(line)):
            end = next(j for j in range(i + 1, len(lines)) if lines[j].strip() == match["quote"])
            key = PROMPT + match["role"]
            if key in pending:
                out += [f"{match['role']} = '''", *pending.pop(key).splitlines(), "'''"]
            else:
                out += lines[i : end + 1]
            i = end + 1
            continue
        elif match := ASSIGNMENT.match(line):
            path = f"{section}.{match['key']}" if section else match["key"]
            if path in pending:
                value = _literal(pending.pop(path), RULES[path].kind)
                line = f"{match['key']} = {value}{match['comment'] or ''}"
        out.append(line)
        i += 1
    if pending:
        raise ValueError(f"not found in the champion's file: {', '.join(sorted(pending))}")
    return "\n".join(out) + "\n"


def write_version(
    champion: PipelineConfig, changes: Changes, root: Path = CONFIG_ROOT
) -> PipelineConfig:
    """Write the champion plus `changes` as the next `vN.toml`; call only once it is accepted."""
    expected = candidate(champion, changes)
    folder = root / "pipeline"
    if load_config(champion.version, root).digest != champion.digest:
        raise ValueError(f"{champion.version}.toml has changed since it was loaded")
    version = _next_version(folder)
    current = params(champion)
    described = ", ".join(
        f"{k} rewritten" if k.startswith(PROMPT) else f"{k} {current[k]!r} → {v!r}"
        for k, v in changes.items()
    )
    header = [
        f"Pipeline config, version {version.removeprefix('v')}: {champion.version} with {described}.",
        "Written by the evolution loop once the gatekeeper accepted it (docs/specs/08).",
        "A change to a version that has been run is a new file.",
    ]
    text = render((folder / f"{champion.version}.toml").read_text(), changes, header)
    written = PipelineConfig(version=version, digest="", **tomllib.loads(text))
    if params(written) != params(expected):
        raise ValueError(f"{version}.toml would not hold the proposed parameters; nothing written")
    with (folder / f"{version}.toml").open("x") as file:
        file.write(text)
    return load_config(version, root)


def champion(root: Path = CONFIG_ROOT) -> str:
    return (root / "pipeline" / CHAMPION_FILE).read_text().strip()


def load_champion(root: Path = CONFIG_ROOT) -> PipelineConfig:
    return load_config(champion(root), root)


def set_champion(version: str, root: Path = CONFIG_ROOT) -> None:
    load_config(version, root)
    (root / "pipeline" / CHAMPION_FILE).write_text(f"{version}\n")


def version_with(key: str, root: Path = CONFIG_ROOT) -> str | None:
    """The existing version whose parameters have this `params_key`, if any."""
    for path in sorted((root / "pipeline").glob("v*.toml")):
        if params_key(load_config(path.stem, root)) == key:
            return path.stem
    return None
