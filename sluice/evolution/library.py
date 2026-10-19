"""Sizing config library: search space, candidates, versions and the champion (task 7.2).

docs/specs/06-evolution.md ("已确认（任务 7.1）" and "（任务 7.2：配置库）"). The
evolution loop changes only `configs/sizing/`, and only the keys in `SPACE`,
within their ranges. A proposal is a set of changes, keyed the way `params`
flattens a config ("stocks.max_weight"). It becomes a candidate in memory for
the gatekeeper; only an accepted one is written, as the next `vN.toml`, copied
from the champion's text with just the changed values replaced, so the comments
that explain each key survive. Every version is kept; `CHAMPION` names the one
in use, and rolling back is pointing it at an older one.
"""

import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sluice.sizing.config import CONFIG_ROOT, SizingConfig, load_sizing_config

CHAMPION_FILE = "CHAMPION"
Changes = dict[str, Any]


@dataclass(frozen=True)
class Bound:
    kind: type  # float, int or bool
    low: float = 0.0  # ignored for bool
    high: float = 0.0


# What a proposal may change, and how far. Left out on purpose: the inputs'
# scoring_config and report_spec (another one means calling Jev again) and the
# exposure benchmark.
SPACE: dict[str, Bound] = {
    "k": Bound(float, 0, 2),
    "coefficients.rating": Bound(float, 0, 2),
    "coefficients.reports": Bound(float, 0, 2),
    "coefficients.news": Bound(float, 0, 2),
    "stocks.vol_lookback": Bound(int, 10, 120),
    "stocks.max_weight": Bound(float, 0.1, 0.5),
    "stocks.vol_scaling": Bound(bool),
    "exposure.vol_lookback": Bound(int, 10, 120),
    "exposure.target_vol": Bound(float, 0.08, 0.25),
    "exposure.floor": Bound(float, 0, 1),
    "exposure.trend_filter": Bound(bool),
    "exposure.trend_window": Bound(int, 50, 250),
    "exposure.trend_exposure": Bound(float, 0, 1),
    "inputs.news_lookback_days": Bound(int, 5, 30),
}


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for path, value in flat.items():
        *sections, key = path.split(".")
        node = tree
        for section in sections:
            node = node.setdefault(section, {})
        node[key] = value
    return tree


def params(config: SizingConfig) -> dict[str, Any]:
    """Every parameter as "section.key" -> value; the version name and file digest left out."""
    return _flatten(config.model_dump(exclude={"version", "digest"}))


def params_key(config: SizingConfig) -> str:
    """Short hash of the parameter values: the same for two files that differ only in comments."""
    text = json.dumps(params(config), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _right_type(value: Any, kind: type) -> bool:
    if kind is bool:
        return type(value) is bool
    if kind is int:
        return type(value) is int
    return type(value) in (int, float) and math.isfinite(value)


def check(champion: SizingConfig, changes: Changes) -> list[str]:
    """What is wrong with a proposal before anything is evaluated; empty when nothing is."""
    if not changes:
        return ["no change proposed"]
    current = params(champion)
    problems = []
    for key, value in changes.items():
        bound = SPACE.get(key)
        if bound is None:
            problems.append(f"{key}: not in the search space")
        elif not _right_type(value, bound.kind):
            problems.append(f"{key}: expected {bound.kind.__name__}, got {value!r}")
        elif bound.kind is not bool and not bound.low <= value <= bound.high:
            problems.append(f"{key}: {value} outside {bound.low}..{bound.high}")
        elif value == current[key]:
            problems.append(f"{key}: already {value!r}")
    return problems


def candidate(champion: SizingConfig, changes: Changes) -> SizingConfig:
    """The champion with `changes` applied, in memory only. Its digest is its `params_key`."""
    problems = check(champion, changes)
    if problems:
        raise ValueError("; ".join(problems))
    tree = _unflatten({**params(champion), **changes})
    config = SizingConfig(version="candidate", digest="", **tree)
    return config.model_copy(update={"digest": params_key(config)})


def _literal(value: Any, kind: type) -> str:
    if kind is bool:
        return "true" if value else "false"
    if kind is int:
        return str(value)
    return repr(float(value))


SECTION = re.compile(r"^\[(?P<name>[A-Za-z_][\w.]*)\]")
ASSIGNMENT = re.compile(r"^(?P<key>[A-Za-z_]\w*)\s*=\s*(?P<value>[^#]*?)(?P<comment>\s*#.*)?$")


def render(source: str, changes: Changes, header: list[str]) -> str:
    """`source` with the leading comment block replaced by `header` and only `changes` rewritten.

    A changed key missing from the text (one with a default, such as
    stocks.vol_scaling in v1) is added after the last key of its section.
    """
    lines = source.splitlines()
    start = 0
    while start < len(lines) and lines[start].startswith("#"):
        start += 1
    out = [f"# {line}" for line in header]
    pending = dict(changes)
    section = ""
    last_key: dict[str, int] = {}  # section -> index in `out` of its last assignment
    for line in lines[start:]:
        if match := SECTION.match(line):
            section = match["name"]
            last_key[section] = len(out)  # an empty section takes new keys right after its header
        elif match := ASSIGNMENT.match(line):
            path = f"{section}.{match['key']}" if section else match["key"]
            if path in pending:
                value = _literal(pending.pop(path), SPACE[path].kind)
                line = f"{match['key']} = {value}{match['comment'] or ''}"
            last_key[section] = len(out)
        out.append(line)
    # Insert from the bottom up so earlier indices stay valid.
    for path in sorted(pending, key=lambda p: last_key.get(p.rpartition(".")[0], -1), reverse=True):
        section, _, key = path.rpartition(".")
        if section not in last_key:
            raise ValueError(f"{path}: no [{section}] section in the source")
        value = _literal(pending[path], SPACE[path].kind)
        out.insert(last_key[section] + 1, f"{key} = {value}  # added by the evolution loop")
    return "\n".join(out) + "\n"


def _next_version(folder: Path) -> str:
    numbers = [
        int(match[1]) for path in folder.glob("v*.toml")
        if (match := re.fullmatch(r"v(\d+)", path.stem))
    ]  # fmt: skip
    return f"v{max(numbers, default=0) + 1}"


def write_version(
    champion: SizingConfig, changes: Changes, root: Path = CONFIG_ROOT
) -> SizingConfig:
    """Write the champion plus `changes` as the next `vN.toml`; call only once it is accepted."""
    expected = candidate(champion, changes)
    folder = root / "sizing"
    if load_sizing_config(champion.version, root).digest != champion.digest:
        raise ValueError(f"{champion.version}.toml has changed since it was loaded")
    version = _next_version(folder)
    current = params(champion)
    described = ", ".join(f"{key} {current[key]!r} → {value!r}" for key, value in changes.items())
    header = [
        f"Sizing config, version {version.removeprefix('v')}: {champion.version} with {described}.",
        "Written by the evolution loop once the gatekeeper accepted it (docs/specs/06-evolution.md).",
        "A change to a version that has been evaluated is a new file.",
    ]
    text = render((folder / f"{champion.version}.toml").read_text(), changes, header)
    written = SizingConfig(version=version, digest="", **tomllib.loads(text))
    if params(written) != params(expected):
        raise ValueError(f"{version}.toml would not hold the proposed parameters; nothing written")
    with (folder / f"{version}.toml").open("x") as file:  # never overwrite a version
        file.write(text)
    return load_sizing_config(version, root)


def champion(root: Path = CONFIG_ROOT) -> str:
    return (root / "sizing" / CHAMPION_FILE).read_text().strip()


def load_champion(root: Path = CONFIG_ROOT) -> SizingConfig:
    return load_sizing_config(champion(root), root)


def set_champion(version: str, root: Path = CONFIG_ROOT) -> None:
    """Point `CHAMPION` at `version`, for an accepted change or a rollback. It must load."""
    load_sizing_config(version, root)
    (root / "sizing" / CHAMPION_FILE).write_text(f"{version}\n")


def next_version(root: Path = CONFIG_ROOT) -> str:
    """The name the next written version will get."""
    return _next_version(root / "sizing")


def version_with(key: str, root: Path = CONFIG_ROOT) -> str | None:
    """The existing version whose parameters have this `params_key`, if any."""
    for path in sorted((root / "sizing").glob("v*.toml")):
        if params_key(load_sizing_config(path.stem, root)) == key:
            return path.stem
    return None
