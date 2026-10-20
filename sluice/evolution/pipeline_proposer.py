"""The LLM proposer for the prompt and rule layer (task 9.10).

docs/specs/08-agent-harness.md ("已确认（任务 9.10）"). One call to the deep
model gives one proposal: some rules, or the whole prompt of one of the three
analysts or the research manager; or nothing. Its material comes only from the
validation decision days the screen does not use, so the screen stays a set of
cells the proposer has never seen: the champion's tool use, its hit rate there,
and its five worst calls with the three reports and the manager's rationale.
It is told the rules and the gate, not anyone's conclusions.

The proposal is written to a folder (a JSON file, and the prompt as a text file
to read or edit) and screened only when the user runs `sluice evolve screen
--proposal <folder> --yes`. An edited prompt is screened as the user's own.
"""

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sluice.agents.config import PipelineConfig
from sluice.agents.llm import CachedLLM, LLMRequest, LLMResponse, Message
from sluice.evaluation.quality import SCORES
from sluice.evolution.gate import REPEATS, THRESHOLD
from sluice.evolution.pipeline_gate import SCREEN_DAYS
from sluice.evolution.pipeline_log import BUDGETS, PipelineEntry
from sluice.evolution.pipelines import PROMPT, PROMPT_GROWTH, RULES

MODEL = "deepseek-v4-pro"
RETRIES = 2
PROPOSABLE = ("market_analyst", "news_analyst", "fundamentals_analyst", "research_manager")
WORST = 5  # calls shown in full
SHOWN_RATIONALE = 400
DEFAULT_ROOT = Path("data/local/proposals")

SYSTEM = f"""\
You improve the prompts and rules of a research pipeline that rates US stocks every two
weeks. Three analysts each fetch their own data with tools, within a limit of calls: the
market analyst (price indicators, daily returns), the news analyst (news search) and the
fundamentals analyst (published statements). Each writes a report citing the reference ids
its tools returned, in square brackets. A bullish and a bearish colleague debate the
reports; the research manager then rates the stock (buy .. sell) in a JSON object, and may
first send each analyst one follow-up question. Code, not the pipeline, turns the ratings
into a portfolio.

Your proposal is judged in two stages, on decision days you are not shown:
- a screen: the rank correlation between ratings and the next 10 trading days' return
  against the S&P 500 must beat the current version's, on 55 fixed runs ({BUDGETS["screen"]}
  screens in all);
- if it passes and the user agrees, a full evaluation: every validation run, {REPEATS}
  times; the portfolio's Sharpe ratio must beat the current version's on every repeat and
  the Jobson-Korkie test of the difference must give z >= {THRESHOLD:g} ({BUDGETS["full"]} in all).

Change one layer only:
- rules: any of the keys listed with their ranges, or
- the whole prompt of ONE of {", ".join(PROPOSABLE)}, at most {PROMPT_GROWTH:g} times as
  long as now. Code checks every report's citations and the manager's JSON, so a prompt
  that drops those instructions makes runs fail. Never ask for data after the decision time
  or for a position size.
If no change seems worth a screen, propose none: an empty "changes".

Answer with JSON only:
{{"changes": {{"<key>": <value>}} or {{"prompts.<role>": "<the whole new prompt>"}},
 "rationale": "<why, in Simplified Chinese>"}}
"""


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: dict[str, Any]
    rationale: str = Field(min_length=1)

    @field_validator("changes")
    @classmethod
    def _proposable(cls, changes: dict[str, Any]) -> dict[str, Any]:
        roles = [k.removeprefix(PROMPT) for k in changes if k.startswith(PROMPT)]
        if wrong := [r for r in roles if r not in PROPOSABLE]:
            raise ValueError(f"prompts that may not be proposed: {wrong}; only {PROPOSABLE}")
        return changes


@dataclass(frozen=True)
class Call:
    """One of the champion's runs on a day the screen does not use, with its outcome."""

    ticker: str
    day: date
    state: dict
    label: float | None

    @property
    def rating(self) -> str | None:
        return (self.state.get("decision") or {}).get("rating")

    @property
    def miss(self) -> float:
        """How wrong a directional rating was: score times label, negative when wrong."""
        if self.rating is None or self.label is None:
            return 0.0
        return SCORES[self.rating] * self.label


def unseen_days(days: list[date]) -> list[date]:
    """The validation decision days the screen does not use."""
    return [d for i, d in enumerate(days) if i not in SCREEN_DAYS]


def _tool_use(calls: list[Call]) -> list[str]:
    per_role: dict[str, list[int]] = {}
    tools: Counter = Counter()
    queries: list[str] = []
    asked = changed = 0
    for call in calls:
        uses = call.state.get("retrieved", [])
        for role in ("market_analyst", "news_analyst", "fundamentals_analyst"):
            main = [u for u in uses if u["role"] == role and u.get("round", 0) == 0]
            per_role.setdefault(role, []).append(len(main))
        for use in uses:
            tools[use["tool"]] += 1
            if use["tool"] == "search_news" and len(queries) < 8:
                queries.append(use["arguments"].get("query", ""))
        if call.state.get("followups"):
            asked += 1
            prelim, final = call.state.get("preliminary") or {}, call.state.get("decision") or {}
            changed += prelim.get("rating") != final.get("rating")
    lines = [
        f"- {role}: {statistics.fmean(n):.1f} tool calls per report (min {min(n)}, max {max(n)})"
        for role, n in per_role.items()
        if n
    ]
    lines.append("- calls per tool: " + ", ".join(f"{t} {n}" for t, n in sorted(tools.items())))
    lines.append("- some news queries: " + "; ".join(repr(q) for q in queries))
    lines.append(
        f"- follow-up questions asked in {asked} of {len(calls)} runs; rating changed {changed}"
    )
    return lines


def _quality(calls: list[Call]) -> str:
    rated = [c for c in calls if c.rating is not None and c.label is not None]
    directional = [c for c in rated if SCORES[c.rating] != 0]
    hits = sum(c.miss > 0 for c in directional)
    ratings = Counter(c.rating or "failed" for c in calls)
    hit = f"{hits / len(directional):.0%}" if directional else "-"
    spread = ", ".join(f"{r} {n}" for r, n in sorted(ratings.items()))
    return (
        f"{len(calls)} runs; ratings: {spread}; directional ratings whose sign matched the "
        f"10-day excess return: {hits} of {len(directional)} ({hit})"
    )


def _case(call: Call) -> str:
    s = call.state
    decision = s.get("decision") or {}
    lines = [
        (
            f"### {call.ticker} {call.day}: rated {call.rating} (confidence "
            f"{decision.get('confidence')}), then the excess return was {call.label:+.2%}"
        )
    ]
    for name, report in sorted(s.get("reports", {}).items()):
        lines.append(f"{name} report:\n{report['text']}")
    lines.append(f"manager's rationale: {decision.get('rationale', '')}")
    return "\n\n".join(lines)


def _log(entries: list[PipelineEntry]) -> str:
    if not entries:
        return "(empty)"
    lines = []
    for e in entries:
        head = f"{e.stage} {e.n}" if e.n else f"{e.stage} (not run)"
        changes = ", ".join(
            f"{k} rewritten" if k.startswith(PROMPT) else f"{k}={v}" for k, v in e.changes.items()
        )
        lines.append(f"- {head} {e.status} against {e.parent}: {changes or 'nothing'}")
        if e.evidence is not None and e.stage == "screen":
            lines.append(
                f"  Spearman {e.evidence.candidate_spearman} vs {e.evidence.champion_spearman}"
            )
        elif e.evidence is not None:
            lines.append(f"  z {e.evidence.z:+.2f}")
        lines += [f"  refused: {p}" for p in e.problems]
        if e.rationale:
            lines.append(f"  rationale: {e.rationale[:SHOWN_RATIONALE]}")
    return "\n".join(lines)


def build_request(
    champion: PipelineConfig,
    remaining: dict[str, int],
    entries: list[PipelineEntry],
    calls: list[Call],
) -> LLMRequest:
    rules = "\n".join(
        f"- {key}: now {_value(champion, key)!r}; "
        + ("true or false" if b.kind is bool else f"{b.kind.__name__} {b.low:g}..{b.high:g}")
        for key, b in RULES.items()
    )
    prompts = "\n\n".join(
        f"## {role} ({len(champion.prompts[role])} characters)\n{champion.prompts[role].strip()}"
        for role in PROPOSABLE
    )
    worst = sorted((c for c in calls if c.miss < 0), key=lambda c: c.miss)[:WORST]
    user = "\n".join(
        [
            (
                f"Current version: {champion.version}-{champion.digest}. Screens left: "
                f"{remaining['screen']} of {BUDGETS['screen']}; full evaluations left: "
                f"{remaining['full']} of {BUDGETS['full']}."
            ),
            "",
            "Rules and their allowed ranges:",
            rules,
            "",
            "Prompts that may be rewritten:",
            prompts,
            "",
            "Experiment log, oldest first:",
            _log(entries),
            "",
            "The current version on validation days the screen does not use:",
            _quality(calls),
            *_tool_use(calls),
            "",
            f"Its {len(worst)} worst directional calls there, with what the team wrote:",
            "",
            "\n\n".join(_case(c) for c in worst) or "(none)",
            "",
            "Propose one change, or none, as JSON.",
        ]
    )
    return LLMRequest(
        model=MODEL,
        messages=(Message(role="system", content=SYSTEM), Message(role="user", content=user)),
        temperature=0.0,
        json_mode=True,
    )


def _value(config: PipelineConfig, key: str) -> Any:
    node: Any = config
    for part in key.split("."):
        node = getattr(node, part)
    return node


def ask(llm: CachedLLM, request: LLMRequest) -> tuple[Proposal, list[LLMResponse]]:
    return llm.complete_checked(
        request, lambda r: Proposal.model_validate_json(r.text), retries=RETRIES
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def save(proposal: Proposal, champion: PipelineConfig, root: Path = DEFAULT_ROOT) -> Path:
    """Write the proposal to a new folder: proposal.json, and <role>.txt for a prompt."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = root / f"{stamp}-{champion.version}"
    folder.mkdir(parents=True)
    record: dict[str, Any] = {
        "against": f"{champion.version}-{champion.digest}",
        "rationale": proposal.rationale,
        "rules": {k: v for k, v in proposal.changes.items() if not k.startswith(PROMPT)},
        "prompts": {},
    }
    for key, text in proposal.changes.items():
        if key.startswith(PROMPT):
            role = key.removeprefix(PROMPT)
            (folder / f"{role}.txt").write_text(text)
            record["prompts"][role] = {"file": f"{role}.txt", "sha": _digest(text)}
    (folder / "proposal.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return folder


def load(folder: Path) -> tuple[dict[str, Any], str, bool]:
    """(changes, rationale, edited): edited when a prompt file no longer matches the proposal."""
    record = json.loads((folder / "proposal.json").read_text())
    changes = dict(record["rules"])
    edited = False
    for role, saved in record["prompts"].items():
        text = (folder / saved["file"]).read_text()
        edited |= _digest(text) != saved["sha"]
        changes[PROMPT + role] = text
    return changes, record["rationale"], edited
