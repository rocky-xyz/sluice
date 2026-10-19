"""The LLM proposer: one sizing change per call, for the gatekeeper to judge (task 7.6).

docs/specs/06-evolution.md ("已确认（任务 7.6：LLM 提议器）"). The model sees how
the formula works, the champion's config file with its comments, the search
space, the gate's rules, the trials left, the experiment log and the champion's
portfolio period by period. It is not told what anyone concluded from these
results. It may propose nothing; that is logged and uses no trial. An answer
that is not the JSON asked for is retried; one that breaks a rule is passed on
and refused by the gate, so the mistake shows in the log.

The prompt lives here, not in `configs/`: the proposer is not what evolves.
"""

import json
import statistics
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sluice.agents.llm import CachedLLM, LLMRequest, LLMResponse, Message
from sluice.evaluation import metrics
from sluice.evolution.experiments import MAX_TRIALS, Entry
from sluice.evolution.gate import REPEATS, THRESHOLD
from sluice.evolution.library import SPACE
from sluice.evolution.validation import Period
from sluice.sizing.config import SizingConfig

MODEL = "deepseek-v4-pro"
RETRIES = 2  # answers that are not the JSON asked for, as in the pipeline
RATIONALE_SHOWN = 400  # characters of an earlier rationale repeated in the prompt

SYSTEM = f"""\
You tune the position-sizing config of a research tool that holds a portfolio of
US stocks. A pipeline of language-model analysts rates each stock every two weeks;
the sizing formula, written in code, turns those ratings and two other signals into
weights. You may change only the sizing config's parameters, never the ratings.

How the formula works, on each decision day:
1. Three inputs per stock: the pipeline's rating (buy +2 .. sell -2), a language
   model's stance on the analysts' three reports, and a news signal from scored
   items over the last `inputs.news_lookback_days` days. Each is turned into a
   cross-sectional z-score over the stocks (a missing value counts as 0).
2. s = the coefficient-weighted average of the three z-scores (`coefficients.*`).
3. tilt = max(0, 1 + k·s).
4. If `stocks.vol_scaling`, the tilt is divided by the stock's realised volatility
   over `stocks.vol_lookback` trading days.
5. Normalised to sum to 1; no stock above `stocks.max_weight`, the excess shared
   among the others pro rata.
6. Multiplied by the exposure E = target_vol / SPY's realised volatility over
   `exposure.vol_lookback` days, at most 1 and at least `exposure.floor`; with
   `exposure.trend_filter`, also at most `exposure.trend_exposure` while SPY closes
   below its `exposure.trend_window`-day average. The rest is cash, earning 0.
The portfolio trades at the next open after each decision day, paying 0.1% of the
value traded, and is left to drift until the next one.

The gatekeeper accepts a proposal only if both hold, on the validation period:
- its annualised Sharpe ratio is above the champion's on every one of the
  {REPEATS} repeats of the pipeline's ratings, and
- on the repeats' daily returns averaged day by day, the Jobson-Korkie test of
  the Sharpe difference (Memmel's correction) gives z >= {THRESHOLD:g}.
An accepted proposal becomes the champion. Each proposal that gets evaluated uses
one of {MAX_TRIALS} trials for the whole search, accepted or not. A proposal outside
the allowed keys or ranges, or with the parameters of an existing version or of an
earlier trial, is refused before evaluation and uses no trial. When you see no change
worth a trial, propose none: an empty `changes`.

Answer with JSON only:
{{"changes": {{"<key>": <value>, ...}}, "rationale": "<why, in Simplified Chinese>"}}
Keys are written as in the allowed list, e.g. "stocks.max_weight". Integers for
integer keys, true/false for switches.
"""


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: dict[str, Any]  # passed on as given: the gate refuses what breaks its rules
    rationale: str = Field(min_length=1)


def _space() -> str:
    lines = []
    for key, bound in SPACE.items():
        if bound.kind is bool:
            lines.append(f"- {key}: true or false")
        else:
            lines.append(f"- {key}: {bound.kind.__name__} from {bound.low:g} to {bound.high:g}")
    return "\n".join(lines)


def _log(entries: list[Entry]) -> str:
    if not entries:
        return "(empty)"
    lines = []
    for e in entries:
        head = f"#{e.n}" if e.n else "(no trial)"
        changes = json.dumps(e.changes, ensure_ascii=False) if e.changes else "{}"
        lines.append(f"{head} {e.status}, by {e.source}, against {e.parent or '-'}: {changes}")
        if ev := e.evidence:
            pairs = ", ".join(
                f"{c:.2f} vs {h:.2f}"
                for c, h in zip(ev.candidate_sharpes, ev.champion_sharpes, strict=True)
            )
            lines.append(
                f"  Sharpe per repeat (proposal vs champion): {pairs}; averaged "
                f"{ev.candidate_sharpe:.2f} vs {ev.champion_sharpe:.2f}, rho {ev.rho:.3f}, "
                f"z {ev.z:+.2f}"
            )
        lines += [f"  refused: {p}" for p in e.problems]
        if e.rationale:
            lines.append(f"  rationale: {e.rationale[:RATIONALE_SHOWN]}")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def _material(periods: list[Period], sharpes: dict[int, float | None], tickers: list[str]) -> str:
    per_repeat = ", ".join(f"{s:.2f}" if s is not None else "-" for s in sharpes.values())
    lines = [
        f"Annualised Sharpe per repeat, after costs: {per_repeat}",
        "",
        "Each period runs from the open after a decision day to the open after the next;",
        "returns before costs, weights and contributions averaged over the repeats.",
        "",
        "| start | E | portfolio | SPY | equal weight | portfolio - SPY |",
        "|---|---|---|---|---|---|",
    ]
    for p in periods:
        lines.append(
            f"| {p.start} | {p.exposure:.2f} | {_pct(p.portfolio)} | {_pct(p.benchmark)} "
            f"| {_pct(p.equal)} | {_pct(p.portfolio - p.benchmark)} |"
        )
    lines += ["", "Target weights (% of the book):", "", "| start | " + " | ".join(tickers) + " |"]
    lines.append("|---" * (len(tickers) + 1) + "|")
    for p in periods:
        cells = " | ".join(f"{100 * p.weights[t]:.1f}" for t in tickers)
        lines.append(f"| {p.start} | {cells} |")
    lines += ["", "Contribution to the period's return (weight × the stock's return, in bp):", ""]
    lines.append("| start | " + " | ".join(tickers) + " |")
    lines.append("|---" * (len(tickers) + 1) + "|")
    for p in periods:
        cells = " | ".join(f"{1e4 * p.contributions[t]:+.0f}" for t in tickers)
        lines.append(f"| {p.start} | {cells} |")
    totals = {t: sum(p.contributions[t] for p in periods) for t in tickers}
    lines.append("| total | " + " | ".join(f"{1e4 * totals[t]:+.0f}" for t in tickers) + " |")
    worst = sorted(periods, key=lambda p: p.portfolio - p.benchmark)[:3]
    lines += ["", "The three periods furthest behind SPY, biggest drags first:"]
    for p in worst:
        drags = sorted(p.contributions.items(), key=lambda kv: kv[1])
        listed = ", ".join(f"{t} {1e4 * c:+.0f}bp" for t, c in drags)
        lines.append(f"- {p.start} ({_pct(p.portfolio - p.benchmark)} vs SPY): {listed}")
    return "\n".join(lines)


def build_request(
    champion: SizingConfig,
    champion_text: str,
    remaining: int,
    entries: list[Entry],
    periods: list[Period],
    returns: dict[int, list[float]],
    tickers: list[str],
) -> LLMRequest:
    sharpes = {r: metrics.sharpe(rs) for r, rs in returns.items()}
    days = statistics.fmean(len(rs) for rs in returns.values())
    user = "\n".join(
        [
            (
                f"Champion: {champion.version}-{champion.digest}. Trials left: {remaining} "
                f"of {MAX_TRIALS}. Validation period: {len(periods)} decision periods, "
                f"{days:.0f} trading days, {len(tickers)} stocks."
            ),
            "",
            "The champion's config file:",
            "```toml",
            champion_text.strip(),
            "```",
            "",
            "Allowed keys and ranges:",
            _space(),
            "",
            "Experiment log, oldest first:",
            _log(entries),
            "",
            "The champion's portfolio on the validation period:",
            _material(periods, sharpes, tickers),
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


def ask(llm: CachedLLM, request: LLMRequest) -> tuple[Proposal, list[LLMResponse]]:
    """The model's proposal; answers that are not the JSON asked for are retried."""
    return llm.complete_checked(
        request, lambda response: Proposal.model_validate_json(response.text), retries=RETRIES
    )
