from datetime import datetime

from sluice.agents.config import PipelineConfig
from sluice.agents.llm import CachedLLM, LLMError, LLMRequest, LLMResponse, Message
from sluice.agents.state import (
    Call,
    Decision,
    FundamentalInput,
    Inputs,
    Meta,
    NewsInput,
    PipelineState,
    PriceInput,
    Report,
    Side,
    SignalInput,
    Turn,
    Verdict,
    cited_refs,
    malformed_citations,
)
from sluice.news.timeutil import ET
from sluice.universe import UNIVERSE

MARKET_ANALYST = "market_analyst"
NEWS_ANALYST = "news_analyst"
FUNDAMENTALS_ANALYST = "fundamentals_analyst"
BULL = "bull"
BEAR = "bear"
RESEARCH_MANAGER = "research_manager"
OUTSIDE_UNIVERSE = "Outside the studied universe: there is no backtest evidence for this ticker."
# Each analyst and the channel key of its report.
ANALYSTS = {MARKET_ANALYST: "market", NEWS_ANALYST: "news", FUNDAMENTALS_ANALYST: "fundamentals"}

PRICE_LABELS = {
    "ret_5d": "return over the last 5 trading days",
    "ret_20d": "return over the last 20 trading days",
    "ret_60d": "return over the last 60 trading days",
    "excess_5d": "return minus the S&P 500 ETF's over the last 5 trading days",
    "excess_20d": "return minus the S&P 500 ETF's over the last 20 trading days",
    "excess_60d": "return minus the S&P 500 ETF's over the last 60 trading days",
    "vol_20d": "annualised volatility of the last 20 daily returns",
    "vs_sma50": "last close relative to its 50-day average",
    "vs_sma200": "last close relative to its 200-day average",
    "rsi_14": "14-day RSI (0 to 100)",
    "volume_ratio_20d": "last day's volume relative to the average of the 20 days before",
}
FIGURE_LABELS = {
    "revenue": "revenue",
    "revenue_yoy": "revenue, change from the same quarter a year earlier",
    "gross_margin": "gross margin",
    "gross_margin_yoy": "gross margin, change from the same quarter a year earlier",
    "operating_margin": "operating margin",
    "operating_margin_yoy": "operating margin, change from the same quarter a year earlier",
    "net_income": "net income",
    "net_income_yoy": "net income, change from the same quarter a year earlier",
    "fcf": "free cash flow",
    "fcf_yoy": "free cash flow, change from the same quarter a year earlier",
    "pe_ttm": "market value over net income of the last four quarters",
    "ps_ttm": "market value over revenue of the last four quarters",
}


def _eastern(moment: datetime) -> str:
    return moment.astimezone(ET).strftime("%Y-%m-%d %H:%M ET")


def _header(meta: Meta) -> str:
    return f"Stock: {meta.ticker}\nDecision time: {_eastern(meta.decision_time)}"


def _calls(role: str, model: str, answers: list[LLMResponse]) -> list[Call]:
    return [
        Call(
            role=role,
            model=model,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            cached=answer.cached,
        )
        for answer in answers
    ]


def _failed(meta: Meta, role: str, error: LLMError, model: str) -> dict:
    """The update of a role that could not produce its output: the run ends without a rating."""
    return {
        "meta": meta.model_copy(update={"status": "failed", "failure": f"{role}: {error}"}),
        "usage": _calls(role, model, error.answers),
    }


def _request(config: PipelineConfig, role: str, user: str, json_mode: bool = False) -> LLMRequest:
    return LLMRequest(
        model=config.models[role],
        messages=(
            Message(role="system", content=config.prompts[role].strip()),
            Message(role="user", content=user),
        ),
        temperature=config.temperature,
        json_mode=json_mode,
    )


def _analyst(
    state: PipelineState,
    llm: CachedLLM,
    config: PipelineConfig,
    role: str,
    known: set[str],
    prompt: str,
    nothing: str,
) -> dict:
    """Writes one report that cites what it was shown. No call when it was shown nothing.

    `known` holds the reference ids in `prompt`: the only ones this analyst may cite.
    """
    meta, key = state["meta"], ANALYSTS[role]
    if not known:
        return {"reports": {key: Report(text=nothing, no_data=True)}}

    def check(answer: LLMResponse) -> Report:
        text = answer.text.strip()
        if malformed := malformed_citations(text):
            raise ValueError(f"the report has malformed citations: {malformed}")
        cited = cited_refs(text)
        if not cited:
            raise ValueError("the report cites no reference id")
        if unknown := sorted(cited - known):
            raise ValueError(f"the report cites ids that were not given: {unknown}")
        return Report(text=text)

    request = _request(config, role, prompt)
    try:
        report, answers = llm.complete_checked(request, check, meta.repeat, config.retries)
    except LLMError as error:
        return _failed(meta, role, error, request.model)
    return {"reports": {key: report}, "usage": _calls(role, request.model, answers)}


def _price_line(item: PriceInput) -> str:
    if item.name.startswith("rsi_"):
        shown = f"{item.value:.1f}"
    elif item.name.startswith("volume_ratio_"):
        shown = f"{item.value:.2f}x"
    elif item.name.startswith("vol_"):
        shown = f"{item.value:.2%}"
    else:
        shown = f"{item.value:+.2%}"
    return f"[{item.ref}] {PRICE_LABELS.get(item.name, item.name)}: {shown}"


def market_prompt(meta: Meta, inputs: Inputs) -> str:
    header = f"{_header(meta)}\nIndicators as of the close of {inputs.prices_as_of}:"
    return "\n".join([header, *(_price_line(item) for item in inputs.prices)])


def market_analyst(state: PipelineState, llm: CachedLLM, config: PipelineConfig) -> dict:
    meta, inputs = state["meta"], state["inputs"]
    known = {item.ref for item in inputs.prices}
    nothing = "No price data was available."
    return _analyst(
        state, llm, config, MARKET_ANALYST, known, market_prompt(meta, inputs), nothing
    )


def _news_line(item: NewsInput) -> str:
    facts = [_eastern(item.published_at), item.publisher or "publisher unknown"]
    if item.session:
        facts.append(f"session: {item.session}")
    if item.reprint_count:
        facts.append(f"reprints: {item.reprint_count}")
    # Only a weighted run has these; relevance only chose the items and is not shown.
    if item.scored is False:
        facts.append("not scored")
    if item.event_type:
        facts.append(f"type: {item.event_type}")
    if item.source_record:
        facts.append(f"source record: {item.source_record}")
    if item.type_record:
        facts.append(f"type record: {item.type_record}")
    lines = [f"[{item.ref}] " + " | ".join(facts), item.title]
    if item.summary:
        lines.append(item.summary)
    return "\n".join(lines)


def _signal_line(signal: SignalInput) -> str:
    return (
        f"[{signal.ref}] news signal: {signal.value:+.2e} (relevance-weighted over "
        f"{signal.n_items} scored items in the window; weights fitted as of "
        f"{signal.weights_as_of:%Y-%m-%d})"
    )


def news_prompt(meta: Meta, inputs: Inputs) -> str:
    head = [_header(meta), *(_signal_line(signal) for signal in inputs.signals), "News items:"]
    return "\n\n".join(["\n\n".join(head), *(_news_line(item) for item in inputs.news)])


def news_analyst(state: PipelineState, llm: CachedLLM, config: PipelineConfig) -> dict:
    meta, inputs = state["meta"], state["inputs"]
    known = {item.ref for item in (*inputs.news, *inputs.signals)}
    return _analyst(
        state, llm, config, NEWS_ANALYST, known, news_prompt(meta, inputs), "No news was available."
    )


def _figure_line(item: FundamentalInput) -> str:
    base = item.name.rsplit("_q", 1)[0]
    if item.unit == "amount":
        shown = f"{item.value:,.0f} {item.currency}"
    elif item.unit == "points":
        shown = f"{item.value * 100:+.2f} percentage points"
    elif item.unit == "multiple":
        shown = f"{item.value:.1f}x"
    else:
        shown = f"{item.value:+.2%}" if base.endswith("_yoy") else f"{item.value:.2%}"
    period = "four quarters to" if item.unit == "multiple" else "quarter ended"
    return f"[{item.ref}] {FIGURE_LABELS.get(base, base)}, {period} {item.quarter_end}: {shown}"


def fundamentals_prompt(meta: Meta, inputs: Inputs) -> str:
    header = f"{_header(meta)}\nFigures from the published financial statements:"
    return "\n".join([header, *(_figure_line(item) for item in inputs.fundamentals)])


def fundamentals_analyst(state: PipelineState, llm: CachedLLM, config: PipelineConfig) -> dict:
    meta, inputs = state["meta"], state["inputs"]
    known = {item.ref for item in inputs.fundamentals}
    nothing = "No financial statements were available."
    return _analyst(
        state, llm, config, FUNDAMENTALS_ANALYST, known, fundamentals_prompt(meta, inputs), nothing
    )


def has_data(reports: dict[str, Report]) -> bool:
    return any(not report.no_data for report in reports.values())


def citable_refs(inputs: Inputs, reports: dict[str, Report]) -> set[str]:
    """What roles after the analysts may cite: ids that a report cites.

    Reports were checked against the inputs, so these all exist there.
    """
    return set().union(*(cited_refs(r.text) for r in reports.values())) & inputs.refs()


def reports_prompt(meta: Meta, reports: dict[str, Report], debate: list[Turn]) -> str:
    """Reports and debate turns only: nobody after the analysts sees the pre-fetched inputs."""
    parts = [_header(meta)]
    for name, report in sorted(reports.items()):
        marker = " (no data)" if report.no_data else ""
        parts.append(f"## {name} report{marker}\n{report.text}")
    for turn in debate:
        parts.append(f"## debate, round {turn.round}, {turn.side}\n{turn.text}")
    return "\n\n".join(parts)


def debater(state: PipelineState, llm: CachedLLM, config: PipelineConfig, side: Side) -> dict:
    """One turn of the debate: argues one side from the reports and the turns so far."""
    meta, reports = state["meta"], state["reports"]
    debate = state.get("debate", [])
    round_number = sum(turn.side == side for turn in debate) + 1
    citable = citable_refs(state["inputs"], reports)

    def check(answer: LLMResponse) -> Turn:
        text = answer.text.strip()
        if malformed := malformed_citations(text):
            raise ValueError(f"the turn has malformed citations: {malformed}")
        cited = cited_refs(text)
        if not cited:
            raise ValueError("the turn cites no reference id")
        if unknown := sorted(cited - citable):
            raise ValueError(f"the turn cites ids that are not in the reports: {unknown}")
        return Turn(round=round_number, side=side, text=text)

    request = _request(config, side, reports_prompt(meta, reports, debate))
    try:
        turn, answers = llm.complete_checked(request, check, meta.repeat, config.retries)
    except LLMError as error:
        return _failed(meta, side, error, request.model)
    return {"debate": [turn], "usage": _calls(side, request.model, answers)}


def research_manager(state: PipelineState, llm: CachedLLM, config: PipelineConfig) -> dict:
    """Reads the reports, gives the rating. Its evidence may only cite what the reports cite."""
    meta, inputs, reports = state["meta"], state["inputs"], state["reports"]
    if not has_data(reports):
        failure = f"{RESEARCH_MANAGER}: no report has any data"
        return {"meta": meta.model_copy(update={"status": "failed", "failure": failure})}

    citable = citable_refs(inputs, reports)

    def check(answer: LLMResponse) -> Verdict:
        verdict = Verdict.model_validate_json(answer.text)
        used = {ref for evidence in verdict.evidence for ref in evidence.refs}
        if unknown := sorted(used - citable):
            raise ValueError(f"the evidence cites ids that are not in the reports: {unknown}")
        return verdict

    prompt = reports_prompt(meta, reports, state.get("debate", []))
    request = _request(config, RESEARCH_MANAGER, prompt, json_mode=True)
    try:
        verdict, answers = llm.complete_checked(request, check, meta.repeat, config.retries)
    except LLMError as error:
        return _failed(meta, RESEARCH_MANAGER, error, request.model)
    notes = [f"Gap in {gap.kind}: {gap.reason}." for gap in inputs.gaps]
    if meta.mode == "adhoc" and meta.ticker not in UNIVERSE:
        notes.append(OUTSIDE_UNIVERSE)
    return {
        "decision": Decision(**verdict.model_dump(), notes=notes),
        "usage": _calls(RESEARCH_MANAGER, request.model, answers),
    }
