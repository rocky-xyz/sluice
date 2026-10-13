from datetime import datetime

from sluice.agents.config import PipelineConfig
from sluice.agents.llm import CachedLLM, LLMError, LLMRequest, LLMResponse, Message
from sluice.agents.state import (
    Call,
    Decision,
    Inputs,
    Meta,
    NewsInput,
    PipelineState,
    Report,
    Verdict,
    cited_refs,
)
from sluice.news.timeutil import ET

NEWS_ANALYST = "news_analyst"
RESEARCH_MANAGER = "research_manager"
# Channel key of each analyst's report.
REPORT_KEYS = {NEWS_ANALYST: "news"}


def _eastern(moment: datetime) -> str:
    return moment.astimezone(ET).strftime("%Y-%m-%d %H:%M ET")


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


def _news_line(item: NewsInput) -> str:
    facts = [_eastern(item.published_at), item.publisher or "publisher unknown"]
    if item.session:
        facts.append(f"session: {item.session}")
    if item.reprint_count:
        facts.append(f"reprints: {item.reprint_count}")
    lines = [f"[{item.ref}] " + " | ".join(facts), item.title]
    if item.summary:
        lines.append(item.summary)
    return "\n".join(lines)


def news_prompt(meta: Meta, inputs: Inputs) -> str:
    header = f"Stock: {meta.ticker}\nDecision time: {_eastern(meta.decision_time)}\n\nNews items:"
    return "\n\n".join([header, *(_news_line(item) for item in inputs.news)])


def news_analyst(state: PipelineState, llm: CachedLLM, config: PipelineConfig) -> dict:
    """Reads the pre-fetched news, writes a report that cites it. No call when there is no news."""
    meta, inputs = state["meta"], state["inputs"]
    key = REPORT_KEYS[NEWS_ANALYST]
    if not inputs.news:
        return {"reports": {key: Report(text="No news was available.", no_data=True)}}

    known = {item.ref for item in inputs.news}  # the only sources this analyst was shown

    def check(answer: LLMResponse) -> Report:
        text = answer.text.strip()
        cited = cited_refs(text)
        if not cited:
            raise ValueError("the report cites no reference id")
        if unknown := sorted(cited - known):
            raise ValueError(f"the report cites ids that were not given: {unknown}")
        return Report(text=text)

    request = _request(config, NEWS_ANALYST, news_prompt(meta, inputs))
    try:
        report, answers = llm.complete_checked(request, check, meta.repeat, config.retries)
    except LLMError as error:
        return _failed(meta, NEWS_ANALYST, error, request.model)
    return {"reports": {key: report}, "usage": _calls(NEWS_ANALYST, request.model, answers)}


def manager_prompt(meta: Meta, reports: dict[str, Report]) -> str:
    """Reports only: the manager never sees the pre-fetched inputs."""
    parts = [f"Stock: {meta.ticker}\nDecision time: {_eastern(meta.decision_time)}"]
    for name, report in sorted(reports.items()):
        marker = " (no data)" if report.no_data else ""
        parts.append(f"## {name} report{marker}\n{report.text}")
    return "\n\n".join(parts)


def research_manager(state: PipelineState, llm: CachedLLM, config: PipelineConfig) -> dict:
    """Reads the reports, gives the rating. Its evidence may only cite what the reports cite."""
    meta, inputs, reports = state["meta"], state["inputs"], state["reports"]
    if all(report.no_data for report in reports.values()):
        failure = f"{RESEARCH_MANAGER}: no report has any data"
        return {"meta": meta.model_copy(update={"status": "failed", "failure": failure})}

    # Reports were checked against the inputs, so these ids all exist there.
    citable = set().union(*(cited_refs(r.text) for r in reports.values())) & inputs.refs()

    def check(answer: LLMResponse) -> Verdict:
        verdict = Verdict.model_validate_json(answer.text)
        used = {ref for evidence in verdict.evidence for ref in evidence.refs}
        if unknown := sorted(used - citable):
            raise ValueError(f"the evidence cites ids that are not in the reports: {unknown}")
        return verdict

    request = _request(config, RESEARCH_MANAGER, manager_prompt(meta, reports), json_mode=True)
    try:
        verdict, answers = llm.complete_checked(request, check, meta.repeat, config.retries)
    except LLMError as error:
        return _failed(meta, RESEARCH_MANAGER, error, request.model)
    notes = [f"No {gap.kind}: {gap.reason}." for gap in inputs.gaps]
    return {
        "decision": Decision(**verdict.model_dump(), notes=notes),
        "usage": _calls(RESEARCH_MANAGER, request.model, answers),
    }
