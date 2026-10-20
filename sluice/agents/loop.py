"""An analyst that fetches its own data with tools, within limits (task 9.4).

docs/specs/08-agent-harness.md. The loop runs inside the analyst's node: the
model is offered its own tools (`tools.ROLE_TOOLS`) until it writes its report
or uses up `max_calls`, after which the tools are switched off and it must
write. A report that cites nothing, cites malformed or unretrieved ids is sent
back with what is wrong, tools still off, at most `retries` times. Every step's
request holds the whole conversation so far, and tool results are fixed by the
stores and the decision time, so a rerun replays every step from the cache.
"""

from sluice.agents import tools
from sluice.agents.config import PipelineConfig
from sluice.agents.llm import CachedLLM, LLMError, LLMRequest, LLMResponse, Message
from sluice.agents.roles import ANALYSTS, _calls, _failed
from sluice.agents.state import PipelineState, Report, ToolUse, cited_refs, malformed_citations

NO_DATA = {
    "market_analyst": "No price data was available.",
    "news_analyst": "No news was available.",
    "fundamentals_analyst": "No financial statements were available.",
}
LIMIT_REACHED = "Error: the limit of tool calls is reached; write the report now."


def report_problem(text: str, known: set[str]) -> str | None:
    """What is wrong with a report, in words for the model; None when nothing is."""
    if malformed := malformed_citations(text):
        return f"Your report has malformed citations: {', '.join(malformed)}"
    cited = cited_refs(text)
    if not cited:
        return "Your report cites no reference id"
    if unknown := sorted(cited - known):
        return f"Your report cites ids the tools did not return: {', '.join(unknown)}"
    return None


def tool_analyst(
    state: PipelineState,
    llm: CachedLLM,
    config: PipelineConfig,
    role: str,
    ctx: tools.ToolContext,
) -> dict:
    meta, key, rules = state["meta"], ANALYSTS[role], ctx.rules
    model = config.models[role]
    offered = tools.specs(config, tools.ROLE_TOOLS[role])
    messages = [
        Message(role="system", content=config.prompts[role].strip()),
        Message(role="user", content=tools.describe(ctx)),
    ]
    uses: list[ToolUse] = []
    known: set[str] = set()
    answers: list[LLMResponse] = []
    fixes = 0
    closed = False  # tools off: the call limit is reached, or a report is being fixed

    def finish(report: Report) -> dict:
        return {"reports": {key: report}, "retrieved": uses, "usage": _calls(role, model, answers)}

    while True:
        closed = closed or len(uses) >= rules.max_calls
        request = LLMRequest(
            model=model,
            messages=tuple(messages),
            temperature=config.temperature,
            tools=offered,
            tool_choice="none" if closed else None,
        )
        try:
            response, got = llm.complete_checked(request, lambda r: r, meta.repeat, config.retries)
        except LLMError as error:
            error.answers = [*answers, *error.answers]
            return {**_failed(meta, role, error, model), "retrieved": uses}
        answers += got

        if response.tool_calls and not closed:
            messages.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                    reasoning_content=response.reasoning if rules.pass_reasoning else None,
                )
            )
            for asked in response.tool_calls:  # every call needs an answer, even past the limit
                if len(uses) >= rules.max_calls:
                    content = LIMIT_REACHED
                else:
                    result = tools.call(ctx, asked.name, asked.arguments)
                    uses.append(
                        ToolUse(
                            role=role, step=len(uses) + 1, tool=asked.name,
                            arguments=result.arguments, refs=result.refs, notes=result.notes,
                            error=result.error, items=result.items,
                        )
                    )  # fmt: skip
                    known.update(result.refs)
                    content = result.content
                messages.append(Message(role="tool", content=content, tool_call_id=asked.id))
            continue

        if not known:  # nothing to analyse: written by code, as in v1
            return finish(Report(text=NO_DATA[role], no_data=True))
        text = response.text.strip()
        problem = report_problem(text, known)
        if problem is None:
            return finish(Report(text=text))
        if fixes >= config.retries:
            error = LLMError(f"no acceptable report after {fixes + 1} tries: {problem}")
            error.answers = answers
            return {**_failed(meta, role, error, model), "retrieved": uses}
        fixes += 1
        closed = True
        messages += [
            Message(role="assistant", content=text),
            Message(
                role="user",
                content=f"{problem}. Rewrite the report, citing in square brackets only "
                "ids the tools returned.",
            ),
        ]
