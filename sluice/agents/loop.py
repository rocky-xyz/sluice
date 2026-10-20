"""An analyst that fetches its own data with tools, within limits (tasks 9.4, 9.5).

docs/specs/08-agent-harness.md. The loop runs inside the analyst's node: the
model is offered its own tools (`tools.ROLE_TOOLS`) until it writes its text
or uses up its calls, after which the tools are switched off and it must
write. A text that cites nothing, cites malformed or unretrieved ids is sent
back with what is wrong, tools still off, at most `retries` times. Every step's
request holds the whole conversation so far, and tool results are fixed by the
stores and the decision time, so a rerun replays every step from the cache.

The same loop answers the research manager's follow-up questions, with fewer
calls and the ids the analyst fetched for its report still citable. A
follow-up that cannot be answered is noted as such; it does not fail the run.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from sluice.agents import tools
from sluice.agents.config import PipelineConfig
from sluice.agents.llm import CachedLLM, LLMError, LLMRequest, LLMResponse, Message
from sluice.agents.roles import ANALYSTS, _calls, _failed
from sluice.agents.state import (
    FollowUp,
    PipelineState,
    Report,
    ToolUse,
    cited_refs,
    malformed_citations,
)

NO_DATA = {
    "market_analyst": "No price data was available.",
    "news_analyst": "No news was available.",
    "fundamentals_analyst": "No financial statements were available.",
}
LIMIT_REACHED = "Error: the limit of tool calls is reached; write the report now."
ROLE_OF = {key: role for role, key in ANALYSTS.items()}  # "news" -> "news_analyst"


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


@dataclass
class Conversation:
    """What one tool loop produced: its text, or why there is none."""

    text: str | None = None
    no_data: bool = False
    error: LLMError | None = None
    uses: list[ToolUse] = field(default_factory=list)
    answers: list[LLMResponse] = field(default_factory=list)


def converse(
    llm: CachedLLM,
    config: PipelineConfig,
    role: str,
    ctx: tools.ToolContext,
    messages: list[Message],
    max_calls: int,
    repeat: int,
    known: set[str] | None = None,
    first_step: int = 1,
    round_: int = 0,
) -> Conversation:
    """Run the tool loop from `messages` until an acceptable text, no data, or a failure."""
    out = Conversation()
    known = set(known or ())
    offered = tools.specs(config, tools.ROLE_TOOLS[role])
    model, rules = config.models[role], ctx.rules
    messages = list(messages)
    fixes = 0
    closed = False  # tools off: the call limit is reached, or a text is being fixed
    while True:
        closed = closed or len(out.uses) >= max_calls
        request = LLMRequest(
            model=model,
            messages=tuple(messages),
            temperature=config.temperature,
            tools=offered,
            tool_choice="none" if closed else None,
        )
        try:
            response, got = llm.complete_checked(request, lambda r: r, repeat, config.retries)
        except LLMError as error:
            out.answers += error.answers
            out.error = error
            return out
        out.answers += got

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
                if len(out.uses) >= max_calls:
                    content = LIMIT_REACHED
                else:
                    result = tools.call(ctx, asked.name, asked.arguments)
                    out.uses.append(
                        ToolUse(
                            role=role, step=first_step + len(out.uses), round=round_,
                            tool=asked.name, arguments=result.arguments, refs=result.refs,
                            notes=result.notes, error=result.error, items=result.items,
                        )
                    )  # fmt: skip
                    known.update(result.refs)
                    content = result.content
                messages.append(Message(role="tool", content=content, tool_call_id=asked.id))
            continue

        if not known:  # nothing to analyse: written by code, as in v1
            out.no_data = True
            return out
        text = response.text.strip()
        problem = report_problem(text, known)
        if problem is None:
            out.text = text
            return out
        if fixes >= config.retries:
            out.error = LLMError(f"no acceptable report after {fixes + 1} tries: {problem}")
            return out
        fixes += 1
        closed = True
        messages += [
            Message(role="assistant", content=text),
            Message(
                role="user",
                content=f"{problem}. Rewrite it, citing in square brackets only "
                "ids the tools returned.",
            ),
        ]


def _opening(config: PipelineConfig, role: str, ctx: tools.ToolContext, extra: str = ""):
    return [
        Message(role="system", content=config.prompts[role].strip()),
        Message(role="user", content=tools.describe(ctx) + extra),
    ]


def tool_analyst(
    state: PipelineState,
    llm: CachedLLM,
    config: PipelineConfig,
    role: str,
    ctx: tools.ToolContext,
) -> dict:
    meta, key = state["meta"], ANALYSTS[role]
    model = config.models[role]
    talk = converse(
        llm, config, role, ctx, _opening(config, role, ctx), ctx.rules.max_calls, meta.repeat
    )
    if talk.error is not None:
        talk.error.answers = talk.answers
        return {**_failed(meta, role, talk.error, model), "retrieved": talk.uses}
    report = Report(text=NO_DATA[role], no_data=True) if talk.no_data else Report(text=talk.text)
    return {
        "reports": {key: report},
        "retrieved": talk.uses,
        "usage": _calls(role, model, talk.answers),
    }


def answer_followups(
    state: PipelineState,
    llm: CachedLLM,
    config: PipelineConfig,
    context: Callable[[], tools.ToolContext],
) -> dict:
    """Each pending question, answered by its analyst in a short tool loop of its own."""
    meta, rules = state["meta"], config.tools
    round_ = max((f.round for f in state.get("followups", [])), default=0) + 1
    answered, uses, usage = [], [], []
    for question in state.get("pending", []):
        role = ROLE_OF[question.analyst]
        ctx = context()
        before = [u for u in state.get("retrieved", []) if u.role == role]
        report = state["reports"].get(question.analyst)
        extra = (
            f"\n\nYour report:\n{report.text if report else '(none)'}\n\n"
            f"The research manager asks: {question.question}\n\n"
            f"Use your tools if you need more data (at most {rules.followup_calls} more calls), "
            "then answer in a short addendum that cites reference ids in square brackets."
        )
        talk = converse(
            llm, config, role, ctx, _opening(config, role, ctx, extra), rules.followup_calls,
            meta.repeat, known={r for u in before for r in u.refs},
            first_step=len(before) + 1, round_=round_,
        )  # fmt: skip
        if talk.error is not None:
            answer, ok = f"The analyst could not answer: {talk.error}", False
        elif talk.no_data:
            answer, ok = "No data was available to answer.", False
        else:
            answer, ok = talk.text, True
        answered.append(
            FollowUp(
                round=round_, analyst=question.analyst, question=question.question,
                answer=answer, answered=ok,
            )
        )  # fmt: skip
        uses += talk.uses
        usage += _calls(role, config.models[role], talk.answers)
    return {"followups": answered, "pending": [], "retrieved": uses, "usage": usage}
