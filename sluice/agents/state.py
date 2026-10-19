import operator
import re
from datetime import date, datetime
from typing import Annotated, Literal, TypedDict

from pydantic import AwareDatetime, BaseModel, Field

from sluice.news.models import Session
from sluice.news.signal import Record
from sluice.news.timeutil import to_utc

RunMode = Literal["backtest", "adhoc"]
Status = Literal["ok", "failed"]
Rating = Literal["buy", "overweight", "hold", "underweight", "sell"]
Side = Literal["bull", "bear"]
GapKind = Literal["prices", "news", "fundamentals"]
FigureUnit = Literal["amount", "ratio", "points", "multiple"]

# How a source is cited in a report: [N48213], [P:rsi14], [F:revenue_q], [S:news_signal]
# or several at once, separated by commas or semicolons: [N1, N2], [N1; N2].
_REF = re.compile(r"N\d+|[PFS]:\w+")
_BRACKETS = re.compile(r"\[([^\[\]]*)\]")
_SEPARATOR = re.compile(r"[,;]")
# An item that is meant as a reference id even when it is not a valid one: [S], [N12x], [S:].
_REF_LIKE = re.compile(r"N\d|[PFS]:|[NPFS]$")


class Meta(BaseModel):
    run_id: str
    ticker: str
    decision_time: AwareDatetime
    mode: RunMode
    config_version: str
    config_digest: str
    repeat: int = Field(ge=0)
    status: Status = "ok"
    failure: str | None = None


def _absent(value: object) -> bool:
    return value is None or value == []


class NewsInput(BaseModel):
    """One news item exactly as the news analyst saw it."""

    ref: str  # "N" + the article's id in the news store
    title: str
    summary: str | None
    publisher: str | None
    published_at: AwareDatetime
    session: Session | None
    reprint_count: int
    # Only in a weighted run (docs/specs/07-agent-pipeline.md, task 5.1). Left out of
    # the saved state when absent, so an unweighted run's state is as it always was.
    scored: bool | None = Field(default=None, exclude_if=_absent)  # False: Jev could not score it
    relevance: float | None = Field(default=None, exclude_if=_absent)
    event_type: str | None = Field(default=None, exclude_if=_absent)
    source_record: Record | None = Field(default=None, exclude_if=_absent)  # None: no weight table
    type_record: Record | None = Field(default=None, exclude_if=_absent)


class SignalInput(BaseModel):
    """A number computed by code from the news, exactly as the news analyst saw it."""

    ref: str  # "S:" + the signal's name
    value: float
    n_items: int  # scored items it was computed from
    weights_as_of: AwareDatetime  # the weight table it used


class PriceInput(BaseModel):
    """One price indicator exactly as the market analyst will see it."""

    ref: str  # "P:" + the indicator's name
    name: str
    value: float


class FundamentalInput(BaseModel):
    """One figure from the financial statements exactly as the fundamentals analyst saw it."""

    ref: str  # "F:" + the figure's name
    name: str
    value: float
    unit: FigureUnit
    currency: str | None  # of an amount
    quarter_end: date  # the quarter it belongs to; for a four-quarter figure, the latest
    visible_from: date  # the first day that quarter counted as published


class Gap(BaseModel):
    kind: GapKind
    reason: str


class Inputs(BaseModel):
    """Snapshot of everything pre-fetched for the run."""

    prices: list[PriceInput] = []
    prices_as_of: date | None = None  # the trading day the indicators are computed on
    news: list[NewsInput] = []
    fundamentals: list[FundamentalInput] = []
    gaps: list[Gap] = []
    signals: list[SignalInput] = Field(default=[], exclude_if=_absent)  # weighted runs only

    def refs(self) -> set[str]:
        items = (*self.prices, *self.news, *self.fundamentals, *self.signals)
        return {item.ref for item in items}


class Report(BaseModel):
    text: str
    no_data: bool = False  # written by code because there was nothing to analyse


class Turn(BaseModel):
    round: int = Field(ge=1)
    side: Side
    text: str


class Evidence(BaseModel):
    claim: str = Field(min_length=1)
    # What the claim says about the stock, whatever the rating.
    direction: Literal["bullish", "bearish"]
    refs: list[str] = Field(min_length=1)


class Verdict(BaseModel):
    """What the research manager must answer with."""

    rating: Rating
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class Decision(Verdict):
    notes: list[str] = []  # added by code: coverage gaps, out-of-universe ticker


class Call(BaseModel):
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool


def merge_meta(old: Meta | None, new: Meta) -> Meta:
    """A run that failed anywhere stays failed, also when parallel nodes report in one step."""
    if old is None or old.status == "ok":
        return new
    if new.status == "ok" or new.failure == old.failure:
        return old
    return old.model_copy(update={"failure": f"{old.failure}; {new.failure}"})


def merge_reports(old: dict[str, Report] | None, new: dict[str, Report]) -> dict[str, Report]:
    return {**(old or {}), **new}


class PipelineState(TypedDict, total=False):
    """The six channels of a run. Channels with a merge rule may be written by parallel nodes."""

    meta: Annotated[Meta, merge_meta]
    inputs: Inputs
    reports: Annotated[dict[str, Report], merge_reports]
    debate: Annotated[list[Turn], operator.add]
    decision: Decision | None
    usage: Annotated[list[Call], operator.add]


# Every class stored in a channel; the checkpointer needs the list to read them back.
STATE_MODELS = (
    Meta,
    PriceInput,
    NewsInput,
    SignalInput,
    FundamentalInput,
    Gap,
    Inputs,
    Report,
    Turn,
    Evidence,
    Decision,
    Call,
)


def make_run_id(
    ticker: str, decision_time: datetime, config_version: str, config_digest: str, repeat: int
) -> str:
    """The same run parameters always give the same id, so a re-run finds its checkpoint.

    The digest is part of it: a config file edited in place is a different run.
    """
    stamp = to_utc(decision_time).strftime("%Y%m%dT%H%MZ")
    return f"{ticker}_{stamp}_{config_version}-{config_digest}_r{repeat}"


def _bracketed(text: str) -> list[tuple[str, list[str]]]:
    """Every square-bracketed group in `text`, with its items."""
    return [
        (match.group(0), [item.strip() for item in _SEPARATOR.split(match.group(1))])
        for match in _BRACKETS.finditer(text)
    ]


def _is_citation(items: list[str]) -> bool:
    return all(_REF.fullmatch(item) for item in items)


def cited_refs(text: str) -> set[str]:
    """Reference ids cited in square brackets in a report."""
    return {item for _, items in _bracketed(text) if _is_citation(items) for item in items}


def malformed_citations(text: str) -> list[str]:
    """Bracketed groups that try to cite but are not all valid ids, such as [S] or [N1, see N2].

    Other brackets, such as [sic] or a word ending like remain[s], are left alone.
    """
    return [
        group
        for group, items in _bracketed(text)
        if not _is_citation(items) and any(_REF_LIKE.match(item) for item in items)
    ]


def dump_state(state: PipelineState) -> dict:
    """The full state as JSON-ready data, with every channel present."""
    decision = state.get("decision")
    return {
        "meta": state["meta"].model_dump(mode="json"),
        "inputs": state.get("inputs", Inputs()).model_dump(mode="json"),
        "reports": {k: v.model_dump(mode="json") for k, v in state.get("reports", {}).items()},
        "debate": [turn.model_dump(mode="json") for turn in state.get("debate", [])],
        "decision": decision.model_dump(mode="json") if decision else None,
        "usage": [call.model_dump(mode="json") for call in state.get("usage", [])],
    }
