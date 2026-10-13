import operator
import re
from datetime import date, datetime
from typing import Annotated, Literal, TypedDict

from pydantic import AwareDatetime, BaseModel, Field

from sluice.news.models import Session
from sluice.news.timeutil import to_utc

RunMode = Literal["backtest", "adhoc"]
Status = Literal["ok", "failed"]
Rating = Literal["buy", "overweight", "hold", "underweight", "sell"]
Side = Literal["bull", "bear"]
GapKind = Literal["prices", "news", "fundamentals"]
FigureUnit = Literal["amount", "ratio", "points", "multiple"]

# How a source is cited in a report: [N48213], [P:rsi14], [F:revenue_q] or [N1, N2].
_REF = r"(?:N\d+|[PF]:\w+)"
_CITATION = re.compile(rf"\[({_REF}(?:\s*,\s*{_REF})*)\]")


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


class NewsInput(BaseModel):
    """One news item exactly as the news analyst saw it."""

    ref: str  # "N" + the article's id in the news store
    title: str
    summary: str | None
    publisher: str | None
    published_at: AwareDatetime
    session: Session | None
    reprint_count: int


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

    def refs(self) -> set[str]:
        return {item.ref for item in (*self.prices, *self.news, *self.fundamentals)}


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


def cited_refs(text: str) -> set[str]:
    """Reference ids cited in square brackets in a report."""
    return {ref.strip() for group in _CITATION.findall(text) for ref in group.split(",")}


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
