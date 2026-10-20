"""The analysts' tools: the data v1 prefetches, fetched on request (task 9.3).

docs/specs/08-agent-harness.md. Point in time is kept by construction: a
`ToolContext` holds the ticker and the decision time, set by code for the run;
no tool takes a date or a ticker, and an argument a tool does not have is
refused, so a model asking for a date is told so rather than silently given
today's data. Every tool reads the stores through the same cut-offs as the
prefetch (`NewsStore.query(before=...)`, `PriceStore.history(through=...)`),
and shows data in the same words as the v1 prompts, so that the one thing that
differs from v1 is who decides what to look at.

A result is plain text for the model plus everything code needs: the
reference ids it holds (the only ones the analyst may cite), notes on
arguments that were cut to the limits, and the items themselves for the state.
Text is built in a fixed order with rounded values, so the same call on the
same run gives the same text and a rerun is answered from the cache.
"""

import json
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sluice.agents.config import PipelineConfig, ToolsConfig
from sluice.agents.llm import ToolSpec
from sluice.agents.prefetch import (
    VALUE_DIGITS,
    _news_input,
    prefetch_fundamentals,
    prefetch_prices,
    select_news,
)
from sluice.agents.roles import _eastern, _figure_line, _news_line, _price_line
from sluice.agents.state import FundamentalInput, Gap, NewsInput, PriceInput
from sluice.fundamentals.store import FundamentalsStore
from sluice.market_calendar import last_closed_session
from sluice.news.models import StoredNews
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore

VOLUME_WINDOW = 20  # days behind each day's volume ratio, as for the indicator


class PriceDay(BaseModel):
    """One day of get_price_history: its return and volume ratio, never the price level."""

    ref: str  # "P:day_" + the date
    day: date
    ret: float  # close over the previous close, minus 1, on adjusted prices
    volume_ratio: float | None  # volume over the average of the 20 days before; None if short


Item = PriceInput | PriceDay | NewsInput | FundamentalInput


@dataclass(frozen=True)
class ToolContext:
    """What one run's tools may read. Set by code; the model never sees or sets it."""

    ticker: str
    decision_time: datetime
    news: NewsStore
    prices: PriceStore
    statements: FundamentalsStore
    config: PipelineConfig

    @property
    def rules(self) -> ToolsConfig:
        return self.config.tools or ToolsConfig()


@dataclass(frozen=True)
class ToolResult:
    name: str
    arguments: dict  # as used, after any cut to the limits; {} when they could not be read
    content: str  # what the model is sent
    refs: list[str]  # ids in `content`, in order
    items: list[Item]
    notes: list[str]  # arguments cut to a limit
    error: str | None = None  # the call was refused; it still counts as one


class _NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _History(BaseModel):
    model_config = ConfigDict(extra="forbid")
    days: int = Field(default=20, ge=1)


class _Search(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = ""
    days: int | None = Field(default=None, ge=1)
    limit: int = Field(default=10, ge=1)


class _Fundamentals(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quarters: int = Field(default=4, ge=1)


def _gaps(gaps: list[Gap]) -> list[str]:
    return [f"Missing: {gap.reason}." for gap in gaps]


def _cut(value: int, limit: int, name: str, notes: list[str]) -> int:
    if value > limit:
        notes.append(f"{name} {value} is above the limit; {limit} was used.")
        return limit
    return value


def _indicators(ctx: ToolContext, args: _NoArguments, notes: list[str]):
    shown, as_of, gaps = prefetch_prices(
        ctx.prices, ctx.ticker, ctx.decision_time, ctx.config.prices
    )
    head = [f"Indicators as of the close of {as_of}:"] if shown else []
    return {}, [*head, *(_price_line(item) for item in shown), *_gaps(gaps)], shown


def _history(ctx: ToolContext, args: _History, notes: list[str]):
    days = _cut(args.days, ctx.rules.history_max_days, "days", notes)
    through = last_closed_session(ctx.decision_time)
    bars = ctx.prices.history(ctx.ticker, through, days + VOLUME_WINDOW + 1)
    rows: list[PriceDay] = []
    for i in range(max(1, len(bars) - days), len(bars)):
        before = bars[max(0, i - VOLUME_WINDOW) : i]
        average = statistics.fmean(b.volume for b in before) if len(before) == VOLUME_WINDOW else 0
        bar = bars[i]
        rows.append(
            PriceDay(
                ref=f"P:day_{bar.day}",
                day=bar.day,
                ret=round(bar.close / bars[i - 1].close - 1, VALUE_DIGITS),
                volume_ratio=round(bar.volume / average, VALUE_DIGITS) if average else None,
            )
        )
    if not rows:
        return {"days": days}, ["Missing: no price history."], []
    if len(rows) < days:
        notes.append(f"only {len(rows)} trading days of history are available.")
    lines = [f"Daily returns, oldest first, through the close of {rows[-1].day}:"]
    for row in rows:
        ratio = "-" if row.volume_ratio is None else f"{row.volume_ratio:.2f}x"
        lines.append(f"[{row.ref}] return {row.ret:+.2%}, volume {ratio} its 20-day average")
    return {"days": days}, lines, rows


_WORD = re.compile(r"[a-z0-9]+")


def _terms(query: str) -> list[str]:
    return sorted(set(_WORD.findall(query.lower())))


def _matches(item: StoredNews, terms: list[str]) -> int:
    words = set(_WORD.findall(f"{item.title} {item.summary or ''}".lower()))
    return sum(term in words for term in terms)


def _search(ctx: ToolContext, args: _Search, notes: list[str]):
    window = ctx.config.news.lookback_days
    days = _cut(args.days or window, window, "days", notes)
    limit = _cut(args.limit, ctx.rules.search_max_limit, "limit", notes)
    since = ctx.decision_time - timedelta(days=days)
    found = ctx.news.query(ctx.ticker, before=ctx.decision_time, since=since)
    terms = _terms(args.query)
    if terms:
        hits = [(n, item) for item in found if (n := _matches(item, terms))]
        ranked = sorted(
            hits, key=lambda h: (h[0], h[1].reprint_count, h[1].published_at, h[1].id), reverse=True
        )
        chosen = [item for _, item in ranked[:limit]]
    else:  # no query: v1's choice, most reprinted then most recent, shown oldest first
        chosen = select_news(found, limit)
    used = {"query": args.query, "days": days, "limit": limit}
    items = [_news_input(item) for item in chosen]
    if not items:
        return used, [f"No news items matched in the {days} days before the decision time."], []
    head = f"{len(items)} news items from the {days} days before the decision time:"
    return used, [head, *(_news_line(item) for item in items)], items


def _fundamentals(ctx: ToolContext, args: _Fundamentals, notes: list[str]):
    limit = ctx.config.fundamentals.quarters
    quarters = _cut(args.quarters, limit, "quarters", notes)
    config = ctx.config.fundamentals.model_copy(update={"quarters": quarters})
    shown, gaps = prefetch_fundamentals(
        ctx.statements, ctx.prices, ctx.ticker, ctx.decision_time, config
    )
    head = ["Figures from the published financial statements:"] if shown else []
    return {"quarters": quarters}, [*head, *(_figure_line(i) for i in shown), *_gaps(gaps)], shown


def _schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "additionalProperties": False}


@dataclass(frozen=True)
class _Tool:
    spec: ToolSpec
    arguments: type[BaseModel]
    run: Callable


def _tools(rules: ToolsConfig, news_days: int, quarters: int) -> dict[str, _Tool]:
    integer = {"type": "integer", "minimum": 1}
    return {
        "get_indicators": _Tool(
            ToolSpec(
                name="get_indicators",
                description="Price indicators of the stock on the last close before the decision "
                "time: returns, returns against the S&P 500 ETF, volatility, moving averages, "
                "RSI and volume.",
                parameters=_schema({}),
            ),
            _NoArguments,
            _indicators,
        ),
        "get_price_history": _Tool(
            ToolSpec(
                name="get_price_history",
                description="The stock's daily returns and volume ratios for the last `days` "
                f"trading days before the decision time (at most {rules.history_max_days}).",
                parameters=_schema({"days": integer}),
            ),
            _History,
            _history,
        ),
        "search_news": _Tool(
            ToolSpec(
                name="search_news",
                description="News items about the stock published before the decision time, "
                f"within the last `days` days (at most {news_days}), one per reprint group. "
                "Items matching more words of `query` come first; an empty query gives the most "
                f"reprinted items. At most {rules.search_max_limit} items per call.",
                parameters=_schema(
                    {"query": {"type": "string"}, "days": integer, "limit": integer}
                ),
            ),
            _Search,
            _search,
        ),
        "get_fundamentals": _Tool(
            ToolSpec(
                name="get_fundamentals",
                description="Figures from the financial statements published by the decision "
                f"time, for the latest `quarters` quarters (at most {quarters}), with valuation.",
                parameters=_schema({"quarters": integer}),
            ),
            _Fundamentals,
            _fundamentals,
        ),
    }


def specs(config: PipelineConfig, names: list[str] | None = None) -> tuple[ToolSpec, ...]:
    """The tool specs a model is offered, in a fixed order."""
    tools = _tools(
        config.tools or ToolsConfig(), config.news.lookback_days, config.fundamentals.quarters
    )
    return tuple(tools[name].spec for name in names or tools)


def call(ctx: ToolContext, name: str, arguments: str) -> ToolResult:
    """Run one call the model asked for. Never raises on the model's mistakes: it says them."""
    tools = _tools(ctx.rules, ctx.config.news.lookback_days, ctx.config.fundamentals.quarters)

    def refused(error: str) -> ToolResult:
        return ToolResult(name, {}, f"Error: {error}", [], [], [], error)

    tool = tools.get(name)
    if tool is None:
        return refused(f"there is no tool {name!r}; the tools are {', '.join(tools)}")
    try:
        raw = json.loads(arguments or "{}")
        parsed = tool.arguments.model_validate(raw)
    except json.JSONDecodeError as exc:
        return refused(f"the arguments are not JSON ({exc.msg})")
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors()
        )
        return refused(f"{problems}. The decision time and the stock are fixed for this run.")
    notes: list[str] = []
    used, lines, items = tool.run(ctx, parsed, notes)
    text = "\n".join([*lines, *(f"Note: {note}" for note in notes)])
    return ToolResult(name, used, text, [item.ref for item in items], list(items), notes)


def describe(ctx: ToolContext) -> str:
    """The run's fixed facts, for the analyst's first message."""
    return f"Stock: {ctx.ticker}\nDecision time: {_eastern(ctx.decision_time)}"
