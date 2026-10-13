from datetime import date, datetime, timedelta

from sluice.agents import fundamentals, indicators
from sluice.agents.config import FundamentalsConfig, NewsConfig, PipelineConfig, PricesConfig
from sluice.agents.state import FundamentalInput, Gap, Inputs, NewsInput, PriceInput
from sluice.fundamentals.store import FundamentalsStore
from sluice.market_calendar import close_of, last_closed_session
from sluice.news.models import StoredNews
from sluice.news.store import NewsStore
from sluice.news.timeutil import ET
from sluice.prices.store import PriceStore
from sluice.universe import canonical

VALUE_DIGITS = 4  # indicators are rounded so that float noise cannot change a prompt


def backtest_decision_time(day: date) -> datetime:
    """The close of trading day `day`: 16:00 Eastern, or the early close.

    Trades and labels start at the next open.
    """
    close = close_of(day)
    if close is None:
        raise ValueError(f"{day} is not a trading day")
    return close


def prefetch_prices(
    prices: PriceStore, ticker: str, decision_time: datetime, config: PricesConfig
) -> tuple[list[PriceInput], date | None, list[Gap]]:
    """Indicators on the last trading day that had closed by the decision time.

    A day's bar counts only once its session is over, so a backtest at the close
    of T uses T's bar and nothing later, and a run during trading hours uses the
    previous day's.
    """
    through = last_closed_session(decision_time)
    bars = prices.history(ticker, through, config.lookback_days)
    if not bars:
        return [], None, [Gap(kind="prices", reason="no price history")]

    gaps = []
    as_of = bars[-1].day
    if as_of < through:
        reason = f"prices end on {as_of}, before the last closed session {through}"
        gaps.append(Gap(kind="prices", reason=reason))
    benchmark = canonical(config.benchmark)
    reference = (
        None  # the benchmark has no excess return over itself
        if canonical(ticker) == benchmark
        else prices.history(benchmark, as_of, config.lookback_days)
    )
    values = indicators.compute(bars, reference)
    expected = [
        name for name in indicators.NAMES if reference is not None or not name.startswith("excess_")
    ]
    if missing := [name for name in expected if name not in values]:
        gaps.append(Gap(kind="prices", reason=f"could not compute: {', '.join(missing)}"))
    shown = [
        PriceInput(ref=f"P:{name}", name=name, value=round(values[name], VALUE_DIGITS))
        for name in indicators.NAMES
        if name in values
    ]
    return shown, as_of, gaps


def select_news(items: list[StoredNews], max_items: int) -> list[StoredNews]:
    """At most `max_items`: the most reprinted, then the most recent. Returned oldest first."""
    ranked = sorted(items, key=lambda item: (item.reprint_count, item.published_at), reverse=True)
    return sorted(ranked[:max_items], key=lambda item: (item.published_at, item.id))


def prefetch_news(
    store: NewsStore, ticker: str, decision_time: datetime, config: NewsConfig
) -> tuple[list[NewsInput], list[Gap]]:
    """News published before the decision time.

    The cut-off is applied by the store's query, which also counts reprints as
    of the decision time.
    """
    since = decision_time - timedelta(days=config.lookback_days)
    found = store.query(ticker, before=decision_time, since=since)
    if not found:
        reason = f"no items in the {config.lookback_days} days before the decision time"
        return [], [Gap(kind="news", reason=reason)]
    return [
        NewsInput(
            ref=f"N{item.id}",
            title=item.title,
            summary=item.summary,
            publisher=item.publisher,
            published_at=item.published_at,
            session=item.session,
            reprint_count=item.reprint_count,
        )
        for item in select_news(found, config.max_items)
    ], []


def prefetch_fundamentals(
    statements: FundamentalsStore,
    prices: PriceStore,
    ticker: str,
    decision_time: datetime,
    config: FundamentalsConfig,
) -> tuple[list[FundamentalInput], list[Gap]]:
    """Figures from the quarters that count as published on the decision day.

    Statements are only read from the store, never fetched here: the vendor's
    daily quota is too small to spend without being asked.
    """

    def gap(reason: str) -> Gap:
        return Gap(kind="fundamentals", reason=reason)

    stored = statements.quarters(ticker)
    if stored is None:
        return [], [gap("no statements stored for this ticker")]
    decision_day = decision_time.astimezone(ET).date()
    seen = fundamentals.visible(stored, decision_day, config.lag_days)
    if not seen:
        return [], [gap("no quarter counts as published yet")]

    last_bar = prices.history(ticker, last_closed_session(decision_time), limit=1)
    close = last_bar[0].raw_close if last_bar else None
    figures = fundamentals.compute(seen, close, config.quarters)
    units = dict(fundamentals.QUARTER_FIGURES)
    shown = []
    for name in fundamentals.names(config.quarters):
        if name not in figures:
            continue
        value, quarter = figures[name]
        unit = units.get(name.rsplit("_q", 1)[0], "multiple")
        shown.append(
            FundamentalInput(
                ref=f"F:{name}",
                name=name,
                value=value if unit == "amount" else round(value, VALUE_DIGITS),
                unit=unit,
                currency=quarter.currency if unit == "amount" else None,
                quarter_end=quarter.end,
                visible_from=fundamentals.visible_from(quarter, config.lag_days),
            )
        )

    gaps = []
    if len(seen) < config.quarters:
        gaps.append(gap(f"only {len(seen)} of {config.quarters} quarters are published"))
    currencies = {quarter.currency for quarter in seen[-4:]}
    if currencies != {fundamentals.PRICE_CURRENCY}:
        gaps.append(gap(f"no valuation: statements are in {', '.join(sorted(currencies))}"))
    if missing := [n for n in fundamentals.names(config.quarters) if n not in figures]:
        gaps.append(gap(f"could not compute: {', '.join(missing)}"))
    return shown, gaps


def prefetch(
    store: NewsStore,
    prices: PriceStore,
    statements: FundamentalsStore,
    ticker: str,
    decision_time: datetime,
    config: PipelineConfig,
) -> Inputs:
    """Everything the analysts will see, none of it from at or after the decision time."""
    indicators_shown, as_of, price_gaps = prefetch_prices(
        prices, ticker, decision_time, config.prices
    )
    news, news_gaps = prefetch_news(store, ticker, decision_time, config.news)
    figures, fundamentals_gaps = prefetch_fundamentals(
        statements, prices, ticker, decision_time, config.fundamentals
    )
    return Inputs(
        prices=indicators_shown,
        prices_as_of=as_of,
        news=news,
        fundamentals=figures,
        gaps=[*price_gaps, *news_gaps, *fundamentals_gaps],
    )
