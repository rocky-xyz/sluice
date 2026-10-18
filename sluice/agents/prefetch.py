from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sluice.agents import fundamentals, indicators
from sluice.agents.config import FundamentalsConfig, NewsConfig, PipelineConfig, PricesConfig
from sluice.agents.state import (
    FundamentalInput,
    Gap,
    Inputs,
    NewsInput,
    PriceInput,
    SignalInput,
)
from sluice.fundamentals.store import FundamentalsStore
from sluice.market_calendar import close_of, last_closed_session
from sluice.news.models import StoredNews
from sluice.news.signal import SignalItem, record, ticker_day_signal
from sluice.news.store import NewsStore
from sluice.news.timeutil import ET
from sluice.news.weight_tables import StoredTable, latest_before
from sluice.news.weights import stance_score
from sluice.prices.store import PriceStore
from sluice.scoring.client import CachedJev, JevError
from sluice.scoring.config import ScoringConfig
from sluice.scoring.score import HISTORY_POOL, score_item
from sluice.scoring.store import ItemScores, ScoreStore
from sluice.universe import canonical

VALUE_DIGITS = 4  # indicators are rounded so that float noise cannot change a prompt
SIGNAL_DIGITS = 8  # the news signal is of the order of 1e-4
NEWS_SIGNAL = "S:news_signal"
UNKNOWN_SOURCE = "(unknown)"  # how the weight tables name a missing publisher


@dataclass(frozen=True)
class Weighting:
    """What a weighted run reads besides the news store (docs/specs/07, task 5.1)."""

    scores: ScoreStore
    tables: list[StoredTable]  # every saved weight table of `scoring.version`
    scoring: ScoringConfig
    jev: CachedJev  # asked only about items that have no score yet


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


def _window(
    store: NewsStore, ticker: str, decision_time: datetime, config: NewsConfig
) -> tuple[list[StoredNews], list[Gap]]:
    """One item per reprint group published in the lookback before the decision time.

    The cut-off is applied by the store's query, which also counts reprints as
    of the decision time.
    """
    since = decision_time - timedelta(days=config.lookback_days)
    found = store.query(ticker, before=decision_time, since=since)
    if not found:
        reason = f"no items in the {config.lookback_days} days before the decision time"
        return [], [Gap(kind="news", reason=reason)]
    return found, []


def _news_input(item: StoredNews, **weighted: object) -> NewsInput:
    return NewsInput(
        ref=f"N{item.id}",
        title=item.title,
        summary=item.summary,
        publisher=item.publisher,
        published_at=item.published_at,
        session=item.session,
        reprint_count=item.reprint_count,
        **weighted,
    )


def prefetch_news(
    store: NewsStore, ticker: str, decision_time: datetime, config: NewsConfig
) -> tuple[list[NewsInput], list[Gap]]:
    """News published before the decision time, as chosen before weights existed."""
    found, gaps = _window(store, ticker, decision_time, config)
    return [_news_input(item) for item in select_news(found, config.max_items)], gaps


def _scores(
    items: list[StoredNews], store: NewsStore, ticker: str, weighting: Weighting
) -> tuple[dict[int, ItemScores], int]:
    """Each item's Jev scores, scoring now what was never scored. Also: how many failed.

    An item scored now is compared with the titles of the ticker's items
    published before it, as `sluice score` does, and its scores are stored. An
    item Jev could not score is left without scores and is tried again next
    time; one already skipped for good (`ScoreStore.done` without scores) is
    not asked again.
    """
    version = weighting.scoring.version
    found: dict[int, ItemScores] = {}
    failed = 0
    for item in items:
        if weighting.scores.done(item.id, version):
            if (scores := weighting.scores.get(item.id, version)) is not None:
                found[item.id] = scores
            continue
        earlier = store.query(ticker, before=item.published_at, limit=HISTORY_POOL)
        history = [earlier_item.title for earlier_item in reversed(earlier)]
        try:
            scores, skip_reason = score_item(item, history, weighting.scoring, weighting.jev)
        except JevError:
            failed += 1
            continue
        digest = weighting.scoring.digest
        if scores is None:
            weighting.scores.skip(item.id, version, digest, skip_reason or "")
            failed += 1
            continue
        weighting.scores.put(item.id, version, digest, scores)
        found[item.id] = scores
    return found, failed


def prefetch_weighted_news(
    store: NewsStore,
    ticker: str,
    decision_time: datetime,
    config: NewsConfig,
    weighting: Weighting,
) -> tuple[list[NewsInput], list[SignalInput], list[Gap]]:
    """News filtered by Jev relevance and labelled by learned weights, plus the news signal.

    docs/specs/07-agent-pipeline.md, task 5.1. The weight table is the latest
    fitted as of a month start not after the decision time. The signal is
    computed over every scored item in the window, before the relevance filter
    and the cap; the items shown are those at or above `min_relevance` (and
    those Jev could not score), chosen by the unweighted rule.
    """
    found, gaps = _window(store, ticker, decision_time, config)
    if not found:
        return [], [], gaps
    stored = latest_before(weighting.tables, decision_time)
    if stored is None:
        gaps.append(Gap(kind="news", reason="no weight table fitted by the decision time"))
    scores, failed = _scores(found, store, ticker, weighting)
    if failed:
        gaps.append(Gap(kind="news", reason=f"{failed} items could not be scored"))

    signals = []
    if stored is not None:
        signal_items = [
            SignalItem(
                raw=stance_score(item_scores.stance_probabilities),
                source=item.publisher or UNKNOWN_SOURCE,
                event_type=item_scores.event_type,
                relevance=item_scores.relevance,
            )
            for item in found
            if (item_scores := scores.get(item.id)) is not None
        ]
        value = ticker_day_signal(signal_items, stored.table)
        if value is None:
            reason = "no news signal: no scored item has any relevance"
            gaps.append(Gap(kind="news", reason=f"{reason} (weights {stored.as_of:%Y-%m-%d})"))
        else:
            signals.append(
                SignalInput(
                    ref=NEWS_SIGNAL,
                    value=round(value, SIGNAL_DIGITS),
                    n_items=len(signal_items),
                    weights_as_of=stored.as_of,
                )
            )

    kept = [
        item
        for item in found
        if item.id not in scores or scores[item.id].relevance >= config.min_relevance
    ]
    if not kept:
        reason = f"no item reached relevance {config.min_relevance}"
        gaps.append(Gap(kind="news", reason=reason))

    def shown(item: StoredNews) -> NewsInput:
        item_scores = scores.get(item.id)
        if item_scores is None:
            return _news_input(item, scored=False)
        labels = {}
        if stored is not None:
            source = item.publisher or UNKNOWN_SOURCE
            labels = {
                "source_record": record(stored.table.source.get(source)),
                "type_record": record(stored.table.event_type.get(item_scores.event_type)),
            }
        return _news_input(
            item,
            scored=True,
            relevance=item_scores.relevance,
            event_type=item_scores.event_type,
            **labels,
        )

    return [shown(item) for item in select_news(kept, config.max_items)], signals, gaps


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
    weighting: Weighting | None = None,
) -> Inputs:
    """Everything the analysts will see, none of it from at or after the decision time.

    `weighting` is needed exactly when the config asks for weighted news.
    """
    if config.news.weighted and weighting is None:
        raise ValueError(f"config {config.version} weights news but no weighting was given")
    indicators_shown, as_of, price_gaps = prefetch_prices(
        prices, ticker, decision_time, config.prices
    )
    signals: list[SignalInput] = []
    if weighting is not None and config.news.weighted:
        news, signals, news_gaps = prefetch_weighted_news(
            store, ticker, decision_time, config.news, weighting
        )
    else:
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
        signals=signals,
    )
