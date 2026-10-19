"""Target weights for each decision day, from what the stores hold (task 6.11).

docs/specs/04-sizing.md: step 1 gathers the three inputs, the stocks'
volatilities and the exposure, all as of the decision time, without asking any
model or Jev; `formula.target_weights` does the rest. What does not depend on
a pipeline config (news scores, volatilities, exposure) is worked out once per
decision day and shared by every config and repeat.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sluice.agents.prefetch import backtest_decision_time
from sluice.evaluation.decisions import Decision
from sluice.market_calendar import last_closed_session
from sluice.news.signal import signal_items, ticker_day_signal
from sluice.news.store import NewsStore
from sluice.news.weight_tables import StoredTable, latest_before
from sluice.prices.store import PriceStore
from sluice.scoring.reports import ReportScoreStore, report_score
from sluice.scoring.store import ScoreStore
from sluice.sizing.config import SizingConfig
from sluice.sizing.formula import Inputs, target_weights
from sluice.sizing.regime import exposure, realised_vol

RATING_SCORES = {"buy": 2.0, "overweight": 1.0, "hold": 0.0, "underweight": -1.0, "sell": -2.0}
Schedule = dict[date, dict[str, float]]  # rebalance day -> target weights


@dataclass(frozen=True)
class Sources:
    news: NewsStore
    scores: ScoreStore
    tables: list[StoredTable]  # weight tables of the config's news scoring
    reports: ReportScoreStore
    prices: PriceStore


@dataclass(frozen=True)
class Market:
    """What one decision day looks like before any rating is read."""

    exposure: float
    vols: dict[str, float]
    news: dict[str, float | None]


def stock_vol(prices: PriceStore, ticker: str, decision_time: datetime, lookback: int) -> float:
    through = last_closed_session(decision_time)
    bars = prices.history(ticker, through, lookback + 1)
    if len(bars) < lookback + 1:
        raise ValueError(f"{ticker}: {len(bars)} bars up to {through}, {lookback + 1} needed")
    return realised_vol([bar.close for bar in bars])


def news_score(
    src: Sources, ticker: str, decision_time: datetime, config: SizingConfig
) -> float | None:
    """The news signal over the window, from scored items only; None without a table or items."""
    stored = latest_before(src.tables, decision_time)
    if stored is None:
        return None
    since = decision_time - timedelta(days=config.inputs.news_lookback_days)
    found = src.news.query(ticker, before=decision_time, since=since)
    version = config.inputs.scoring_config
    scores = {item.id: s for item in found if (s := src.scores.get(item.id, version)) is not None}
    return ticker_day_signal(signal_items(found, scores), stored.table)


def market(src: Sources, tickers: list[str], day: date, config: SizingConfig) -> Market:
    decision_time = backtest_decision_time(day)
    return Market(
        exposure=exposure(src.prices, decision_time, config.exposure).value,
        vols={
            t: stock_vol(src.prices, t, decision_time, config.stocks.vol_lookback)
            for t in tickers
        },
        news={t: news_score(src, t, decision_time, config) for t in tickers},
    )


def _weights(inputs: Inputs, m: Market, config: SizingConfig, **override) -> dict[str, float]:
    return target_weights(
        inputs,
        m.vols,
        m.exposure,
        override.get("k", config.k),
        config.coefficients,
        config.stocks.max_weight,
        scale_by_vol=override.get("scale_by_vol", config.stocks.vol_scaling),
    )


def ablations(
    markets: dict[date, Market], rebalances: dict[date, date], config: SizingConfig
) -> dict[str, Schedule]:
    """The two controls that read no rating: exposure alone, and volatility scaling with it."""
    out: dict[str, Schedule] = {"只做总仓位（等权 × E）": {}, "波动率缩放 + 总仓位（k = 0）": {}}
    for day, m in markets.items():
        none = dict.fromkeys(m.vols)
        blank = Inputs(rating=none, reports=none, news=none)
        out["只做总仓位（等权 × E）"][rebalances[day]] = _weights(
            blank, m, config, k=0.0, scale_by_vol=False
        )
        out["波动率缩放 + 总仓位（k = 0）"][rebalances[day]] = _weights(blank, m, config, k=0.0)
    return out


def formula_schedules(
    decisions: list[Decision],
    markets: dict[date, Market],
    rebalances: dict[date, date],
    src: Sources,
    config: SizingConfig,
) -> dict[int, tuple[Schedule, int]]:
    """Per repeat: the full formula's targets, and how many ratings were missing."""
    by_repeat: dict[int, dict[tuple[date, str], Decision]] = defaultdict(dict)
    for decision in decisions:
        by_repeat[decision.repeat][(decision.day, decision.ticker)] = decision
    out = {}
    for repeat, cells in sorted(by_repeat.items()):
        schedule: Schedule = {}
        missing = 0
        for day, m in markets.items():
            rating: dict[str, float | None] = {}
            reports: dict[str, float | None] = {}
            for ticker in m.vols:
                decision = cells.get((day, ticker))
                has_rating = decision is not None and decision.rating is not None
                rating[ticker] = RATING_SCORES[decision.rating] if has_rating else None
                missing += not has_rating
                reports[ticker] = (
                    report_score(src.reports, decision.run_id, config.inputs.report_spec)
                    if decision is not None and decision.run_id
                    else None
                )
            schedule[rebalances[day]] = _weights(Inputs(rating, reports, m.news), m, config)
        out[repeat] = (schedule, missing)
    return out
