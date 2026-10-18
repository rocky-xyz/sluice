"""Weighted news signal: per-item and per-ticker-day aggregation (task 4.5).

ŷ_i applies a fitted `WeightTable`'s learned source/event-type weights to one
item's own stance score, per docs/specs/03-news-weights.md ("已确认", tasks
4.1/4.3/4.5). A ticker-day's signal is the *relevance*-weighted average of its
items' ŷ_i — not a plain average, and not weighted by the learned weight
itself (that would risk a near-zero or sign-flipping denominator now that some
learned weights are negative in practice, e.g. PR Newswire/Business Wire in
the task-4.3 dry run). It is `None` when every item that day has zero
relevance: no confident aggregate to report, not a confident zero.

Items are grouped by the trading day they become actionable on — the "next
open" entry day already used for labels (`sluice.news.labels`) — not by the
raw calendar date of `published_at`, so Friday-after-close and Monday-premarket
news about the same stock land in the bucket an analyst deciding on Monday
would actually see them in.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from sluice.market_calendar import next_open_after
from sluice.news.store import NewsStore
from sluice.news.weights import WeightEstimate, WeightTable, stance_score
from sluice.scoring.store import ScoreStore

# What a source's or event type's learned weight says, in words an analyst can use
# (docs/specs/07-agent-pipeline.md, "已确认（任务 5.1）"). The weight multiplies the
# item's stance score, so its sign says whether the stance came true, not
# whether the news was good or bad.
Record = Literal["confirmed", "reversed", "no reliable record", "no record"]


def record(estimate: WeightEstimate | None) -> Record:
    """Confirmed / reversed only when the whole bootstrap interval is above / below 0."""
    if estimate is None:
        return "no record"  # the table never saw this source or type
    if estimate.low > 0:
        return "confirmed"
    if estimate.high < 0:
        return "reversed"
    return "no reliable record"


def item_signal(raw: float, source: str, event_type: str, weights: WeightTable) -> float:
    """ŷ_i = raw_i · (β_source[source] + β_type[event_type]); 0 for a group the table never saw."""
    source_weight = weights.source[source].weight if source in weights.source else 0.0
    type_weight = weights.event_type[event_type].weight if event_type in weights.event_type else 0.0
    return raw * (source_weight + type_weight)


@dataclass(frozen=True)
class SignalItem:
    raw: float
    source: str
    event_type: str
    relevance: float


def ticker_day_signal(items: list[SignalItem], weights: WeightTable) -> float | None:
    """Relevance-weighted average of each item's ŷ_i. `None` when every relevance is 0."""
    numerator = denominator = 0.0
    for item in items:
        numerator += item.relevance * item_signal(item.raw, item.source, item.event_type, weights)
        denominator += item.relevance
    return numerator / denominator if denominator else None


def group_by_ticker_day(
    news: NewsStore,
    scores: ScoreStore,
    companies: list[str],
    as_of: datetime,
    config_version: str = "v1",
) -> dict[tuple[str, date], list[SignalItem]]:
    """Scored items published before `as_of`, grouped by (ticker, entry day).

    A news item tagged to several tickers contributes to each of them
    independently — unlike task 4.3's fitting rows (one averaged observation
    per item), these are separate per-ticker signals, not one shared
    statistical observation.
    """
    groups: dict[tuple[str, date], list[SignalItem]] = defaultdict(list)
    for news_id, score in scores.iter_all(config_version):
        item = news.get(news_id)
        if item is None or item.published_at is None or item.published_at >= as_of:
            continue
        entry_day = next_open_after(item.published_at)
        signal_item = SignalItem(
            raw=stance_score(score.stance_probabilities),
            source=item.publisher or "(unknown)",
            event_type=score.event_type,
            relevance=score.relevance,
        )
        for ticker in sorted(set(item.tickers) & set(companies)):
            groups[(ticker, entry_day)].append(signal_item)
    return groups


def ticker_day_signals(
    news: NewsStore,
    scores: ScoreStore,
    companies: list[str],
    weights: WeightTable,
    as_of: datetime,
    config_version: str = "v1",
) -> dict[tuple[str, date], float | None]:
    groups = group_by_ticker_day(news, scores, companies, as_of, config_version)
    return {key: ticker_day_signal(items, weights) for key, items in groups.items()}
