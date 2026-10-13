from datetime import date, datetime, timedelta

from sluice.agents.config import NewsConfig
from sluice.agents.state import Gap, Inputs, NewsInput
from sluice.market_calendar import close_of
from sluice.news.models import StoredNews
from sluice.news.store import NewsStore


def backtest_decision_time(day: date) -> datetime:
    """The close of trading day `day`: 16:00 Eastern, or the early close.

    Trades and labels start at the next open.
    """
    close = close_of(day)
    if close is None:
        raise ValueError(f"{day} is not a trading day")
    return close


def select_news(items: list[StoredNews], max_items: int) -> list[StoredNews]:
    """At most `max_items`: the most reprinted, then the most recent. Returned oldest first."""
    ranked = sorted(items, key=lambda item: (item.reprint_count, item.published_at), reverse=True)
    return sorted(ranked[:max_items], key=lambda item: (item.published_at, item.id))


def prefetch(
    store: NewsStore, ticker: str, decision_time: datetime, news: NewsConfig
) -> Inputs:
    """Everything the analysts will see, none of it published at or after the decision time.

    The cut-off is applied by the store's query, which also counts reprints as
    of the decision time.
    """
    since = decision_time - timedelta(days=news.lookback_days)
    found = store.query(ticker, before=decision_time, since=since)
    if not found:
        reason = f"no items in the {news.lookback_days} days before the decision time"
        return Inputs(gaps=[Gap(kind="news", reason=reason)])
    return Inputs(
        news=[
            NewsInput(
                ref=f"N{item.id}",
                title=item.title,
                summary=item.summary,
                publisher=item.publisher,
                published_at=item.published_at,
                session=item.session,
                reprint_count=item.reprint_count,
            )
            for item in select_news(found, news.max_items)
        ]
    )
