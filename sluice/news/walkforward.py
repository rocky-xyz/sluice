"""Walk-forward monthly refitting of the source/event-type weights (task 4.4).

Per docs/specs/03-news-weights.md ("已确认", task 4.1): an expanding window,
refit on the first of every calendar month. Each month's fit is just task
4.3's `fit_weights`/`assemble_rows` run as of that month's start, so it only
ever sees news whose label had resolved by then; what is new here is the
monthly schedule, and keeping every month's table rather than only the
latest one.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from sluice.news.store import NewsStore
from sluice.news.weights import WeightTable, assemble_rows, fit_weights
from sluice.prices.store import PriceStore
from sluice.scoring.store import ScoreStore

# Below this many resolved rows, a month's fit is noise rather than signal
# about the sources/types it happened to see; keep the cutoff but drop the table.
MIN_ROWS = 500


def monthly_cutoffs(start: datetime, end: datetime) -> list[datetime]:
    """UTC midnight on the first of every calendar month from `start`'s month
    through `end`'s month, inclusive of both.
    """
    cutoffs = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        cutoffs.append(datetime(year, month, 1, tzinfo=UTC))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return cutoffs


@dataclass(frozen=True)
class Refit:
    as_of: datetime
    n_rows: int
    table: WeightTable | None  # None below `min_rows`


def walk_forward_fit(
    news: NewsStore,
    scores: ScoreStore,
    prices: PriceStore,
    companies: list[str],
    cutoffs: list[datetime],
    config_version: str = "v1",
    min_rows: int = MIN_ROWS,
    seed: int = 0,
) -> list[Refit]:
    """One `fit_weights` per cutoff, each seeing only rows resolved by then."""
    refits = []
    for as_of in cutoffs:
        rows = assemble_rows(news, scores, prices, companies, as_of, config_version)
        table = fit_weights(rows, seed=seed) if len(rows) >= min_rows else None
        refits.append(Refit(as_of=as_of, n_rows=len(rows), table=table))
    return refits
