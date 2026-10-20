"""Ten years of daily bars for the pattern study, kept apart from the pipeline's (task 10.2).

A store of its own (`data/local/patterns/prices.db`) so that 500 extra series
never touch what the pipeline and the evaluation read. Fetching is per ticker
through `PriceStore.history`, which only fetches a series that ends before the
last closed session; so a rerun skips what is current and an interrupted run
picks up where it stopped. A ticker that fails is noted and the rest go on.
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sluice.prices.store import PriceStore
from sluice.universe import canonical

DEFAULT_DB = Path("data/local/patterns/prices.db")
START = date(2016, 1, 1)
BENCHMARK = "SPY"  # excess returns are measured against it, as everywhere in the project
PAUSE = 0.5  # seconds between Yahoo requests


@dataclass
class FetchResult:
    current: int = 0  # already up to date: nothing asked
    fetched: int = 0
    empty: list[str] = field(default_factory=list)  # Yahoo had no bars
    failed: dict[str, str] = field(default_factory=dict)


def fetch_all(
    store: PriceStore,
    symbols: Iterable[str],
    through: date,
    sleep: Callable[[float], None] = time.sleep,
    on_step: Callable[[str], None] = lambda symbol: None,
) -> FetchResult:
    result = FetchResult()
    for symbol in dict.fromkeys(canonical(s) for s in [BENCHMARK, *symbols]):
        latest = store.latest(symbol)
        if latest is not None and latest >= through:
            result.current += 1
        else:
            try:
                bars = store.history(symbol, through, 1)
            except Exception as exc:  # noqa: BLE001 - one broken ticker must not stop the list
                result.failed[symbol] = f"{type(exc).__name__}: {exc}"[:200]
            else:
                if bars:
                    result.fetched += 1
                else:
                    result.empty.append(symbol)
            sleep(PAUSE)
        on_step(symbol)
    return result
