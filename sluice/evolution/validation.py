"""The gatekeeper's backtest: a sizing config's portfolio on the validation period (task 7.5).

The same engine and calendar as `sluice evaluate --sizing` (docs/specs/05),
at the main cost, on pipeline v1's first `REPEATS` repeats. Every input comes
from the stores and the cached runs; nothing is asked of a model or Jev.
Refuses until every repeat has a run file for every stock and decision day, so
the gate never judges on a batch still running.

`periods` breaks the same portfolio down by decision period for the proposer
(task 7.6): what was held, what each stock added, against SPY and equal weight.
"""

import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sluice.agents.batch import run_dir
from sluice.agents.config import PipelineConfig, load_config
from sluice.evaluation.backtest import simulate
from sluice.evaluation.decisions import Decision, load_decisions
from sluice.evaluation.report import COSTS, Calendar, calendar, load_prices
from sluice.evolution.gate import REPEATS
from sluice.prices.store import PriceStore
from sluice.sizing import plan
from sluice.sizing.config import SizingConfig

PIPELINE = "v1"  # unweighted news: the one with three repeats (docs/specs/06, task 7.1)
BENCHMARK = "SPY"


def require_complete(
    decisions: list[Decision], tickers: list[str], cal: Calendar, name: str = PIPELINE
) -> None:
    grid = {(day, ticker) for day in cal.decisions for ticker in tickers}
    for repeat in range(REPEATS):
        have = {(d.day, d.ticker) for d in decisions if d.repeat == repeat}
        if missing := grid - have:
            raise ValueError(
                f"{name} repeat {repeat} has {len(grid) - len(missing)} of {len(grid)} "
                "validation runs; finish the batch first"
            )


@dataclass(frozen=True)
class Period:
    """One decision period, averaged over the repeats; returns before costs."""

    start: date  # the rebalance day
    exposure: float
    weights: dict[str, float]
    contributions: dict[str, float]  # weight × the stock's return over the period
    portfolio: float  # the sum of the contributions
    benchmark: float
    equal: float  # equal weight, fully invested


class Validation:
    """Call it with a sizing config for its daily returns per repeat (a `gate.Backtest`)."""

    def __init__(
        self,
        src: plan.Sources,
        prices: PriceStore,
        tickers: list[str],
        runs: Path,
        pipeline: PipelineConfig | None = None,  # whose ratings; v1 unless given (task 9.9)
    ):
        self.src, self.tickers = src, tickers
        self.cal = calendar("validation")
        self.bars = load_prices(prices, [*tickers, BENCHMARK], self.cal)
        folder = run_dir(runs, pipeline or load_config(PIPELINE))
        self.decisions = [d for d in load_decisions(folder, "validation") if d.repeat < REPEATS]
        require_complete(self.decisions, tickers, self.cal, folder.name)
        self.rebalances = dict(zip(self.cal.decisions, self.cal.rebalances, strict=True))

    def _plan(self, config: SizingConfig):
        markets = {d: plan.market(self.src, self.tickers, d, config) for d in self.cal.decisions}
        schedules = plan.formula_schedules(
            self.decisions, markets, self.rebalances, self.src, config
        )
        return markets, {repeat: schedule for repeat, (schedule, _) in schedules.items()}

    def __call__(self, config: SizingConfig) -> dict[int, list[float]]:
        _, schedules = self._plan(config)
        return {
            repeat: simulate(schedule, self.cal.days, self.bars, COSTS[0]).returns
            for repeat, schedule in schedules.items()
        }

    def _growth(self, ticker: str, start: date, end: date | None) -> float:
        """From the open on `start` to the open on `end`, or to the last close when None."""
        first = self.bars[ticker][start].open
        last = self.bars[ticker][end].open if end else self.bars[ticker][self.cal.end].close
        return last / first - 1

    def periods(self, config: SizingConfig) -> list[Period]:
        markets, schedules = self._plan(config)
        starts = self.cal.rebalances
        out = []
        for decision, start, end in zip(
            self.cal.decisions, starts, [*starts[1:], None], strict=True
        ):
            moves = {t: self._growth(t, start, end) for t in self.tickers}
            weights = {
                t: statistics.fmean(schedules[r][start].get(t, 0.0) for r in schedules)
                for t in self.tickers
            }
            contributions = {t: weights[t] * moves[t] for t in self.tickers}
            out.append(
                Period(
                    start=start,
                    exposure=markets[decision].exposure,
                    weights=weights,
                    contributions=contributions,
                    portfolio=sum(contributions.values()),
                    benchmark=self._growth(BENCHMARK, start, end),
                    equal=statistics.fmean(moves.values()),
                )
            )
        return out
