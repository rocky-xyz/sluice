"""The gatekeeper's backtest: a sizing config's portfolio on the validation period (task 7.5).

The same engine and calendar as `sluice evaluate --sizing` (docs/specs/05),
at the main cost, on pipeline v1's first `REPEATS` repeats. Every input comes
from the stores and the cached runs; nothing is asked of a model or Jev.
Refuses until every repeat has a run file for every stock and decision day, so
the gate never judges on a batch still running.
"""

from pathlib import Path

from sluice.agents.batch import run_dir
from sluice.agents.config import load_config
from sluice.evaluation.backtest import simulate
from sluice.evaluation.decisions import Decision, load_decisions
from sluice.evaluation.report import COSTS, Calendar, calendar, load_prices
from sluice.evolution.gate import REPEATS, Backtest
from sluice.prices.store import PriceStore
from sluice.sizing import plan
from sluice.sizing.config import SizingConfig

PIPELINE = "v1"  # unweighted news: the one with three repeats (docs/specs/06, task 7.1)


def require_complete(decisions: list[Decision], tickers: list[str], cal: Calendar) -> None:
    grid = {(day, ticker) for day in cal.decisions for ticker in tickers}
    for repeat in range(REPEATS):
        have = {(d.day, d.ticker) for d in decisions if d.repeat == repeat}
        if missing := grid - have:
            raise ValueError(
                f"{PIPELINE} repeat {repeat} has {len(grid) - len(missing)} of {len(grid)} "
                "validation runs; finish the batch first"
            )


def validation_backtest(
    src: plan.Sources, prices: PriceStore, tickers: list[str], runs: Path
) -> Backtest:
    cal = calendar("validation")
    bars = load_prices(prices, tickers, cal)
    folder = run_dir(runs, load_config(PIPELINE))
    decisions = [d for d in load_decisions(folder, "validation") if d.repeat < REPEATS]
    require_complete(decisions, tickers, cal)
    rebalances = dict(zip(cal.decisions, cal.rebalances, strict=True))

    def run(config: SizingConfig) -> dict[int, list[float]]:
        markets = {day: plan.market(src, tickers, day, config) for day in cal.decisions}
        schedules = plan.formula_schedules(decisions, markets, rebalances, src, config)
        return {
            repeat: simulate(schedule, cal.days, bars, COSTS[0]).returns
            for repeat, (schedule, _) in schedules.items()
        }

    return run
