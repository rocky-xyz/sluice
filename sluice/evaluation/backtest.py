"""Daily portfolio values for a schedule of target weights (docs/specs/05-evaluation.md).

Weights are set at the open of each rebalance day and then left to drift with
prices until the next one. On a rebalance day the value moves in three steps:
the overnight gap on the weights held, the trade at the open (charged `cost`
on the value traded), and the open-to-close move on the new weights. Before the
first rebalance everything is cash, which earns nothing; so the first trade
buys the whole book and pays for it, for every strategy and baseline alike.

Prices are adjusted opens and closes. The cost is taken from the value after
the weights are set, a slight simplification: it scales every position down
by the same factor instead of coming out of one of them.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PriceBar:
    open: float
    close: float


Prices = dict[str, dict[date, PriceBar]]  # ticker -> day -> bar


@dataclass(frozen=True)
class Path:
    days: list[date]
    values: list[float]  # at each day's close, starting from 1 before the first day
    returns: list[float]  # daily, one per day
    turnover: list[float]  # one per rebalance: the sum of |target - held| weights
    costs: float  # total paid, as a share of the starting value


def simulate(
    schedule: dict[date, dict[str, float]],
    days: list[date],
    prices: Prices,
    cost: float,
) -> Path:
    """Run the schedule over `days` (trading days, the first being the first rebalance).

    `schedule` maps rebalance days to target weights summing to at most 1; the
    rest is cash.
    """
    if not days or days[0] not in schedule:
        raise ValueError("the first day must be a rebalance day")
    if unknown := sorted(set(schedule) - set(days)):
        raise ValueError(f"rebalance days outside the simulated days: {unknown}")

    value, held, cash = 1.0, {}, 1.0  # held: ticker -> share of value at the last close
    previous_close: dict[str, float] = {}
    values, returns, turnover, paid = [value], [], [], 0.0
    for day in days:
        start = value
        if day in schedule:
            # 1. overnight: what was held moves from yesterday's close to today's open
            moved = {t: w * prices[t][day].open / previous_close[t] for t, w in held.items()}
            at_open = start * (cash + sum(moved.values()))
            drifted = {t: m * start / at_open for t, m in moved.items()}
            # 2. trade at the open
            target = schedule[day]
            traded = sum(
                abs(target.get(t, 0.0) - drifted.get(t, 0.0)) for t in set(target) | set(drifted)
            )
            charge = cost * traded * at_open
            turnover.append(traded)
            paid += charge
            after = at_open - charge
            # 3. open to close on the new weights
            cash = 1.0 - sum(target.values())
            grown = {t: w * prices[t][day].close / prices[t][day].open for t, w in target.items()}
            value = after * (cash + sum(grown.values()))
            held = {t: g * after / value for t, g in grown.items()}
            cash = cash * after / value
        else:
            grown = {t: w * prices[t][day].close / previous_close[t] for t, w in held.items()}
            value = start * (cash + sum(grown.values()))
            held = {t: g * start / value for t, g in grown.items()}
            cash = cash * start / value
        previous_close = {t: prices[t][day].close for t in held}
        values.append(value)
        returns.append(value / start - 1)
    return Path(days=days, values=values, returns=returns, turnover=turnover, costs=paid)
