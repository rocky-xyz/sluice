"""Portfolio metrics on daily returns (docs/specs/05-evaluation.md, task 6.1).

Annualised by √252 with a risk-free rate of 0, the same for every strategy and
baseline.
"""

import math
import statistics

TRADING_DAYS = 252


def sharpe(returns: list[float]) -> float | None:
    if len(returns) < 2 or (spread := statistics.stdev(returns)) == 0:
        return None
    return statistics.fmean(returns) / spread * math.sqrt(TRADING_DAYS)


def sortino(returns: list[float]) -> float | None:
    """Like Sharpe, but only days below 0 count as risk: root mean square of min(r, 0)."""
    if len(returns) < 2:
        return None
    downside = math.sqrt(statistics.fmean(min(r, 0.0) ** 2 for r in returns))
    if downside == 0:
        return None
    return statistics.fmean(returns) / downside * math.sqrt(TRADING_DAYS)


def max_drawdown(values: list[float]) -> float:
    """The largest fall from a running peak, as a share of that peak (0 when it never falls)."""
    peak, worst = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, 1 - value / peak)
    return worst


def total_return(values: list[float]) -> float:
    return values[-1] / values[0] - 1
