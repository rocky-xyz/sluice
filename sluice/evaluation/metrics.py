"""Portfolio metrics on daily returns (docs/specs/05-evaluation.md, task 6.1).

Annualised by √252 with a risk-free rate of 0, the same for every strategy and
baseline.
"""

import math
import statistics
from dataclasses import dataclass

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


@dataclass(frozen=True)
class SharpeDifference:
    z: float
    rho: float  # correlation of the two daily return series
    days: int


def sharpe_difference(a: list[float], b: list[float]) -> SharpeDifference | None:
    """Is a's Sharpe above b's beyond chance, on the same days? (task 7.5, the gatekeeper's test)

    Jobson and Korkie (1981) with Memmel's (2003) correction, on daily,
    unannualised Sharpe ratios: Var = [2(1 − ρ) + ½(SR_a² + SR_b² − 2·SR_a·SR_b·ρ²)] / T,
    z = (SR_a − SR_b) / √Var. The closer two strategies move together, the
    smaller the variance, so a small but real difference between similar
    configs can show. Kept here, under the frozen evaluation, so the evolution
    loop cannot change the test it is judged by. None when either series is
    flat or the two cannot be told apart (Var = 0).
    """
    if len(a) != len(b):
        raise ValueError(f"{len(a)} days against {len(b)}: the series must cover the same days")
    if len(a) < 3:
        return None
    spread_a, spread_b = statistics.stdev(a), statistics.stdev(b)
    if spread_a == 0 or spread_b == 0:
        return None
    sr_a, sr_b = statistics.fmean(a) / spread_a, statistics.fmean(b) / spread_b
    rho = statistics.correlation(a, b)
    days = len(a)
    variance = (2 * (1 - rho) + 0.5 * (sr_a**2 + sr_b**2 - 2 * sr_a * sr_b * rho**2)) / days
    if variance <= 0:
        return None
    return SharpeDifference(z=(sr_a - sr_b) / math.sqrt(variance), rho=rho, days=days)
