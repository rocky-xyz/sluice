"""Ratings to target weights, until the sizing module replaces this (docs/specs/05, task 6.4).

A tilt on equal weight that is always fully invested: each stock's weight is
its rating's multiple over the sum of every stock's multiple that period. No
shorting, no leverage. With every rating "hold" this is exactly equal weight,
so the strategy differs from the equal-weight baseline by the tilts alone.
"""

from sluice.agents.state import Rating

MULTIPLES: dict[Rating, float] = {
    "buy": 2.0,
    "overweight": 1.5,
    "hold": 1.0,
    "underweight": 0.5,
    "sell": 0.0,
}
NEUTRAL: Rating = "hold"  # stands in for a failed or missing run


def target_weights(ratings: dict[str, Rating | None]) -> dict[str, float]:
    """Weights summing to 1 over every ticker in `ratings`; None counts as hold.

    When every multiple is 0 (all "sell") there is nothing to tilt towards and
    the weights fall back to equal: the portfolio stays invested, as the
    baselines do.
    """
    multiples = {t: MULTIPLES[r if r is not None else NEUTRAL] for t, r in ratings.items()}
    total = sum(multiples.values())
    if total == 0:
        return {t: 1 / len(multiples) for t in multiples}
    return {t: m / total for t, m in multiples.items()}
