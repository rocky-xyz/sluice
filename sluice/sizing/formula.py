"""The sizing formula, steps 2 to 9 (docs/specs/04-sizing.md, "已确认（任务 6.8）").

Pure arithmetic on numbers already computed: no store, no model, no Jev. Every
sum runs over the tickers in sorted order, so the same inputs give the same
weights to the last bit whatever order they came in.
"""

import math
from dataclasses import dataclass

from sluice.sizing.config import Coefficients


@dataclass(frozen=True)
class Inputs:
    """Per ticker, each input as computed; None where it is missing."""

    rating: dict[str, float | None]  # buy +2 .. sell -2
    reports: dict[str, float | None]  # mean P(bullish) - P(bearish) over the analyst reports
    news: dict[str, float | None]  # the news signal


def zscores(values: dict[str, float | None]) -> dict[str, float]:
    """Cross-sectional z-scores; missing values, and everything when there is no spread, are 0."""
    present = {t: v for t, v in sorted(values.items()) if v is not None}
    if len(present) < 2:
        return dict.fromkeys(values, 0.0)
    mean = math.fsum(present.values()) / len(present)
    spread = math.sqrt(math.fsum((v - mean) ** 2 for v in present.values()) / len(present))
    if spread == 0:
        return dict.fromkeys(values, 0.0)
    return {t: (present[t] - mean) / spread if t in present else 0.0 for t in values}


def combined(inputs: Inputs, coefficients: Coefficients) -> dict[str, float]:
    """s_i: the coefficient-weighted mean of the three z-scores."""
    parts = [
        (coefficients.rating, zscores(inputs.rating)),
        (coefficients.reports, zscores(inputs.reports)),
        (coefficients.news, zscores(inputs.news)),
    ]
    total = math.fsum(c for c, _ in parts)
    if total == 0:
        return dict.fromkeys(inputs.rating, 0.0)
    return {
        t: math.fsum(c * z[t] for c, z in parts) / total for t in sorted(inputs.rating)
    }


def cap(weights: dict[str, float], max_weight: float) -> dict[str, float]:
    """Clip at `max_weight` and share the excess among the others in proportion, until none is over.

    `weights` sum to 1. Refuses when even equal weights would break the cap.
    """
    if max_weight * len(weights) < 1:
        raise ValueError(f"{len(weights)} stocks cannot sum to 1 under a cap of {max_weight}")
    capped: set[str] = set()
    result = dict(weights)
    while over := sorted(t for t, w in result.items() if t not in capped and w > max_weight):
        capped.update(over)
        free = [t for t in sorted(result) if t not in capped]
        room = 1 - max_weight * len(capped)
        free_total = math.fsum(result[t] for t in free)
        updated = dict.fromkeys(capped, max_weight)
        for t in free:
            updated[t] = result[t] / free_total * room if free_total else room / len(free)
        result = {t: updated[t] for t in sorted(result)}
    return result


def target_weights(
    inputs: Inputs,
    vols: dict[str, float],
    exposure: float,
    k: float,
    coefficients: Coefficients,
    max_weight: float,
    scale_by_vol: bool = True,
) -> dict[str, float]:
    """Steps 3 to 9: the target weight of every ticker; the rest of the book is cash."""
    tickers = sorted(inputs.rating)
    s = combined(inputs, coefficients)
    tilt = {t: max(0.0, 1 + k * s[t]) for t in tickers}
    raw = {t: tilt[t] / vols[t] if scale_by_vol else tilt[t] for t in tickers}
    total = math.fsum(raw.values())
    if total:
        shares = {t: raw[t] / total for t in tickers}
    else:  # every tilt is 0: nothing to lean towards, so equal weight
        shares = {t: 1 / len(tickers) for t in tickers}
    return {t: exposure * w for t, w in cap(shares, max_weight).items()}
