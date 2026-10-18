"""Source/event-type weight learning (docs/specs/03-news-weights.md, task 4.3).

Model: y_i = Σ_s 1[source_i=s]·raw_i·β_s + Σ_t 1[type_i=t]·raw_i·β_t + ε_i,
raw_i = stance_score_i (P(bullish) − P(bearish)). No intercept — that is the
model itself, not an omission. Ridge-penalized: each dummy's penalty is
λ0/n_group, with n_group counted once from the full dataset and then held
fixed, so a source or event type seen rarely shrinks harder toward zero (no
excess-return signal) than one seen often, in both the main fit and every
bootstrap resample. λ0 itself is chosen by a single time-ordered holdout
(earliest 80% fit, latest 20% validate) — never a shuffled k-fold, per the
project's walk-forward-only rule.

Uncertainty is a plain item-level (i.i.d.) bootstrap. A news item's label
spans several trading days, so items published close together can share
overlapping price history and are not really independent draws; the
percentile intervals below are therefore likely a bit narrower than the true
sampling uncertainty. Flagged rather than fixed — a block bootstrap is the
documented upgrade path if this turns out to matter in practice.

On the real v1-scored dataset (task 3.6 already found near-zero raw
correlations between any Jev dimension and the excess-return labels), the
holdout keeps preferring more regularization all the way to the top of
`LAMBDA_GRID`, with the validation MSE essentially flat across five orders of
magnitude of λ0. That is itself the finding, not a bug to chase: it says the
cross-validation cannot tell a fitted source/type effect apart from "no
effect" on this horizon with this feature. If `lambda0 == LAMBDA_GRID[-1]` in
a `WeightTable`, read every weight as "shrunk to essentially indistinguishable
from zero," not as a confident point estimate.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from sluice.news.labels import item_label
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore
from sluice.scoring.store import ScoreStore

HORIZON_DAYS = 3  # primary horizon, per docs/specs/03-news-weights.md
LAMBDA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0)
BOOTSTRAP_DRAWS = 200


@dataclass(frozen=True)
class FittingRow:
    published_at: datetime
    source: str
    event_type: str
    raw: float  # stance_score
    label: float  # the item's primary-horizon excess-return label


def stance_score(probabilities: dict[str, float]) -> float:
    return probabilities.get("bullish", 0.0) - probabilities.get("bearish", 0.0)


def assemble_rows(
    news: NewsStore,
    scores: ScoreStore,
    prices: PriceStore,
    companies: list[str],
    as_of: datetime,
    config_version: str = "v1",
) -> list[FittingRow]:
    """One row per scored news item whose primary-horizon label has resolved by `as_of`."""
    rows = []
    for news_id, score in scores.iter_all(config_version):
        item = news.get(news_id)
        if item is None or item.published_at is None:
            continue
        tickers = sorted(set(item.tickers) & set(companies))
        label = item_label(tickers, item.published_at, HORIZON_DAYS, prices, as_of)
        if label is None:
            continue
        rows.append(
            FittingRow(
                published_at=item.published_at,
                source=item.publisher or "(unknown)",
                event_type=score.event_type,
                raw=stance_score(score.stance_probabilities),
                label=label,
            )
        )
    return rows


@dataclass(frozen=True)
class WeightEstimate:
    weight: float
    low: float  # 2.5th percentile of the bootstrap
    high: float  # 97.5th percentile of the bootstrap
    n: int  # rows this group had in the fitted dataset


@dataclass(frozen=True)
class WeightTable:
    lambda0: float
    source: dict[str, WeightEstimate]
    event_type: dict[str, WeightEstimate]


def _design(
    rows: list[FittingRow], sources: list[str], types: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each row touches exactly one source column and one type column."""
    source_index = {s: i for i, s in enumerate(sources)}
    type_index = {t: i for i, t in enumerate(types)}
    source_idx = np.array([source_index[r.source] for r in rows])
    type_idx = np.array([type_index[r.event_type] for r in rows])
    raw = np.array([r.raw for r in rows])
    y = np.array([r.label for r in rows])
    return source_idx, type_idx, raw, y


def _solve(
    source_idx: np.ndarray,
    type_idx: np.ndarray,
    raw: np.ndarray,
    y: np.ndarray,
    n_source: np.ndarray,
    n_type: np.ndarray,
    lambda0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """β = (XᵀX + D)⁻¹Xᵀy, with XᵀX built from grouped sums — the dense design
    matrix (mostly zeros: one row touches only 2 of its ~280 columns) is never
    materialized.
    """
    n_sources, n_types = len(n_source), len(n_type)
    raw2 = raw * raw
    source_source = np.bincount(source_idx, weights=raw2, minlength=n_sources)
    type_type = np.bincount(type_idx, weights=raw2, minlength=n_types)
    cross = np.bincount(
        source_idx * n_types + type_idx, weights=raw2, minlength=n_sources * n_types
    ).reshape(n_sources, n_types)

    xtx = np.zeros((n_sources + n_types, n_sources + n_types))
    xtx[:n_sources, :n_sources] = np.diag(source_source)
    xtx[n_sources:, n_sources:] = np.diag(type_type)
    xtx[:n_sources, n_sources:] = cross
    xtx[n_sources:, :n_sources] = cross.T

    penalty = np.concatenate([lambda0 / n_source, lambda0 / n_type])
    xty = np.concatenate(
        [
            np.bincount(source_idx, weights=raw * y, minlength=n_sources),
            np.bincount(type_idx, weights=raw * y, minlength=n_types),
        ]
    )
    beta = np.linalg.solve(xtx + np.diag(penalty), xty)
    return beta[:n_sources], beta[n_sources:]


def _choose_lambda0(
    rows: list[FittingRow],
    sources: list[str],
    types: list[str],
    n_source: np.ndarray,
    n_type: np.ndarray,
) -> float:
    """A single time-ordered 80/20 holdout. Never a shuffled k-fold: this project's
    time-series validation is walk-forward only.
    """
    if len(rows) < 10:
        return LAMBDA_GRID[len(LAMBDA_GRID) // 2]
    ordered = sorted(rows, key=lambda r: r.published_at)
    split = int(len(ordered) * 0.8)
    train_idx, valid_idx = _design(ordered[:split], sources, types), _design(
        ordered[split:], sources, types
    )

    best_lambda, best_mse = LAMBDA_GRID[0], float("inf")
    for lambda0 in LAMBDA_GRID:
        beta_s, beta_t = _solve(*train_idx, n_source, n_type, lambda0)
        valid_source_idx, valid_type_idx, valid_raw, valid_y = valid_idx
        predicted = valid_raw * (beta_s[valid_source_idx] + beta_t[valid_type_idx])
        mse = float(np.mean((valid_y - predicted) ** 2))
        if mse < best_mse:
            best_lambda, best_mse = lambda0, mse
    return best_lambda


def fit_weights(rows: list[FittingRow], seed: int = 0) -> WeightTable:
    if len(rows) < 2:
        raise ValueError("need at least 2 fitted rows")
    sources = sorted({r.source for r in rows})
    types = sorted({r.event_type for r in rows})
    source_idx, type_idx, raw, y = _design(rows, sources, types)
    n_source = np.bincount(source_idx, minlength=len(sources)).astype(float)
    n_type = np.bincount(type_idx, minlength=len(types)).astype(float)

    lambda0 = _choose_lambda0(rows, sources, types, n_source, n_type)
    beta_s, beta_t = _solve(source_idx, type_idx, raw, y, n_source, n_type, lambda0)

    rng = np.random.default_rng(seed)
    n = len(rows)
    boot_s = np.empty((BOOTSTRAP_DRAWS, len(sources)))
    boot_t = np.empty((BOOTSTRAP_DRAWS, len(types)))
    for b in range(BOOTSTRAP_DRAWS):
        draw = rng.integers(0, n, size=n)
        boot_s[b], boot_t[b] = _solve(
            source_idx[draw], type_idx[draw], raw[draw], y[draw], n_source, n_type, lambda0
        )
    source_low, source_high = np.percentile(boot_s, [2.5, 97.5], axis=0)
    type_low, type_high = np.percentile(boot_t, [2.5, 97.5], axis=0)

    return WeightTable(
        lambda0=lambda0,
        source={
            s: WeightEstimate(
                weight=float(beta_s[i]),
                low=float(source_low[i]),
                high=float(source_high[i]),
                n=int(n_source[i]),
            )
            for i, s in enumerate(sources)
        },
        event_type={
            t: WeightEstimate(
                weight=float(beta_t[i]),
                low=float(type_low[i]),
                high=float(type_high[i]),
                n=int(n_type[i]),
            )
            for i, t in enumerate(types)
        },
    )
