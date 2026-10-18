import random
from datetime import UTC, date, datetime, timedelta

import pytest

from sluice.news.models import NewsItem
from sluice.news.store import NewsStore
from sluice.news.timeutil import ET
from sluice.news.weights import FittingRow, assemble_rows, fit_weights, stance_score
from sluice.prices.store import Bar, PriceStore
from sluice.scoring.store import ItemScores, ScoreStore

START = datetime(2026, 1, 1, tzinfo=UTC)
TYPES = ("earnings", "guidance")


def test_stance_score_is_bullish_minus_bearish():
    assert stance_score({"bullish": 0.7, "bearish": 0.2, "neutral": 0.1}) == pytest.approx(0.5)


def _rows(n: int, source: str, true_beta: float, rng) -> list[FittingRow]:
    """Noiseless rows for one source: label = raw * true_beta, type picked with no
    effect of its own (split evenly, independent of the label).
    """
    rows = []
    for i in range(n):
        raw = 1.0 if rng.random() < 0.5 else -1.0
        event_type = TYPES[i % 2]  # alternates independently of raw/label
        rows.append(
            FittingRow(
                published_at=START + timedelta(hours=i),
                source=source,
                event_type=event_type,
                raw=raw,
                label=raw * true_beta,
            )
        )
    return rows


def test_fit_weights_needs_at_least_two_rows():
    with pytest.raises(ValueError):
        fit_weights([FittingRow(published_at=START, source="A", event_type="earnings", raw=1.0, label=0.01)])


def test_a_sparse_source_is_shrunk_harder_than_a_well_sampled_one():
    # Two true effect levels (+0.05 / -0.05), each carried by one populous and one
    # sparse source, so a 2-category event_type genuinely cannot stand in for the
    # source effect (it would have to be both +0.05 and -0.05 at once). This keeps
    # the comparison clean: any deviation from the true value is ridge shrinkage,
    # not the model routing the shared signal through the other dimension instead.
    rng = random.Random(0)
    rows = (
        _rows(2000, "popular_a", 0.05, rng)
        + _rows(2000, "popular_b", -0.05, rng)
        + _rows(5, "rare_a", 0.05, rng)
        + _rows(5, "rare_b", -0.05, rng)
    )

    table = fit_weights(rows, seed=0)
    popular_a, rare_a = table.source["popular_a"], table.source["rare_a"]

    assert popular_a.n == 2000
    assert rare_a.n == 5
    # Same true effect (+0.05), but ridge's group-size-scaled penalty should pull
    # the 5-row source noticeably closer to zero than the 2000-row one.
    assert abs(popular_a.weight - 0.05) < abs(rare_a.weight - 0.05)
    assert popular_a.weight == pytest.approx(0.05, abs=1e-4)
    # event_type can't fake the divergent a/b pattern, so it should sit near zero.
    for estimate in table.event_type.values():
        assert abs(estimate.weight) < 1e-6


def test_event_type_without_its_own_effect_fits_near_zero():
    rng = random.Random(1)
    rows = _rows(1000, "only_source", 0.02, rng)
    table = fit_weights(rows, seed=0)
    for estimate in table.event_type.values():
        assert abs(estimate.weight) < 0.01


def _score(**overrides) -> ItemScores:
    fields = {
        "relevance": 0.9,
        "stance": "bullish",
        "stance_confidence": 0.8,
        "stance_probabilities": {"bullish": 0.7, "bearish": 0.1, "neutral": 0.2},
        "novelty": 0.5,
        "rumour": 0.1,
        "event_type": "earnings",
        "event_type_confidence": 0.8,
        "event_type_probabilities": {"earnings": 0.8},
    }
    fields.update(overrides)
    return ItemScores(**fields)


def test_assemble_rows_falls_back_to_unknown_publisher_and_skips_unresolved_labels(tmp_path):
    published = datetime(2026, 9, 28, 11, 0, tzinfo=ET)  # Monday, mid-session
    entry_day, exit_day = date(2026, 9, 29), date(2026, 10, 1)
    after_exit_close = datetime(2026, 10, 1, 21, 0, tzinfo=UTC)
    too_soon = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)  # before the 3-day window closes

    with (
        NewsStore(tmp_path / "news.db") as news,
        ScoreStore(tmp_path / "scores.db") as scores,
        PriceStore(tmp_path / "prices.db") as prices,
    ):
        news.add(
            [
                NewsItem(
                    title="No named publisher",
                    source="finnhub",
                    tickers=["NVDA"],
                    publisher=None,
                    published_at=published,
                    fetched_at=published,
                )
            ]
        )
        (news_id,) = (item.id for item in news.query("NVDA", before=after_exit_close))
        scores.put(news_id, "v1", "digest", _score())

        for ticker, entry_open, exit_close in [("NVDA", 100.0, 110.0), ("SPY", 400.0, 400.0)]:
            prices.replace(
                ticker,
                [
                    Bar(day=entry_day, open=entry_open, close=entry_open, raw_close=entry_open, volume=1.0),
                    Bar(day=exit_day, open=exit_close, close=exit_close, raw_close=exit_close, volume=1.0),
                ],
            )

        resolved = assemble_rows(news, scores, prices, ["NVDA"], as_of=after_exit_close)
        unresolved = assemble_rows(news, scores, prices, ["NVDA"], as_of=too_soon)

    assert unresolved == []
    (row,) = resolved
    assert row.source == "(unknown)"
    assert row.event_type == "earnings"
    assert row.raw == pytest.approx(0.6)  # P(bullish) - P(bearish) = 0.7 - 0.1
