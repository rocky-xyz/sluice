from datetime import UTC, datetime

import pytest

from sluice.news.models import StoredNews
from sluice.scoring.calibration import (
    CalibrationCandidate,
    build_review,
    candidates,
    choice_bins,
    noul_bins,
    parse_review,
    stratified_sample,
)
from sluice.scoring.config import QuestionSpec, ScoringConfig
from sluice.scoring.store import ItemScores, ScoreStore


def item(**overrides) -> StoredNews:
    fields = {
        "id": 1,
        "url": "https://example.com/a",
        "title": "Nvidia beats earnings estimates",
        "summary": "Nvidia reported strong results.",
        "publisher": "Example Wire",
        "source": "finnhub",
        "published_at": datetime(2026, 10, 1, tzinfo=UTC),
        "time_precision": "exact",
        "fetched_at": datetime(2026, 10, 1, tzinfo=UTC),
        "tickers": ["NVDA"],
    }
    fields.update(overrides)
    return StoredNews(**fields)


def scores(**overrides) -> ItemScores:
    fields = {
        "relevance": 0.9,
        "stance": "bullish",
        "stance_confidence": 0.8,
        "stance_probabilities": {"bullish": 0.8, "bearish": 0.15, "neutral": 0.05},
        "novelty": 0.7,
        "rumour": 0.1,
        "event_type": "earnings",
        "event_type_confidence": 0.95,
        "event_type_probabilities": {"earnings": 0.95},
        "input_tokens": 900,
        "output_tokens": 10,
    }
    fields.update(overrides)
    return ItemScores(**fields)


def candidate(**overrides) -> CalibrationCandidate:
    fields = {"ticker": "NVDA", "item": item(), "references": ["An older headline"], "scores": scores()}
    fields.update(overrides)
    return CalibrationCandidate(**fields)


# --- stratified_sample ---


def test_stratified_sample_spreads_across_event_types():
    pool = [
        candidate(item=item(id=i), scores=scores(event_type=et))
        for et in ("earnings", "macro", "other")
        for i in range(10)
    ]
    chosen = stratified_sample(pool, n=6, seed=0)
    counts: dict[str, int] = {}
    for c in chosen:
        counts[c.scores.event_type] = counts.get(c.scores.event_type, 0) + 1
    assert counts == {"earnings": 2, "macro": 2, "other": 2}


def test_stratified_sample_backfills_from_other_types_when_one_runs_out():
    pool = [candidate(item=item(id=1), scores=scores(event_type="earnings"))]
    pool += [candidate(item=item(id=i), scores=scores(event_type="macro")) for i in range(2, 6)]
    chosen = stratified_sample(pool, n=4, seed=0)
    assert len(chosen) == 4
    assert sum(1 for c in chosen if c.scores.event_type == "earnings") == 1


def test_stratified_sample_caps_at_the_pool_size():
    pool = [candidate(item=item(id=1))]
    assert len(stratified_sample(pool, n=10, seed=0)) == 1


def test_stratified_sample_is_deterministic_for_a_given_seed():
    pool = [candidate(item=item(id=i), scores=scores(event_type="earnings")) for i in range(20)]
    first = [c.item.id for c in stratified_sample(pool, n=5, seed=1)]
    second = [c.item.id for c in stratified_sample(pool, n=5, seed=1)]
    assert first == second


# --- build_review / parse_review round trip ---


def test_build_review_keeps_jevs_scores_out_of_the_text():
    text, answers = build_review([candidate(item=item(id=1))])
    assert "0.9" not in text  # relevance
    assert "bullish" not in text.split("relevant")[0]  # not leaked before the blank
    assert answers[1] == scores()


def test_build_review_includes_the_reference_headlines():
    text, _ = build_review([candidate(item=item(id=1), references=["Old headline one"])])
    assert "Old headline one" in text


def test_parse_review_reads_back_filled_in_blanks():
    text, _ = build_review([candidate(item=item(id=1))])
    filled = (
        text.replace("relevant (0 or 1):", "relevant (0 or 1): 1")
        .replace("stance (bullish/bearish/neutral):", "stance (bullish/bearish/neutral): bearish")
        .replace("novel (0 or 1):", "novel (0 or 1): 0")
        .replace("rumour (0 or 1):", "rumour (0 or 1): 1")
        .replace(
            "event_type (earnings/guidance/m_and_a/regulatory/contract_product/analyst_action/macro/other):",
            "event_type (earnings/guidance/m_and_a/regulatory/contract_product/analyst_action/macro/other): guidance",
        )
    )
    labels = parse_review(filled)
    assert labels[1].relevant is True
    assert labels[1].stance == "bearish"
    assert labels[1].novel is False
    assert labels[1].rumour is True
    assert labels[1].event_type == "guidance"


def test_parse_review_round_trips_several_items():
    text, _ = build_review([candidate(item=item(id=1)), candidate(item=item(id=2))])
    filled = text
    for field, value in (
        ("relevant (0 or 1):", "relevant (0 or 1): 1"),
        ("stance (bullish/bearish/neutral):", "stance (bullish/bearish/neutral): neutral"),
        ("novel (0 or 1):", "novel (0 or 1): 1"),
        ("rumour (0 or 1):", "rumour (0 or 1): 0"),
    ):
        filled = filled.replace(field, value)
    filled = filled.replace(
        "event_type (earnings/guidance/m_and_a/regulatory/contract_product/analyst_action/macro/other):",
        "event_type (earnings/guidance/m_and_a/regulatory/contract_product/analyst_action/macro/other): other",
    )
    labels = parse_review(filled)
    assert set(labels) == {1, 2}


def test_parse_review_raises_on_a_blank_still_empty():
    text, _ = build_review([candidate(item=item(id=1))])
    with pytest.raises(ValueError, match="news_id=1"):
        parse_review(text)


# --- candidates() ---


class FakeNewsStore:
    def __init__(self, items_by_ticker: dict[str, list[StoredNews]]):
        self._items = items_by_ticker

    def query(self, ticker, before, dedup=True):
        return list(self._items.get(ticker, []))


def config() -> ScoringConfig:
    return ScoringConfig(
        version="test",
        digest="deadbeef",
        model="jev-1.13.0",
        novelty_reference_k=5,
        questions={"relevance": QuestionSpec(type="noul", instructions="x")},
    )


def test_candidates_skips_items_that_were_never_scored(tmp_path):
    with ScoreStore(tmp_path / "scores.db") as score_store:
        score_store.put(1, "test", "deadbeef", scores())
        news = FakeNewsStore({"NVDA": [item(id=1), item(id=2, title="Nvidia announces buyback")]})
        found = candidates(["NVDA"], news, score_store, config())
    assert [c.item.id for c in found] == [1]


def test_candidates_recomputes_the_reference_headlines(tmp_path):
    older = item(id=1, title="Nvidia announces buyback", published_at=datetime(2026, 9, 1, tzinfo=UTC))
    newer = item(id=2, title="Nvidia beats earnings estimates", published_at=datetime(2026, 10, 1, tzinfo=UTC))
    with ScoreStore(tmp_path / "scores.db") as score_store:
        score_store.put(2, "test", "deadbeef", scores())
        news = FakeNewsStore({"NVDA": [older, newer]})
        found = candidates(["NVDA"], news, score_store, config())
    assert found[0].references == ["Nvidia announces buyback"]


# --- noul_bins / choice_bins ---


def _rounded(rows: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    return [(round(a, 6), round(b, 6), n) for a, b, n in rows]


def test_noul_bins_groups_by_predicted_probability():
    pairs = [(0.05, False), (0.1, True), (0.9, True), (0.95, True)]
    assert _rounded(noul_bins(pairs, bins=2)) == [(0.075, 0.5, 2), (0.925, 1.0, 2)]


def test_noul_bins_skips_empty_buckets():
    assert noul_bins([(0.95, True)], bins=5) == [(0.95, 1.0, 1)]


def test_choice_bins_groups_by_confidence():
    triples = [(True, 0.1), (False, 0.2), (True, 0.8), (True, 0.9)]
    assert _rounded(choice_bins(triples, bins=2)) == [(0.15, 0.5, 2), (0.85, 1.0, 2)]
