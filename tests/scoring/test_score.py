from datetime import UTC, datetime

import pytest

from sluice.news.models import StoredNews
from sluice.scoring.client import JevError, JevResponse
from sluice.scoring.config import QuestionSpec, ScoringConfig
from sluice.scoring.score import build_state, jobs_for_ticker, score_all, score_item
from sluice.scoring.store import ItemScores, ScoreStore


def item(**overrides) -> StoredNews:
    fields = {
        "id": 1,
        "url": "https://example.com/a",
        "title": "Nvidia beats earnings estimates",
        "summary": None,
        "publisher": "Example Wire",
        "source": "finnhub",
        "published_at": datetime(2026, 10, 1, tzinfo=UTC),
        "time_precision": "exact",
        "fetched_at": datetime(2026, 10, 1, tzinfo=UTC),
        "tickers": ["NVDA"],
    }
    fields.update(overrides)
    return StoredNews(**fields)


def config(novelty_reference_k: int = 2) -> ScoringConfig:
    return ScoringConfig(
        version="test",
        digest="deadbeef",
        model="jev-1.13.0",
        novelty_reference_k=novelty_reference_k,
        questions={
            "relevance": QuestionSpec(type="noul", instructions="is this relevant?"),
            "stance": QuestionSpec(
                type="choice",
                instructions="pick a stance",
                criteria={"bullish": "x", "bearish": "y", "neutral": "z"},
            ),
            "novelty": QuestionSpec(type="noul", instructions="is this novel?"),
            "rumour": QuestionSpec(type="noul", instructions="is this a rumour?"),
            "event_type": QuestionSpec(
                type="choice",
                instructions="classify the event",
                criteria={"earnings": "x", "other": "y"},
            ),
        },
    )


def full_answer() -> dict:
    return {
        "relevance": {"type": "noul", "noul": 0.9},
        "stance": {
            "type": "choice",
            "choice": "bullish",
            "probabilities": {"bullish": 0.8, "bearish": 0.1, "neutral": 0.1},
            "confidence": 0.8,
        },
        "novelty": {"type": "noul", "noul": 0.6},
        "rumour": {"type": "noul", "noul": 0.1},
        "event_type": {
            "type": "choice",
            "choice": "earnings",
            "probabilities": {"earnings": 0.9, "other": 0.1},
            "confidence": 0.9,
        },
    }


class FakeJev:
    """Stands in for CachedJev: answers in order, records every request sent."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return JevResponse(answers=outcome, input_tokens=10, output_tokens=2)


# --- build_state ---


def test_build_state_with_nothing_extra():
    assert build_state("Title here", None, []) == "Headline: Title here"


def test_build_state_includes_summary_and_references():
    state = build_state("Title", "Summary text", ["Older headline one", "Older headline two"])
    assert state == (
        "Headline: Title\n"
        "Summary: Summary text\n"
        "Recent headlines about the same company, most similar first:\n"
        "- Older headline one\n"
        "- Older headline two"
    )


# --- score_item: happy path ---


def test_score_item_returns_parsed_scores_on_the_first_try():
    jev = FakeJev(full_answer())
    scores, skip_reason = score_item(item(), [], config(), jev)

    assert skip_reason is None
    assert scores == ItemScores(
        relevance=0.9,
        stance="bullish",
        stance_confidence=0.8,
        stance_probabilities={"bullish": 0.8, "bearish": 0.1, "neutral": 0.1},
        novelty=0.6,
        rumour=0.1,
        event_type="earnings",
        event_type_confidence=0.9,
        event_type_probabilities={"earnings": 0.9, "other": 0.1},
        input_tokens=10,
        output_tokens=2,
    )
    assert len(jev.requests) == 1


def test_a_full_history_pool_is_narrowed_to_the_most_similar_references():
    jev = FakeJev(full_answer())
    history = ["Coca-Cola launches new flavor", "Nvidia misses earnings estimates", "Nvidia beats earnings estimates"]
    score_item(item(), history, config(novelty_reference_k=2), jev)

    state = jev.requests[0].state
    assert "Nvidia beats earnings estimates" in state
    assert "Nvidia misses earnings estimates" in state
    assert "Coca-Cola launches new flavor" not in state


# --- score_item: shrinking a too-large state ---


def test_dropping_the_reference_list_is_tried_before_touching_the_summary():
    too_large = JevError("HTTP 422", too_large=True)
    jev = FakeJev(too_large, full_answer())
    history = ["Some older headline"]
    score_item(item(summary="A short summary."), history, config(novelty_reference_k=1), jev)

    assert len(jev.requests) == 2
    assert "Recent headlines" in jev.requests[0].state
    assert "Recent headlines" not in jev.requests[1].state
    assert "A short summary." in jev.requests[1].state  # the summary itself is kept


def test_the_summary_is_shrunk_to_its_most_relevant_sentences_next():
    too_large = JevError("HTTP 422", too_large=True)
    summary = "First sentence here. Second sentence here. Third sentence here."
    jev = FakeJev(too_large, too_large, full_answer())
    scores, skip_reason = score_item(
        item(summary=summary), [], config(), jev
    )

    assert skip_reason is None
    assert scores is not None
    assert len(jev.requests) == 3
    # attempt 1: the full summary; attempt 2: two of its three sentences
    assert summary in jev.requests[0].state
    assert jev.requests[1].state != jev.requests[0].state
    assert "Summary:" in jev.requests[2].state


def test_giving_up_returns_state_too_large_and_tries_every_variant():
    too_large = JevError("HTTP 422", too_large=True)
    summary = "First sentence here. Second sentence here."
    jev = FakeJev(*[too_large] * 10)
    scores, skip_reason = score_item(item(summary=summary), [], config(), jev)

    assert scores is None
    assert skip_reason == "state_too_large"
    # full summary, 1-sentence variant, title-only: 3 attempts for a 2-sentence summary
    assert len(jev.requests) == 3


def test_a_non_too_large_error_is_not_caught():
    busy = JevError("HTTP 503", retryable=True, too_large=False)
    jev = FakeJev(busy)
    with pytest.raises(JevError):
        score_item(item(), [], config(), jev)


# --- jobs_for_ticker ---


def test_jobs_for_ticker_orders_oldest_first_with_growing_history():
    items = [
        item(id=1, title="first", published_at=datetime(2026, 10, 1, tzinfo=UTC)),
        item(id=2, title="second", published_at=datetime(2026, 10, 3, tzinfo=UTC)),
        item(id=3, title="third", published_at=datetime(2026, 10, 2, tzinfo=UTC)),  # out of order
    ]
    jobs = jobs_for_ticker(items)
    assert [j[0].id for j in jobs] == [1, 3, 2]  # by published_at, not input order
    assert [j[1] for j in jobs] == [[], ["first"], ["first", "third"]]


def test_jobs_for_ticker_caps_the_history_window():
    items = [item(id=i, title=f"t{i}", published_at=datetime(2026, 10, i, tzinfo=UTC)) for i in range(1, 6)]
    jobs = jobs_for_ticker(items, history_pool=2)
    assert [j[1] for j in jobs] == [[], ["t1"], ["t1", "t2"], ["t2", "t3"], ["t3", "t4"]]


# --- score_all ---


class FakeNewsStore:
    """Stands in for NewsStore: one fixed item list per ticker, point-in-time ignored."""

    def __init__(self, items_by_ticker: dict[str, list[StoredNews]]):
        self._items = items_by_ticker

    def query(self, ticker, before, dedup=True):
        return list(self._items.get(ticker, []))


@pytest.fixture
def scores(tmp_path):
    with ScoreStore(tmp_path / "scores.db") as scores:
        yield scores


def test_score_all_scores_every_new_item(scores):
    news = FakeNewsStore({"NVDA": [item(id=1), item(id=2, title="Nvidia announces buyback")]})
    jev = FakeJev(full_answer(), full_answer())
    results = score_all(["NVDA"], news, scores, config(), jev)

    assert {r.status for r in results} == {"scored"}
    assert scores.done(1, "test") and scores.done(2, "test")


def test_score_all_skips_an_item_already_done(scores):
    scores.put(1, "test", "deadbeef", score_item(item(), [], config(), FakeJev(full_answer()))[0])
    news = FakeNewsStore({"NVDA": [item(id=1)]})
    results = score_all(["NVDA"], news, scores, config(), FakeJev())

    assert [r.status for r in results] == ["done"]


def test_score_all_dedupes_an_item_seen_under_two_tickers(scores):
    shared = item(id=1, tickers=["NVDA", "AMD"])
    news = FakeNewsStore({"NVDA": [shared], "AMD": [shared]})
    jev = FakeJev(full_answer())
    results = score_all(["NVDA", "AMD"], news, scores, config(), jev)

    assert len(results) == 1
    assert len(jev.requests) == 1


def test_score_all_respects_limit(scores):
    news = FakeNewsStore({"NVDA": [item(id=1), item(id=2, title="Nvidia announces buyback")]})
    jev = FakeJev(full_answer())
    results = score_all(["NVDA"], news, scores, config(), jev, limit=1)

    assert len(results) == 1


def test_score_all_reports_an_error_without_persisting_it(scores):
    news = FakeNewsStore({"NVDA": [item(id=1)]})
    jev = FakeJev(JevError("HTTP 503", retryable=True))
    results = score_all(["NVDA"], news, scores, config(), jev)

    assert results[0].status == "error"
    assert scores.done(1, "test") is False


def test_score_all_persists_a_skip_without_raising(scores):
    too_large = JevError("HTTP 422", too_large=True)
    news = FakeNewsStore({"NVDA": [item(id=1)]})
    jev = FakeJev(*[too_large] * 10)
    results = score_all(["NVDA"], news, scores, config(), jev)

    assert results[0].status == "skipped"
    assert results[0].detail == "state_too_large"
    assert scores.done(1, "test") is True
    assert scores.get(1, "test") is None
