import pytest

from sluice.scoring.store import ItemScores, ScoreStore


@pytest.fixture
def store(tmp_path):
    with ScoreStore(tmp_path / "scores.db") as store:
        yield store


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


def test_a_scored_item_round_trips(store):
    store.put(1, "v1", "abcd1234", scores())
    assert store.get(1, "v1") == scores()


def test_an_unknown_item_returns_none(store):
    assert store.get(1, "v1") is None


def test_done_is_false_until_a_row_exists(store):
    assert store.done(1, "v1") is False
    store.put(1, "v1", "abcd1234", scores())
    assert store.done(1, "v1") is True


def test_a_skipped_item_counts_as_done_but_has_no_scores(store):
    store.skip(2, "v1", "abcd1234", "state_too_large")
    assert store.done(2, "v1") is True
    assert store.get(2, "v1") is None


def test_different_config_versions_are_independent(store):
    store.put(1, "v1", "abcd1234", scores(relevance=0.9))
    assert store.done(1, "v2") is False
    store.put(1, "v2", "efgh5678", scores(relevance=0.1))
    assert store.get(1, "v1").relevance == 0.9
    assert store.get(1, "v2").relevance == 0.1


def test_putting_the_same_key_twice_keeps_the_first(store):
    store.put(1, "v1", "abcd1234", scores(relevance=0.9))
    store.put(1, "v1", "abcd1234", scores(relevance=0.1))
    assert store.get(1, "v1").relevance == 0.9


def test_len_counts_every_row_scored_and_skipped(store):
    store.put(1, "v1", "abcd1234", scores())
    store.skip(2, "v1", "abcd1234", "state_too_large")
    assert len(store) == 2


def test_reopening_the_same_file_keeps_the_rows(tmp_path):
    with ScoreStore(tmp_path / "scores.db") as store:
        store.put(1, "v1", "abcd1234", scores())
    with ScoreStore(tmp_path / "scores.db") as reopened:
        assert reopened.get(1, "v1") == scores()
