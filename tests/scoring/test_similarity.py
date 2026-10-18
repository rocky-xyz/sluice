import pytest

from sluice.scoring.similarity import most_similar, similarity, split_sentences


def test_identical_text_is_maximally_similar():
    assert similarity("Nvidia beats earnings", "Nvidia beats earnings") == pytest.approx(1.0)


def test_unrelated_text_is_not_similar():
    assert similarity("Nvidia beats earnings", "Coca-Cola launches new flavor") == 0.0


def test_empty_text_is_not_similar_to_anything():
    assert similarity("", "Nvidia beats earnings") == 0.0
    assert similarity("Nvidia beats earnings", "") == 0.0
    assert similarity("", "") == 0.0


def test_shared_words_score_between_zero_and_one():
    score = similarity("Nvidia beats earnings estimates", "Nvidia misses earnings estimates")
    assert 0.0 < score < 1.0


def test_is_case_insensitive():
    assert similarity("NVIDIA", "nvidia") == pytest.approx(1.0)


def test_most_similar_ranks_by_closeness_to_the_query():
    candidates = [
        "Coca-Cola launches new flavor",
        "Nvidia beats earnings estimates",
        "Nvidia misses earnings estimates",
    ]
    assert most_similar("Nvidia beats earnings", candidates, 2) == [
        "Nvidia beats earnings estimates",
        "Nvidia misses earnings estimates",
    ]


def test_most_similar_caps_at_the_number_of_candidates():
    assert len(most_similar("x", ["a", "b"], 5)) == 2


def test_most_similar_with_k_zero_is_empty():
    assert most_similar("x", ["a", "b"], 0) == []


def test_split_sentences_splits_on_terminal_punctuation():
    text = "Revenue rose 10%. Guidance was cut! Investors reacted. Is that bad?"
    assert split_sentences(text) == [
        "Revenue rose 10%.",
        "Guidance was cut!",
        "Investors reacted.",
        "Is that bad?",
    ]


def test_split_sentences_handles_a_single_sentence():
    assert split_sentences("One sentence with no terminal punctuation") == [
        "One sentence with no terminal punctuation"
    ]


def test_split_sentences_drops_empty_pieces():
    assert split_sentences("  ") == []
