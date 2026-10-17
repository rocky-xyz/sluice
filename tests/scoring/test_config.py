import pytest
from pydantic import ValidationError

from sluice.scoring.config import QuestionSpec, load_scoring_config


def test_v1_loads_the_five_questions():
    config = load_scoring_config("v1")
    assert config.version == "v1"
    assert config.model == "jev-1.13.0"
    assert config.novelty_reference_k > 0
    assert set(config.questions) == {"relevance", "stance", "novelty", "rumour", "event_type"}
    assert config.questions["relevance"].type == "noul"
    assert config.questions["stance"].type == "choice"
    assert set(config.questions["stance"].criteria) == {"bullish", "bearish", "neutral"}


def test_editing_the_file_in_place_changes_the_digest(tmp_path):
    path = tmp_path / "scoring"
    path.mkdir()
    (path / "v9.toml").write_text(
        'model = "jev-1.13.0"\n'
        "novelty_reference_k = 15\n"
        '[questions.relevance]\ntype = "noul"\ninstructions = "is it relevant?"\n'
    )
    first = load_scoring_config("v9", root=tmp_path)

    (path / "v9.toml").write_text(
        'model = "jev-1.13.0"\n'
        "novelty_reference_k = 20\n"
        '[questions.relevance]\ntype = "noul"\ninstructions = "is it relevant?"\n'
    )
    second = load_scoring_config("v9", root=tmp_path)
    assert first.digest != second.digest


def test_a_choice_question_without_criteria_is_rejected():
    with pytest.raises(ValidationError):
        QuestionSpec(type="choice", instructions="pick one")


def test_a_noul_question_with_criteria_is_rejected():
    with pytest.raises(ValidationError):
        QuestionSpec(type="noul", instructions="is it true?", criteria={"true": "x"})
