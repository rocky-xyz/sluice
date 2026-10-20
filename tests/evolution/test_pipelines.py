import shutil
import tomllib
from pathlib import Path

import pytest

from sluice.agents.config import load_config
from sluice.evolution.pipelines import (
    RULES,
    candidate,
    champion,
    check,
    layer,
    load_champion,
    params,
    params_key,
    set_champion,
    version_with,
    write_version,
)

CONFIGS = Path(__file__).parents[2] / "configs"
V3 = load_config("v3", CONFIGS)
NEWS = V3.prompts["news_analyst"]
SHORTER = NEWS.replace("At most 400 words.", 'At most 300 words. Say "why" \\ briefly.')


@pytest.fixture
def root(tmp_path):
    (tmp_path / "pipeline").mkdir()
    for name in ("v1.toml", "v3.toml", "CHAMPION"):
        shutil.copy(CONFIGS / "pipeline" / name, tmp_path / "pipeline" / name)
    return tmp_path


def test_every_rule_exists_in_v3_within_its_range():
    for key, bound in RULES.items():
        assert key in params(V3), key
        if bound.kind is not bool:
            assert bound.low <= params(V3)[key] <= bound.high, key


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({}, "no change proposed"),
        ({"tools.max_calls": 8, "prompts.news_analyst": SHORTER}, "one layer at a time"),
        ({"prompts.news_analyst": SHORTER, "prompts.bull": "Argue."}, "one role's prompt"),
        ({"prompts.trader": "Trade."}, "no role 'trader'"),
        ({"prompts.news_analyst": "   "}, "non-empty text"),
        ({"prompts.news_analyst": "x'''y"}, "may not contain '''"),
        ({"prompts.news_analyst": NEWS * 2}, "at most"),
        ({"prompts.news_analyst": "\n" + NEWS + "  "}, "unchanged"),
        ({"models.bull": "deepseek-v4-pro"}, "not a rule"),
        ({"temperature": 0.7}, "not a rule"),
        ({"fundamentals.lag_days": 0}, "not a rule"),
        ({"tools.max_calls": 11}, "outside"),
        ({"tools.max_calls": 2.5}, "expected int"),
        ({"tools.pass_reasoning": 1}, "expected bool"),
        ({"debate_rounds": 1}, "already"),
    ],
)
def test_check_refuses(changes, problem):
    problems = check(V3, changes)
    assert len(problems) == 1 and problem in problems[0]


def test_rules_need_a_champion_with_tools():
    assert "no such setting" in check(load_config("v1", CONFIGS), {"tools.max_calls": 4})[0]


def test_layers():
    assert layer({"tools.max_calls": 4, "debate_rounds": 0}) == "rules"
    assert layer({"prompts.bull": "x"}) == "prompt"
    assert layer({"prompts.bull": "x", "debate_rounds": 0}) is None


def test_a_candidate_changes_one_thing_and_its_key_ignores_spacing():
    rules = candidate(V3, {"tools.max_calls": 4, "news.lookback_days": 21})
    assert (rules.tools.max_calls, rules.news.lookback_days) == (4, 21)
    assert rules.prompts == V3.prompts and rules.digest == params_key(rules)
    prompt = candidate(V3, {"prompts.news_analyst": SHORTER})
    assert prompt.prompts["news_analyst"] == SHORTER.strip() + "\n"
    assert candidate(V3, {"prompts.news_analyst": "\n\n" + SHORTER}).digest == prompt.digest


def test_a_new_prompt_is_written_as_a_literal_string_and_nothing_else_moves(root):
    before = (root / "pipeline" / "v3.toml").read_text()
    v4 = write_version(V3, {"prompts.news_analyst": SHORTER}, root)
    assert v4.version == "v4"
    assert params(v4) == params(candidate(V3, {"prompts.news_analyst": SHORTER}))
    assert "\\ briefly" in v4.prompts["news_analyst"]  # the backslash survived as written
    text = (root / "pipeline" / "v4.toml").read_text()
    assert "news_analyst = '''" in text
    assert text.startswith("# Pipeline config, version 4: v3 with prompts.news_analyst rewritten")
    old_body = before.split("[prompts]")[0].splitlines()
    new_body = text.split("[prompts]")[0].splitlines()
    assert [l for l in new_body if not l.startswith("#")] == [
        l for l in old_body if not l.startswith("#")
    ]
    assert (root / "pipeline" / "v3.toml").read_text() == before


def test_rules_keep_their_comments_and_versions_go_up(root):
    v4 = write_version(V3, {"tools.max_calls": 4, "debate_rounds": 0}, root)
    text = (root / "pipeline" / "v4.toml").read_text()
    assert "max_calls = 4  # per analyst, for its report" in text
    assert "debate_rounds = 0  # one round is" in text
    assert (
        tomllib.loads(text)["prompts"]
        == tomllib.loads((root / "pipeline" / "v3.toml").read_text())["prompts"]
    )
    v5 = write_version(v4, {"prompts.news_analyst": SHORTER}, root)
    assert (v5.version, v5.tools.max_calls, v5.debate_rounds) == ("v5", 4, 0)
    # a literal-string prompt is found and kept when the next version is written
    v6 = write_version(v5, {"tools.max_calls": 5}, root)
    assert v6.prompts == v5.prompts


def test_write_version_refuses_bad_proposals_and_edited_champions(root):
    with pytest.raises(ValueError, match="not a rule"):
        write_version(V3, {"temperature": 0.5}, root)
    stale = V3.model_copy(update={"digest": "00000000"})
    with pytest.raises(ValueError, match="has changed"):
        write_version(stale, {"tools.max_calls": 4}, root)
    assert not (root / "pipeline" / "v4.toml").exists()


def test_champion_rollback_and_finding_a_version(root):
    assert champion(root) == "v3"
    v4 = write_version(V3, {"tools.max_calls": 4}, root)
    set_champion("v4", root)
    assert load_champion(root).tools.max_calls == 4
    assert version_with(params_key(v4), root) == "v4"
    assert version_with(params_key(V3), root) == "v3"
    set_champion("v3", root)
    with pytest.raises(FileNotFoundError):
        set_champion("v9", root)
    assert champion(CONFIGS) == "v3"
