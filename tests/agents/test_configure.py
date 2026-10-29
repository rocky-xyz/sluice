import json
import shutil
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sluice.agents import indicators
from sluice.agents.config import ANALYST_KEYS, load_config
from sluice.agents.configure import (
    GENERIC_MARKET_PROMPT,
    GROUPS,
    SUGGESTED,
    Choices,
    build_config_text,
    changes,
    load_last,
    name_problem,
    save_last,
    write_config,
)

SHIPPED = Path(__file__).parents[2] / "configs"
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


@pytest.fixture
def root(tmp_path):
    """A configs folder holding copies of the shipped pipeline configs."""
    (tmp_path / "pipeline").mkdir()
    for version in ("v1", "v2", "v3"):
        shutil.copy(SHIPPED / "pipeline" / f"{version}.toml", tmp_path / "pipeline")
    return tmp_path


def source(root, version: str) -> str:
    return (root / "pipeline" / f"{version}.toml").read_text()


def build(root, version: str, choices: Choices, name: str = "custom") -> str:
    return build_config_text(source(root, version), load_config(version, root), choices, name, NOW)


def loaded(root, text: str, name: str = "custom"):
    write_config(name, text, root)
    return load_config(name, root)


SAME_AS_V1 = Choices(
    analysts=ANALYST_KEYS, indicators=indicators.NAMES, debate_rounds=1, weighted=False
)
MARKET_NEWS = Choices(
    analysts=("market", "news"), indicators=SUGGESTED, debate_rounds=3, weighted=False
)


def test_the_checkbox_groups_cover_every_indicator_once():
    names = [name for _, group in GROUPS for name in group]
    assert sorted(names) == sorted(indicators.ALL) and len(names) == len(set(names))
    assert set(SUGGESTED) <= set(indicators.ALL)


def test_choosing_what_the_base_has_changes_only_the_header(root):
    text = build(root, "v1", SAME_AS_V1)
    v1, config = load_config("v1", root), loaded(root, text)
    assert config.model_dump(exclude={"version", "digest"}) == v1.model_dump(
        exclude={"version", "digest"}
    )
    body = [line for line in text.splitlines() if not line.startswith("#")]
    assert body == [line for line in source(root, "v1").splitlines() if not line.startswith("#")]
    assert "#   none" in text


def test_the_choices_are_written_and_everything_else_is_kept(root):
    text = build(root, "v1", MARKET_NEWS)
    v1, config = load_config("v1", root), loaded(root, text)

    assert config.selected_analysts == ("market", "news")
    assert config.prices.selected_indicators == SUGGESTED
    assert config.debate_rounds == 3
    assert config.prompts["market_analyst"] == GENERIC_MARKET_PROMPT
    others = {role: prompt for role, prompt in v1.prompts.items() if role != "market_analyst"}
    assert {role: config.prompts[role] for role in others} == others
    assert (config.models, config.news, config.fundamentals) == (v1.models, v1.news, v1.fundamentals)
    assert "lookback_days = 14  # one decision period" in text  # comments stay
    assert "#   analysts: market, news" in text and "#   debate rounds: 1 -> 3" in text


def test_the_generic_prompt_names_no_indicator_but_the_id_example():
    named = [name for name in indicators.ALL if name in GENERIC_MARKET_PROMPT]
    assert named == ["ret_20d"]  # only in "for example [P:ret_20d]"
    assert "[P:ret_20d]" in GENERIC_MARKET_PROMPT and "\\" not in GENERIC_MARKET_PROMPT


def test_the_same_indicators_keep_the_base_prompt(root):
    choices = SAME_AS_V1.model_copy(update={"analysts": ("market",), "debate_rounds": 0})
    config = loaded(root, build(root, "v1", choices))
    assert config.prompts["market_analyst"] == load_config("v1", root).prompts["market_analyst"]
    assert config.prices.indicators is None


def test_leaving_the_market_analyst_out_drops_its_indicators(root):
    first = loaded(root, build(root, "v1", MARKET_NEWS), "first")
    choices = Choices(analysts=("news",), indicators=(), debate_rounds=1, weighted=False)
    text = build_config_text(source(root, "first"), first, choices, "second", NOW)
    second = loaded(root, text, "second")
    assert second.prices.indicators is None and second.selected_analysts == ("news",)


def test_a_generated_config_can_be_the_base_without_doubling_its_keys(root):
    first = loaded(root, build(root, "v1", MARKET_NEWS), "first")
    choices = MARKET_NEWS.model_copy(
        update={"analysts": ("fundamentals", "market"), "indicators": ("rsi_14", "kdj_k")}
    )
    text = build_config_text(source(root, "first"), first, choices, "second", NOW)
    assert text.count("\nanalysts = ") == 1 and text.count("\nindicators = ") == 1
    second = loaded(root, text, "second")
    assert second.selected_analysts == ("market", "fundamentals")
    assert second.prices.selected_indicators == ("rsi_14", "kdj_k")


def test_all_three_analysts_leave_the_key_out(root):
    first = loaded(root, build(root, "v1", MARKET_NEWS), "first")
    choices = MARKET_NEWS.model_copy(update={"analysts": ANALYST_KEYS})
    text = build_config_text(source(root, "first"), first, choices, "second", NOW)
    assert "\nanalysts = " not in text
    assert loaded(root, text, "second").analysts is None


def test_weighting_is_turned_on_and_off_in_the_news_section(root):
    on = build(root, "v1", SAME_AS_V1.model_copy(update={"weighted": True}))
    assert tomllib.loads(on)["news"]["weighted"] is True
    assert "#   weighted news: on" in on

    off = build(root, "v2", SAME_AS_V1)  # v2 weights its news
    assert "weighted" not in tomllib.loads(off)["news"]
    assert tomllib.loads(off)["news"]["min_relevance"] == 0.5  # the rest of v2 stays

    no_news = Choices(analysts=("market",), indicators=indicators.NAMES, debate_rounds=1,
                      weighted=True)  # fmt: skip
    assert "weighted" not in tomllib.loads(build(root, "v2", no_news))["news"]


def test_a_config_with_tools_keeps_them_and_takes_a_new_depth(root):
    v3 = load_config("v3", root)
    choices = Choices(analysts=ANALYST_KEYS, indicators=v3.prices.selected_indicators,
                      debate_rounds=5, weighted=False)  # fmt: skip
    config = loaded(root, build(root, "v3", choices))
    assert config.tools == v3.tools and config.debate_rounds == 5


def test_choices_that_do_not_fit_are_not_written(root):
    choices = Choices(analysts=("market", "news"), indicators=("rsi_14",), debate_rounds=1,
                      weighted=False)  # fmt: skip
    with pytest.raises(ValueError, match=r"\[tools\]"):
        build(root, "v3", choices)


def test_the_header_says_where_the_config_came_from(root):
    text = build(root, "v1", MARKET_NEWS)
    v1 = load_config("v1", root)
    assert text.startswith("# Pipeline config custom: written by `sluice configure` on 2026-10-29")
    assert f"# from v1-{v1.digest}. Changes from it:" in text
    assert "#   indicators added: macd_hist_pct, bb_pctb_20, bb_width_20, atr_14_pct, range_250d" in text


def test_changes_are_listed_against_the_base():
    v2 = load_config("v2", SHIPPED)
    choices = Choices(analysts=("market",), indicators=("rsi_14", "kdj_k"), debate_rounds=0,
                      weighted=False)  # fmt: skip
    listed = changes(v2, choices)
    assert listed[0] == "analysts: market"
    assert listed[1] == "indicators added: kdj_k"
    assert listed[2].startswith("indicators removed: ret_5d, ret_20d")
    assert listed[3:] == ["debate rounds: 1 -> 0", "weighted news: off"]


@pytest.mark.parametrize("name, problem", [
    ("Market", "lower-case"),
    ("market news", "lower-case"),
    ("market--news", "lower-case"),
    ("", "lower-case"),
    ("v4", "evolution loop"),
    ("v1", "evolution loop"),
])  # fmt: skip
def test_names_that_cannot_be_used(root, name, problem):
    assert problem in name_problem(name, root)


def test_an_existing_name_is_refused_and_never_overwritten(root):
    text = build(root, "v1", MARKET_NEWS, "market-news")
    write_config("market-news", text, root)
    assert "exists" in name_problem("market-news", root)
    with pytest.raises(FileExistsError):
        write_config("market-news", "overwritten\n", root)
    assert (root / "pipeline" / "market-news.toml").read_text() == text
    assert name_problem("market-news-2", root) is None


def test_the_last_choices_come_back_without_what_is_no_longer_offered(root, tmp_path):
    path = tmp_path / "last.json"
    save_last("v2", MARKET_NEWS, path)
    assert load_last(path, root) == {
        "base": "v2",
        "analysts": ["market", "news"],
        "indicators": list(SUGGESTED),
        "debate_rounds": 3,
        "weighted": False,
    }

    path.write_text(json.dumps({
        "base": "gone", "analysts": ["news", "macro"], "indicators": ["sma_7"],
        "debate_rounds": True, "weighted": "yes",
    }))  # fmt: skip
    assert load_last(path, root) == {"analysts": ["news"]}
    path.write_text(json.dumps({"debate_rounds": [1]}))
    assert load_last(path, root) == {}


def test_no_or_unreadable_last_choices_mean_none(tmp_path):
    assert load_last(tmp_path / "missing.json") == {}
    (tmp_path / "bad.json").write_text("{not json")
    assert load_last(tmp_path / "bad.json") == {}
    (tmp_path / "list.json").write_text("[1, 2]")
    assert load_last(tmp_path / "list.json") == {}
