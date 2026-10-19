import shutil
from pathlib import Path

import pytest

from sluice.evolution.library import (
    SPACE,
    candidate,
    champion,
    check,
    load_champion,
    params,
    params_key,
    render,
    set_champion,
    write_version,
)
from sluice.sizing.config import load_sizing_config

CONFIGS = Path(__file__).parents[2] / "configs"
V1 = load_sizing_config("v1", CONFIGS)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "sizing").mkdir()
    for name in ("v1.toml", "CHAMPION"):
        shutil.copy(CONFIGS / "sizing" / name, tmp_path / "sizing" / name)
    return tmp_path


def test_v1_is_unchanged_by_the_new_key():
    assert V1.digest == "1bedf5b4"
    assert V1.stocks.vol_scaling is True


def test_the_search_space_covers_every_parameter_but_the_costly_ones():
    left_out = set(params(V1)) - set(SPACE)
    assert left_out == {"inputs.scoring_config", "inputs.report_spec", "exposure.benchmark"}
    assert set(SPACE) <= set(params(V1))


def test_every_v1_value_is_inside_its_range():
    for key, bound in SPACE.items():
        if bound.kind is not bool:
            assert bound.low <= params(V1)[key] <= bound.high, key


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({}, "no change proposed"),
        ({"exposure.benchmark": "QQQ"}, "not in the search space"),
        ({"inputs.report_spec": "reports-v2"}, "not in the search space"),
        ({"cost": 0.0}, "not in the search space"),
        ({"stocks.vol_lookback": 20.5}, "expected int"),
        ({"stocks.vol_lookback": True}, "expected int"),
        ({"stocks.vol_scaling": 0}, "expected bool"),
        ({"k": "0.3"}, "expected float"),
        ({"k": float("nan")}, "expected float"),
        ({"k": 2.5}, "outside"),
        ({"coefficients.news": -0.5}, "outside"),
        ({"stocks.max_weight": 0.05}, "outside"),
        ({"k": 0.5}, "already"),
    ],
)
def test_check_refuses(changes, problem):
    problems = check(V1, changes)
    assert len(problems) == 1
    assert problem in problems[0]


def test_check_reports_every_problem():
    assert len(check(V1, {"k": 3.0, "cost": 1.0, "stocks.vol_scaling": False})) == 2


def test_a_candidate_applies_the_changes_in_memory(root):
    config = candidate(V1, {"k": 0, "stocks.vol_scaling": False})
    assert (config.version, config.k, config.stocks.vol_scaling) == ("candidate", 0.0, False)
    assert config.coefficients == V1.coefficients
    assert config.digest == params_key(config)
    assert sorted(p.name for p in (root / "sizing").iterdir()) == ["CHAMPION", "v1.toml"]
    with pytest.raises(ValueError, match="outside"):
        candidate(V1, {"k": 9.0})


def test_the_parameter_key_ignores_comments_and_int_versus_float():
    assert candidate(V1, {"k": 1}).digest == candidate(V1, {"k": 1.0}).digest
    assert candidate(V1, {"k": 1.0}).digest != params_key(V1)


def test_render_replaces_values_keeps_comments_and_adds_missing_keys():
    source = (
        "# old header\n# second line\n\nk = 0.5  # tilt\n\n"
        "[stocks]\nvol_lookback = 60  # days\nmax_weight = 0.20\n\n[exposure]\nfloor = 0.5\n"
    )
    text = render(source, {"k": 1, "stocks.vol_scaling": False, "exposure.floor": 0.6}, ["new"])
    assert text == (
        "# new\n\nk = 1.0  # tilt\n\n"
        "[stocks]\nvol_lookback = 60  # days\nmax_weight = 0.20\n"
        "vol_scaling = false  # added by the evolution loop\n\n[exposure]\nfloor = 0.6\n"
    )


def test_write_version_writes_the_next_file_and_keeps_the_old_one(root):
    before = (root / "sizing" / "v1.toml").read_text()
    v2 = write_version(V1, {"k": 0.0, "stocks.vol_scaling": False}, root)
    assert (v2.version, v2.k, v2.stocks.vol_scaling) == ("v2", 0.0, False)
    assert params(v2) == params(candidate(V1, {"k": 0.0, "stocks.vol_scaling": False}))
    assert (root / "sizing" / "v1.toml").read_text() == before
    text = (root / "sizing" / "v2.toml").read_text()
    assert text.startswith("# Sizing config, version 2: v1 with k 0.5 → 0.0")
    assert "# tilt strength" in text  # v1's comments survive
    # only the header and the changed lines differ from v1
    changed = set(text.splitlines()) ^ set(before.splitlines())
    assert {line for line in changed if not line.startswith("#")} == {
        "k = 0.5  # tilt strength: weight ∝ max(0, 1 + k·s), s the combined cross-sectional z-score",
        "k = 0.0  # tilt strength: weight ∝ max(0, 1 + k·s), s the combined cross-sectional z-score",
        "vol_scaling = false  # added by the evolution loop",
    }
    v3 = write_version(v2, {"exposure.floor": 0.7}, root)
    assert v3.version == "v3"
    assert (v3.k, v3.stocks.vol_scaling, v3.exposure.floor) == (0.0, False, 0.7)


def test_write_version_never_overwrites_and_refuses_an_edited_champion(root):
    stale = V1.model_copy(update={"digest": "00000000"})
    with pytest.raises(ValueError, match="has changed"):
        write_version(stale, {"k": 0.0}, root)
    (root / "sizing" / "v2.toml").write_text("")  # a stray file takes the number
    assert write_version(V1, {"k": 0.0}, root).version == "v3"
    assert (root / "sizing" / "v2.toml").read_text() == ""


def test_write_version_refuses_a_bad_proposal_and_writes_nothing(root):
    with pytest.raises(ValueError, match="not in the search space"):
        write_version(V1, {"exposure.benchmark": "QQQ"}, root)
    assert not (root / "sizing" / "v2.toml").exists()


def test_champion_and_rollback(root):
    assert champion(root) == "v1"
    write_version(V1, {"k": 0.0}, root)
    set_champion("v2", root)
    assert load_champion(root).k == 0.0
    set_champion("v1", root)  # rollback
    assert load_champion(root).digest == V1.digest
    with pytest.raises(FileNotFoundError):
        set_champion("v9", root)
    assert champion(root) == "v1"


def test_the_shipped_champion_is_v1():
    assert champion(CONFIGS) == "v1"
