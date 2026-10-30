import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from rich.console import Console

from sluice.agents import configure as cfg
from sluice.agents.config import ANALYST_KEYS, load_config
from sluice.agents.configure import SUGGESTED, Choices
from sluice.session import flow
from sluice.session.flow import Plan

SHIPPED = Path(__file__).parents[2] / "configs"
NOW = datetime(2026, 10, 30, 21, 0, tzinfo=UTC)  # Friday, after the close


@pytest.fixture
def root(tmp_path):
    (tmp_path / "pipeline").mkdir()
    for version in ("v1", "v2", "v3"):
        shutil.copy(SHIPPED / "pipeline" / f"{version}.toml", tmp_path / "pipeline")
    return tmp_path


@pytest.mark.parametrize("text, ok", [
    ("NVDA", True), ("nvda ", True), ("BRK.B", True), ("BF-B", True),
    ("", False), ("123", False), ("TOOLONGTICKER", False), ("NV DA", False),
])  # fmt: skip
def test_tickers(text, ok):
    assert (flow.ticker_problem(text) is None) is ok


def test_the_universe_is_the_study_s():
    assert not flow.outside_universe("nvda")
    assert flow.outside_universe("PLTR")


@pytest.mark.parametrize("text, problem", [
    ("2026-10-30", None),
    ("2026-10-31", "not a trading day"),  # Saturday
    ("2026-12-25", "not a trading day"),
    ("2026-11-02", "has not closed yet"),
    ("30/10/2026", "YYYY-MM-DD"),
])  # fmt: skip
def test_backtest_days(text, problem):
    found = flow.date_problem(text, NOW)
    assert found is None if problem is None else problem in found


def test_a_day_closing_later_today_is_not_offered_yet():
    morning = datetime(2026, 10, 30, 14, 0, tzinfo=UTC)  # 10:00 in New York
    assert "has not closed yet" in flow.date_problem("2026-10-30", morning)


def test_each_config_has_a_one_line_summary(root):
    assert flow.config_line(load_config("v1", root)) == (
        "market+news+fundamentals · 11 indicators · 1 round"
    )
    assert flow.config_line(load_config("v2", root)).endswith("weighted news")
    assert "tools" in flow.config_line(load_config("v3", root))
    custom = Choices(analysts=("news",), indicators=(), debate_rounds=0, weighted=False)
    text = cfg.build_config_text(
        (root / "pipeline" / "v1.toml").read_text(), load_config("v1", root), custom, "solo", NOW
    )
    cfg.write_config("solo", text, root)
    assert flow.config_line(load_config("solo", root)) == "news · no debate"


def test_v1_comes_first(root):
    (root / "pipeline" / "alpha.toml").write_text((root / "pipeline" / "v1.toml").read_text())
    assert flow.config_names(root) == ["v1", "alpha", "v2", "v3"]


def test_the_last_answers_come_back_without_what_is_gone(root, tmp_path):
    path = tmp_path / "last.json"
    flow.save_last(Plan("NVDA", "backtest", date(2026, 9, 30), "v2"), path)
    assert flow.load_last(path, root) == {
        "ticker": "NVDA", "mode": "backtest", "day": "2026-09-30", "config": "v2",
    }  # fmt: skip
    path.write_text(json.dumps({"ticker": "1", "mode": "later", "day": "soon", "config": "v9"}))
    assert flow.load_last(path, root) == {}
    path.write_text("{broken")
    assert flow.load_last(path, root) == {}
    assert flow.load_last(tmp_path / "missing.json", root) == {}


def test_an_ad_hoc_plan_is_saved_without_a_day(tmp_path):
    path = tmp_path / "last.json"
    flow.save_last(Plan("AMD", "adhoc", None, "v1"), path)
    assert json.loads(path.read_text())["day"] is None


def test_new_writes_the_config_and_returns_its_name(root, monkeypatch, tmp_path):
    choices = Choices(analysts=ANALYST_KEYS, indicators=SUGGESTED, debate_rounds=3, weighted=False)
    monkeypatch.setattr(cfg, "interview", lambda r: (load_config("v1", r), choices, "deep"))
    monkeypatch.setattr(cfg, "confirm", lambda message: True)
    monkeypatch.setattr(cfg, "LAST_CHOICES", tmp_path / "configure-last.json")
    monkeypatch.setattr(cfg.save_last, "__defaults__", (tmp_path / "configure-last.json",))
    console = Console(record=True, width=100)
    assert flow.new_config(console, root) == "deep"
    assert load_config("deep", root).debate_rounds == 3
    assert "debate rounds: 1 -> 3" in console.export_text()


def test_new_writes_nothing_when_cancelled(root, monkeypatch):
    monkeypatch.setattr(cfg, "interview", lambda r: None)
    assert flow.new_config(Console(), root) is None
    choices = Choices(analysts=ANALYST_KEYS, indicators=SUGGESTED, debate_rounds=3, weighted=False)
    monkeypatch.setattr(cfg, "interview", lambda r: (load_config("v1", r), choices, "deep"))
    monkeypatch.setattr(cfg, "confirm", lambda message: False)
    assert flow.new_config(Console(), root) is None
    assert not (root / "pipeline" / "deep.toml").exists()


def test_the_welcome_and_the_summary_draw(root):
    console = Console(record=True, width=100)
    flow.welcome(console)
    plan = Plan("NVDA", "backtest", date(2026, 9, 30), "v1")
    console.print(flow.summary(plan, load_config("v1", root)))
    text = console.export_text()
    assert "not trading advice" in text and "backtest, the close of 2026-09-30" in text
