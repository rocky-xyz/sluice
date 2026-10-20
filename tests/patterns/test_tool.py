import json
from datetime import date, timedelta

import pytest

from sluice.agents.config import ToolsConfig
from sluice.agents.state import cited_refs
from sluice.agents.tools import ROLE_TOOLS, ToolContext, call
from sluice.fundamentals.store import FundamentalsStore
from sluice.market_calendar import close_of
from sluice.news.store import NewsStore
from sluice.patterns.study import Row, StoredStats
from sluice.patterns.tool import PatternSource
from sluice.prices.store import PriceStore
from tests.agents.test_pipeline import CONFIG
from tests.patterns.test_detect import RULES, drift, wedge_bars

AFTER = [(21.6, 21.5), (21.7, 21.55), (21.8, 21.65), (21.9, 21.7)]


def row(stage, horizon, mean, low, high, n=31) -> Row:
    return Row("wedge", stage, horizon, n, mean, low, high, 0.4, -0.001, mean + 0.001, n < 30)


@pytest.fixture
def world(tmp_path):
    bars, t = wedge_bars(AFTER)
    store = PriceStore(tmp_path / "ohlc.db")
    store.replace("AAA", bars)
    store.replace("SPY", drift([b.day for b in bars], 400, 0.001))
    later = bars[t].day + timedelta(days=1)  # after the breakout, before the last bar
    tables = [
        StoredStats(date(2025, 6, 1), "v1-x", "test", [row("confirmed", 10, -0.02, -0.03, -0.001)]),
        StoredStats(later, "v1-x", "test", [row("confirmed", 10, 0.01, -0.01, 0.03)]),
    ]
    source = PatternSource(store, tables, RULES)
    with NewsStore(tmp_path / "n.db") as news:
        yield bars, t, source, news, tmp_path


def context(world, day):
    _, _, source, news, tmp_path = world
    config = CONFIG.model_copy(update={"tools": ToolsConfig()})
    return ToolContext(
        "AAA", close_of(day), news, source.store, FundamentalsStore(tmp_path / "s"), config, source
    )


def test_it_is_offered_to_no_analyst_yet():
    assert all("get_patterns" not in tools for tools in ROLE_TOOLS.values())


def test_without_a_pattern_source_it_says_so(world):
    ctx = context(world, world[0][-1].day)
    bare = ToolContext(ctx.ticker, ctx.decision_time, ctx.news, ctx.prices, ctx.statements,
                       ctx.config)  # fmt: skip
    result = call(bare, "get_patterns", "{}")
    assert "not available" in result.content and result.refs == []


def test_events_and_their_history_as_of_the_decision_time(world):
    bars, t, *_ = world
    result = call(context(world, bars[-1].day), "get_patterns", json.dumps({"days": 10}))
    assert [r.split("_")[2] for r in result.refs] == ["candidate", "confirmed"]
    assert set(cited_refs(result.content)) == set(result.refs)  # every id parses as a citation
    assert "confirmed by hold" in result.content
    later = bars[t].day + timedelta(days=1)
    assert f"history from the table of {later}" in result.content  # the latest one by then
    assert "+1.10% against the average stock-day" in result.content  # 1.00% over a -0.10% base
    assert "no reliable edge either way" in result.content  # the interval crosses zero
    assert result.items[0].levels["breakout_level"] > 20  # kept in the state, not shown
    assert "from the last close" in result.content and "20.25" not in result.content


def test_nothing_after_the_decision_time_and_an_older_table_before_its_month(world):
    bars, t, *_ = world
    on_breakout = call(context(world, bars[t].day), "get_patterns", "{}")
    assert [r.split("_")[2] for r in on_breakout.refs] == ["candidate"]  # not yet confirmed
    assert "history from the table of 2025-06-01" in on_breakout.content  # the later one is not yet


def test_the_window_and_the_names_narrow_it(world):
    bars, *_ = world
    ctx = context(world, bars[-1].day)
    assert call(ctx, "get_patterns", json.dumps({"days": 1})).refs == []
    only = call(ctx, "get_patterns", json.dumps({"patterns": ["cuorou"]}))
    assert only.refs == [] and "None." in only.content
    capped = call(ctx, "get_patterns", json.dumps({"days": 500}))
    assert capped.arguments["days"] == 60 and "above the limit" in capped.content
    wrong = call(ctx, "get_patterns", json.dumps({"patterns": ["head_and_shoulders"]}))
    assert wrong.error is not None


def test_a_ticker_the_store_lacks_is_fetched_into_it(world):
    bars, _, source, news, _ = world
    asked = []

    def fetch(ticker, start):
        asked.append(ticker)
        return list(bars)

    source.store._fetch = fetch
    ctx = context(world, bars[-1].day)
    ctx = ToolContext(
        "TSM", ctx.decision_time, news, ctx.prices, ctx.statements, ctx.config, source
    )
    assert call(ctx, "get_patterns", "{}").refs
    assert asked == ["TSM"]
