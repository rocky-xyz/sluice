import json
from datetime import date, timedelta

import pytest

from sluice.agents.config import ToolsConfig
from sluice.agents.state import cited_refs
from sluice.agents.tools import ToolContext, call, specs
from sluice.fundamentals.store import FundamentalsStore
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore
from tests.agents.test_pipeline import CLOSE, CONFIG, DAY, article
from tests.fundamentals.statements import stock
from tests.prices.series import series

LATER = date(2026, 10, 9)  # prices are stored past the decision day


@pytest.fixture
def ctx(tmp_path):
    with (
        NewsStore(tmp_path / "news.db") as news,
        PriceStore(tmp_path / "prices.db") as prices,
    ):
        news.add(
            [
                article(1, CLOSE - timedelta(days=2), "Export controls hit chip sales"),
                article(2, CLOSE - timedelta(seconds=1), "Chipmaker beats on revenue"),
                article(3, CLOSE, "Published at the close: too late"),
                article(4, CLOSE + timedelta(hours=1), "Published after the close"),
                article(5, CLOSE - timedelta(days=20), "Older than the lookback"),
            ]
        )
        prices.replace("NVDA", series(LATER, 270, drift=0.002))
        prices.replace("SPY", series(LATER, 270, drift=0.001))
        statements = FundamentalsStore(tmp_path / "statements")
        stock(statements, "NVDA")
        config = CONFIG.model_copy(update={"tools": ToolsConfig()})
        yield ToolContext("NVDA", CLOSE, news, prices, statements, config)


def test_no_tool_takes_a_date_or_a_ticker():
    for spec in specs(CONFIG):
        assert spec.parameters["additionalProperties"] is False
        assert not {"date", "ticker", "as_of", "until"} & set(spec.parameters["properties"])


def test_an_argument_the_tool_does_not_have_is_refused(ctx):
    result = call(ctx, "search_news", json.dumps({"query": "chip", "date": "2026-10-08"}))
    assert result.error and "date" in result.error and "fixed for this run" in result.error
    assert result.refs == [] and result.content.startswith("Error:")


def test_news_search_never_returns_anything_from_the_decision_time_on(ctx):
    result = call(ctx, "search_news", "{}")
    titles = result.content
    assert "too late" not in titles and "after the close" not in titles
    assert "Older than the lookback" not in titles  # 14 days by default
    assert len(result.refs) == 2 and all(ref.startswith("N") for ref in result.refs)


def test_news_search_ranks_by_words_matched(ctx):
    both = call(ctx, "search_news", json.dumps({"query": "revenue export controls"}))
    assert [i.title for i in both.items] == [
        "Export controls hit chip sales",  # two words matched
        "Chipmaker beats on revenue",  # one is enough to be found
    ]
    # whole words: "chip" does not find "Chipmaker"
    chip = call(ctx, "search_news", json.dumps({"query": "chip"}))
    assert [i.title for i in chip.items] == ["Export controls hit chip sales"]


def test_limits_are_cut_and_said(ctx):
    result = call(ctx, "search_news", json.dumps({"days": 60, "limit": 99}))
    assert result.arguments == {"query": "", "days": 14, "limit": 20}
    assert "Note: days 60 is above the limit; 14 was used." in result.content
    assert "Note: limit 99 is above the limit; 20 was used." in result.content
    history = call(ctx, "get_price_history", json.dumps({"days": 400}))
    assert history.arguments == {"days": 250}


def test_price_history_ends_at_the_decision_day_and_gives_no_price_level(ctx):
    result = call(ctx, "get_price_history", json.dumps({"days": 5}))
    assert [row.day for row in result.items][-1] == DAY
    assert result.refs[-1] == f"P:day_{DAY}" and len(result.refs) == 5
    assert "2026-10-" not in result.content
    assert all(row.volume_ratio is not None for row in result.items)
    assert "$" not in result.content and "100." not in result.content


def test_indicators_and_fundamentals_are_v1s(ctx):
    indicators = call(ctx, "get_indicators", "{}")
    assert f"Indicators as of the close of {DAY}:" in indicators.content
    assert "P:rsi_14" in indicators.refs
    assert set(cited_refs(indicators.content)) == set(indicators.refs)
    figures = call(ctx, "get_fundamentals", json.dumps({"quarters": 2}))
    assert figures.refs and all(ref.startswith("F:") for ref in figures.refs)
    assert set(cited_refs(figures.content)) == set(figures.refs)
    assert call(ctx, "get_fundamentals", json.dumps({"quarters": 9})).arguments == {"quarters": 4}


def test_the_same_call_gives_the_same_text(ctx):
    for name, arguments in [
        ("get_indicators", "{}"),
        ("get_price_history", '{"days": 30}'),
        ("search_news", '{"query": "chip"}'),
        ("get_fundamentals", "{}"),
    ]:
        assert call(ctx, name, arguments).content == call(ctx, name, arguments).content


@pytest.mark.parametrize(
    ("name", "arguments", "problem"),
    [
        ("get_quotes", "{}", "there is no tool 'get_quotes'"),
        ("search_news", "{query", "not JSON"),
        ("get_price_history", '{"days": 0}', "days"),
        ("get_price_history", '{"days": "many"}', "days"),
        ("get_indicators", "[]", "arguments"),
    ],
)
def test_mistakes_are_told_to_the_model_not_raised(ctx, name, arguments, problem):
    result = call(ctx, name, arguments)
    assert result.error and problem in result.error and result.refs == []


def test_without_data_the_tool_says_so(tmp_path):
    with NewsStore(tmp_path / "n.db") as news, PriceStore(tmp_path / "p.db") as prices:
        empty = ToolContext("NVDA", CLOSE, news, prices, FundamentalsStore(tmp_path / "s"), CONFIG)
        assert "No news items matched" in call(empty, "search_news", "{}").content
        assert "Missing: no price history." in call(empty, "get_price_history", "{}").content
        assert "Missing:" in call(empty, "get_fundamentals", "{}").content
