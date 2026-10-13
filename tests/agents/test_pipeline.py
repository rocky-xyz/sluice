import json
import sqlite3
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from sluice.agents import indicators
from sluice.agents.cache import CallCache
from sluice.agents.config import NewsConfig, PipelineConfig, PricesConfig, load_config
from sluice.agents.graph import Deps, build_graph, open_checkpointer, run
from sluice.agents.llm import CachedLLM, LLMError, LLMRequest, LLMResponse
from sluice.agents.prefetch import (
    backtest_decision_time,
    prefetch_news,
    prefetch_prices,
    select_news,
)
from sluice.agents.state import (
    Gap,
    Meta,
    cited_refs,
    dump_state,
    make_run_id,
    merge_meta,
)
from sluice.news.models import NewsItem
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore
from tests.prices.series import series

DAY = date(2026, 9, 30)
CLOSE = datetime(2026, 9, 30, 20, 0, tzinfo=UTC)  # 16:00 Eastern
NEWS = NewsConfig(lookback_days=14, max_items=50)
CONFIG = PipelineConfig(
    version="t1",
    digest="abc12345",
    temperature=0.0,
    retries=2,
    models={"news_analyst": "fast-model", "research_manager": "deep-model"},
    prices=PricesConfig(lookback_days=250, benchmark="SPY"),
    news=NEWS,
    prompts={"news_analyst": "You are the news analyst.", "research_manager": "You decide."},
)


def article(n: int, published: datetime, title: str | None = None) -> NewsItem:
    return NewsItem(
        title=title or f"Chipmaker headline {n}",
        summary=f"Summary of story {n}.",
        source="finnhub",
        publisher="Reuters",
        tickers=["NVDA"],
        url=f"https://example.com/story/{n}",
        published_at=published,
        fetched_at=published + timedelta(minutes=5),
    )


@pytest.fixture
def store(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        yield store


@pytest.fixture
def stocked(store):
    """Two stories before the close, one at the close, one after."""
    store.add(
        [
            article(1, CLOSE - timedelta(days=2)),
            article(2, CLOSE - timedelta(seconds=1)),
            article(3, CLOSE),
            article(4, CLOSE + timedelta(hours=1)),
        ]
    )
    return store


def verdict(*refs: str, rating: str = "overweight") -> str:
    return json.dumps(
        {
            "rating": rating,
            "confidence": 0.6,
            "evidence": [{"claim": "Demand is strong.", "direction": "supports", "refs": list(refs)}],
            "rationale": "The news leans positive.",
        }
    )


class FakeModel:
    """Stands in for DeepSeek. Each role answers from its own queue; the last answer repeats."""

    def __init__(self, analyst=("Demand is strong [N1]. Supply is tight [N2].",), manager=None):
        self.answers = {
            "fast-model": list(analyst),
            "deep-model": list(manager if manager is not None else [verdict("N1")]),
        }
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        queue = self.answers[request.model]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return LLMResponse(text=answer, input_tokens=100, output_tokens=20)

    def asked(self, model: str) -> list[LLMRequest]:
        return [r for r in self.requests if r.model == model]


def pipeline(store, tmp_path, model, cache_name="llm.db", checkpoints=True):
    cache = CallCache(tmp_path / cache_name)
    saver = open_checkpointer(tmp_path / "checkpoints.db") if checkpoints else None
    prices = PriceStore(tmp_path / "prices.db")
    if prices.latest("NVDA") is None:
        prices.replace("NVDA", series(DAY, 260, drift=0.002))
        prices.replace("SPY", series(DAY, 260, drift=0.001))
    llm = CachedLLM(cache, model, sleep=lambda _: None)
    return build_graph(Deps(store, prices, llm, CONFIG), saver)


def go(graph, repeat=0):
    return run(graph, CONFIG, "NVDA", CLOSE, repeat=repeat)


# --- decision time ---


def test_backtest_decides_at_the_close():
    assert backtest_decision_time(DAY) == CLOSE


def test_early_close_day_decides_at_the_early_close():
    # the day after Thanksgiving closes at 13:00 Eastern
    assert backtest_decision_time(date(2026, 11, 27)) == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)


@pytest.mark.parametrize("day", [date(2026, 10, 3), date(2026, 12, 25)])
def test_a_day_without_trading_is_refused(day):
    with pytest.raises(ValueError, match="not a trading day"):
        backtest_decision_time(day)


# --- prefetch ---


def test_nothing_published_at_or_after_the_decision_time_is_fetched(stocked):
    news, gaps = prefetch_news(stocked, "NVDA", CLOSE, NEWS)
    assert [item.title for item in news] == ["Chipmaker headline 1", "Chipmaker headline 2"]
    assert all(item.published_at < CLOSE for item in news)
    assert gaps == []


def test_news_older_than_the_lookback_is_left_out(store):
    store.add([article(1, CLOSE - timedelta(days=14, seconds=1)), article(2, CLOSE - timedelta(days=14))])
    news, _ = prefetch_news(store, "NVDA", CLOSE, NEWS)
    assert [item.title for item in news] == ["Chipmaker headline 2"]


def test_each_item_is_numbered_by_its_store_id(stocked):
    news, _ = prefetch_news(stocked, "NVDA", CLOSE, NEWS)
    stored = {f"N{item.id}": item.title for item in stocked.query("NVDA", before=CLOSE)}
    assert {item.ref: item.title for item in news} == stored


def test_no_news_is_recorded_as_a_gap(store):
    news, gaps = prefetch_news(store, "NVDA", CLOSE, NEWS)
    assert news == []
    assert [gap.kind for gap in gaps] == ["news"]


def test_over_the_cap_the_most_reprinted_then_most_recent_are_kept(store):
    store.add([article(n, CLOSE - timedelta(days=5) + timedelta(hours=n)) for n in range(1, 5)])
    # a second vendor-side copy of story 1, an hour later, makes it a reprinted story
    store.add([article(9, CLOSE - timedelta(days=5) + timedelta(hours=2), title="Chipmaker headline 1")])
    store.rebuild_clusters()
    found = store.query("NVDA", before=CLOSE)
    assert {item.title: item.reprint_count for item in found}["Chipmaker headline 1"] == 1

    kept = select_news(found, max_items=2)
    assert [item.title for item in kept] == ["Chipmaker headline 1", "Chipmaker headline 4"]


# --- state helpers ---


def test_citations_are_read_from_square_brackets():
    text = "Up [N12]. Down [N3, N40] and [P:rsi14]; not N99, not [note] and not [F: x]."
    assert cited_refs(text) == {"N12", "N3", "N40", "P:rsi14"}


def test_run_id_follows_from_the_run_parameters():
    first = make_run_id("NVDA", CLOSE, "v1", "abc12345", 0)
    assert first == make_run_id("NVDA", CLOSE, "v1", "abc12345", 0)
    assert first == "NVDA_20260930T2000Z_v1-abc12345_r0"
    others = {
        make_run_id("AMD", CLOSE, "v1", "abc12345", 0),
        make_run_id("NVDA", CLOSE + timedelta(days=1), "v1", "abc12345", 0),
        make_run_id("NVDA", CLOSE, "v2", "abc12345", 0),
        make_run_id("NVDA", CLOSE, "v1", "ffff0000", 0),
        make_run_id("NVDA", CLOSE, "v1", "abc12345", 1),
    }
    assert len(others | {first}) == 6


def test_a_failed_run_stays_failed_when_updates_are_merged():
    ok = Meta(
        run_id="r", ticker="NVDA", decision_time=CLOSE, mode="backtest",
        config_version="t1", config_digest="d", repeat=0,
    )  # fmt: skip
    one = ok.model_copy(update={"status": "failed", "failure": "a"})
    two = ok.model_copy(update={"status": "failed", "failure": "b"})
    assert merge_meta(None, ok) == ok
    assert merge_meta(ok, one) == one
    assert merge_meta(one, ok) == one
    assert merge_meta(one, two).failure == "a; b"


# --- config ---


def test_the_shipped_config_loads():
    config = load_config("v1", Path(__file__).parents[2] / "configs")
    assert config.version == "v1"
    assert set(config.models) == set(config.prompts) == {"news_analyst", "research_manager"}
    assert (config.prices.lookback_days, config.prices.benchmark) == (250, "SPY")
    assert (config.temperature, config.retries) == (0.0, 2)
    assert (config.news.lookback_days, config.news.max_items) == (14, 50)


def test_an_edited_config_has_another_digest_and_unknown_keys_are_refused(tmp_path):
    source = (Path(__file__).parents[2] / "configs/pipeline/v1.toml").read_text()
    folder = tmp_path / "pipeline"
    folder.mkdir()
    (folder / "v1.toml").write_text(source)
    (folder / "v2.toml").write_text(source.replace("max_items = 50", "max_items = 40"))
    (folder / "v3.toml").write_text("temprature = 1.0\n" + source)

    assert load_config("v1", tmp_path).digest != load_config("v2", tmp_path).digest
    with pytest.raises(ValueError):
        load_config("v3", tmp_path)


# --- the graph ---


def test_a_run_ends_with_a_rating_and_its_evidence(stocked, tmp_path):
    model = FakeModel()
    state = go(pipeline(stocked, tmp_path, model))

    assert state["meta"].status == "ok"
    assert state["meta"].run_id == "NVDA_20260930T2000Z_t1-abc12345_r0"
    assert state["reports"]["news"].text == "Demand is strong [N1]. Supply is tight [N2]."
    decision = state["decision"]
    assert (decision.rating, decision.confidence) == ("overweight", 0.6)
    assert decision.evidence[0].refs == ["N1"]
    assert decision.notes == []
    assert [(c.role, c.model, c.cached) for c in state["usage"]] == [
        ("news_analyst", "fast-model", False),
        ("research_manager", "deep-model", False),
    ]


def test_each_role_uses_its_configured_model_prompt_and_temperature(stocked, tmp_path):
    model = FakeModel()
    go(pipeline(stocked, tmp_path, model))

    analyst, manager = model.requests
    assert (analyst.model, analyst.messages[0].content) == ("fast-model", "You are the news analyst.")
    assert (manager.model, manager.messages[0].content) == ("deep-model", "You decide.")
    assert (analyst.temperature, manager.temperature) == (0.0, 0.0)
    assert (analyst.json_mode, manager.json_mode) == (False, True)


def test_the_analyst_sees_the_snapshot_and_the_manager_only_the_report(stocked, tmp_path):
    model = FakeModel()
    state = go(pipeline(stocked, tmp_path, model))

    analyst, manager = (request.messages[1].content for request in model.requests)
    for item in state["inputs"].news:
        assert f"[{item.ref}]" in analyst and item.title in analyst and item.summary in analyst
        assert item.title not in manager and item.summary not in manager
    assert "Chipmaker headline 3" not in analyst and "Chipmaker headline 4" not in analyst
    assert state["reports"]["news"].text in manager


def test_a_report_citing_an_unknown_id_is_asked_again(stocked, tmp_path):
    model = FakeModel(analyst=["Big news [N999].", "No sources at all.", "Demand is strong [N1]."])
    state = go(pipeline(stocked, tmp_path, model))

    assert state["reports"]["news"].text == "Demand is strong [N1]."
    assert len(model.asked("fast-model")) == 3


def test_evidence_citing_an_unknown_id_is_asked_again(stocked, tmp_path):
    # N999 does not exist; N2 exists but the report never cites it
    model = FakeModel(analyst=["Demand is strong [N1]."], manager=[verdict("N999"), verdict("N2"), verdict("N1")])
    state = go(pipeline(stocked, tmp_path, model))

    assert state["decision"].evidence[0].refs == ["N1"]
    assert len(model.asked("deep-model")) == 3
    assert [c.role for c in state["usage"]].count("research_manager") == 3


@pytest.mark.parametrize("bad", ["not json", '{"rating": "strong buy"}', verdict(), verdict("N999")])
def test_without_an_acceptable_answer_the_run_fails_and_has_no_rating(stocked, tmp_path, bad):
    model = FakeModel(manager=[bad])
    state = go(pipeline(stocked, tmp_path, model))

    assert state["meta"].status == "failed"
    assert state["meta"].failure.startswith("research_manager:")
    assert state.get("decision") is None
    assert len(model.asked("deep-model")) == CONFIG.retries + 1
    assert dump_state(state)["decision"] is None


def test_a_failed_analyst_ends_the_run_before_the_manager(stocked, tmp_path):
    model = FakeModel(analyst=[LLMError("HTTP 401")])
    state = go(pipeline(stocked, tmp_path, model))

    assert state["meta"].status == "failed"
    assert state["meta"].failure == "news_analyst: HTTP 401"
    assert model.asked("deep-model") == []


def test_without_news_nobody_is_asked_and_the_gap_is_recorded(store, tmp_path):
    model = FakeModel()
    state = go(pipeline(store, tmp_path, model))

    assert model.requests == []
    assert state["reports"]["news"].no_data
    assert [gap.kind for gap in state["inputs"].gaps] == ["news"]
    # with a single analyst there is then nothing to decide on
    assert state["meta"].status == "failed"
    assert state.get("decision") is None


# --- cache, repeats, checkpoints ---


def test_a_finished_run_is_returned_without_new_calls(stocked, tmp_path):
    model = FakeModel()
    first = go(pipeline(stocked, tmp_path, model))
    again = go(pipeline(stocked, tmp_path, model))

    assert len(model.requests) == 2
    assert dump_state(again) == dump_state(first)


def test_without_checkpoints_a_rerun_is_served_by_the_cache(stocked, tmp_path):
    model = FakeModel()
    first = go(pipeline(stocked, tmp_path, model, checkpoints=False))
    again = go(pipeline(stocked, tmp_path, model, checkpoints=False))

    assert len(model.requests) == 2
    assert all(call.cached for call in again["usage"])
    assert again["decision"] == first["decision"]
    assert again["inputs"] == first["inputs"]


def test_another_repeat_is_another_run_with_new_calls(stocked, tmp_path):
    model = FakeModel()
    graph = pipeline(stocked, tmp_path, model)
    first, second = go(graph, repeat=0), go(graph, repeat=1)

    assert len(model.requests) == 4
    assert (first["meta"].repeat, second["meta"].repeat) == (0, 1)
    assert first["meta"].run_id != second["meta"].run_id


def test_an_interrupted_run_continues_after_its_last_finished_node(stocked, tmp_path):
    model = FakeModel(manager=[RuntimeError("power cut"), verdict("N1")])
    with pytest.raises(RuntimeError, match="power cut"):
        go(pipeline(stocked, tmp_path, model))

    # an empty cache: only the checkpoint can spare the analyst a second call
    state = go(pipeline(stocked, tmp_path, model, cache_name="fresh.db"))
    assert state["meta"].status == "ok"
    assert len(model.asked("fast-model")) == 1
    assert [c.role for c in state["usage"]] == ["news_analyst", "research_manager"]


def test_a_failed_run_starts_over_and_only_asks_what_failed(stocked, tmp_path):
    model = FakeModel(manager=[LLMError("HTTP 401"), verdict("N1")])
    failed = go(pipeline(stocked, tmp_path, model))
    assert failed["meta"].status == "failed"

    state = go(pipeline(stocked, tmp_path, model))
    assert state["meta"].status == "ok"
    assert len(model.asked("fast-model")) == 1
    assert [(c.role, c.cached) for c in state["usage"]] == [
        ("news_analyst", True),
        ("research_manager", False),
    ]


def test_checkpoints_are_read_back_as_the_same_objects_without_warnings(stocked, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = go(pipeline(stocked, tmp_path, FakeModel()))
        saver = SqliteSaver(sqlite3.connect(tmp_path / "checkpoints.db"))
        assert saver.get_tuple({"configurable": {"thread_id": first["meta"].run_id}}) is not None
        again = go(pipeline(stocked, tmp_path, FakeModel()))
    assert isinstance(again["meta"], Meta)
    assert again == first


def test_the_saved_state_has_all_six_layers(stocked, tmp_path):
    saved = dump_state(go(pipeline(stocked, tmp_path, FakeModel())))
    assert list(saved) == ["meta", "inputs", "reports", "debate", "decision", "usage"]
    assert json.loads(json.dumps(saved)) == saved
    assert saved["meta"]["decision_time"] == "2026-09-30T20:00:00Z"
    assert saved["debate"] == []


# --- prices ---

PRICES = CONFIG.prices
NOON = datetime(2026, 9, 30, 16, 0, tzinfo=UTC)  # 12:00 Eastern, the session still open
DAY_BEFORE = date(2026, 9, 29)


@pytest.fixture
def prices(tmp_path):
    with PriceStore(tmp_path / "prices.db") as prices:
        prices.replace("NVDA", series(date(2026, 10, 9), 267, drift=0.002))
        prices.replace("SPY", series(date(2026, 10, 9), 267, drift=0.001))
        yield prices


def test_a_backtest_at_the_close_uses_that_days_bar(prices):
    shown, as_of, gaps = prefetch_prices(prices, "NVDA", CLOSE, PRICES)
    assert as_of == DAY
    assert [item.name for item in shown] == list(indicators.NAMES)
    assert all(item.ref == f"P:{item.name}" for item in shown)
    assert gaps == []


def test_during_trading_hours_the_last_finished_day_is_used(prices):
    _, as_of, _ = prefetch_prices(prices, "NVDA", NOON, PRICES)
    assert as_of == DAY_BEFORE


def test_changing_every_later_price_changes_no_indicator(prices):
    before = prefetch_prices(prices, "NVDA", CLOSE, PRICES)
    for ticker in ("NVDA", "SPY"):
        bars = prices.history(ticker, date(2026, 10, 9), limit=1000)
        prices.replace(
            ticker,
            [
                bar if bar.day <= DAY else bar.model_copy(update={"close": 1.0, "volume": 9e9})
                for bar in bars
            ],
        )
    assert prefetch_prices(prices, "NVDA", CLOSE, PRICES) == before

    # and the check has teeth: changing the decision day's own bar does change them
    bars = prices.history("NVDA", date(2026, 10, 9), limit=1000)
    prices.replace(
        "NVDA", [bar if bar.day != DAY else bar.model_copy(update={"close": 1.0}) for bar in bars]
    )
    assert prefetch_prices(prices, "NVDA", CLOSE, PRICES) != before


def test_indicators_match_the_formulas_on_the_visible_bars(prices):
    shown, _, _ = prefetch_prices(prices, "NVDA", CLOSE, PRICES)
    values = {item.name: item.value for item in shown}
    own = prices.history("NVDA", DAY, limit=250)
    index = prices.history("SPY", DAY, limit=250)

    assert len(own) == 250
    assert values["ret_5d"] == round(own[-1].close / own[-6].close - 1, 4)
    expected = (own[-1].close / own[-21].close) - (index[-1].close / index[-21].close)
    assert values["excess_20d"] == round(expected, 4)


def test_no_price_level_is_shown(prices):
    shown, _, _ = prefetch_prices(prices, "NVDA", CLOSE, PRICES)
    closes = {round(bar.close, 4) for bar in prices.history("NVDA", DAY, limit=250)}
    assert not closes & {item.value for item in shown}


def test_a_short_history_is_used_as_far_as_it_goes_and_the_rest_is_a_gap(tmp_path):
    with PriceStore(tmp_path / "short.db") as short:
        short.replace("NEWCO", series(DAY, 30))
        short.replace("SPY", series(DAY, 250))
        shown, _, gaps = prefetch_prices(short, "NEWCO", CLOSE, PRICES)

    assert {item.name for item in shown} == {
        "ret_5d", "ret_20d", "excess_5d", "excess_20d", "vol_20d", "rsi_14", "volume_ratio_20d",
    }  # fmt: skip
    assert [gap.reason for gap in gaps] == [
        "could not compute: ret_60d, excess_60d, vs_sma50, vs_sma200"
    ]


def test_no_prices_and_stale_prices_are_gaps(tmp_path):
    with PriceStore(tmp_path / "gaps.db") as store:
        assert prefetch_prices(store, "NVDA", CLOSE, PRICES) == (
            [], None, [Gap(kind="prices", reason="no price history")],
        )  # fmt: skip
        store.replace("NVDA", series(DAY_BEFORE, 250))
        store.replace("SPY", series(DAY_BEFORE, 250))
        _, as_of, gaps = prefetch_prices(store, "NVDA", CLOSE, PRICES)
    assert as_of == DAY_BEFORE
    assert gaps[0].reason == "prices end on 2026-09-29, before the last closed session 2026-09-30"


def test_the_benchmark_itself_has_no_excess_return_and_no_gap(prices):
    shown, _, gaps = prefetch_prices(prices, "SPY", CLOSE, PRICES)
    assert not any(item.name.startswith("excess_") for item in shown)
    assert len(shown) == len(indicators.NAMES) - 3
    assert gaps == []


def test_the_run_snapshot_holds_the_indicators_but_the_news_analyst_cannot_cite_them(
    stocked, tmp_path
):
    model = FakeModel(analyst=["Momentum is strong [P:ret_5d].", "Demand is strong [N1]."])
    state = go(pipeline(stocked, tmp_path, model))

    assert state["inputs"].prices_as_of == DAY
    assert len(state["inputs"].prices) == len(indicators.NAMES)
    assert "P:ret_5d" in state["inputs"].refs()
    assert len(model.asked("fast-model")) == 2
    assert "ret_5d" not in model.requests[0].messages[1].content
    assert dump_state(state)["inputs"]["prices"][0] == {
        "ref": "P:ret_5d", "name": "ret_5d", "value": state["inputs"].prices[0].value,
    }  # fmt: skip
