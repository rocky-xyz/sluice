"""Weighted news in the pipeline (docs/specs/07-agent-pipeline.md, tasks 5.1-5.2)."""

import json
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from sluice.agents.batch import grid, unprepared
from sluice.agents.config import NewsConfig, load_config
from sluice.agents.prefetch import NEWS_SIGNAL, Weighting, prefetch, prefetch_weighted_news
from sluice.agents.roles import news_prompt
from sluice.agents.state import Inputs, Meta, dump_state
from sluice.news.models import NewsItem
from sluice.news.signal import SignalItem, ticker_day_signal
from sluice.news.store import NewsStore
from sluice.news.weight_tables import StoredTable
from sluice.news.weights import WeightEstimate, WeightTable
from sluice.scoring.client import JevError
from sluice.scoring.store import ItemScores, ScoreStore
from tests.agents.test_pipeline import CLOSE, CONFIG, FakeModel, go, pipeline
from tests.scoring.test_score import FakeJev, config, full_answer

SEP = datetime(2026, 9, 1, tzinfo=UTC)
OCT = datetime(2026, 10, 1, tzinfo=UTC)
WEIGHTED = NewsConfig(lookback_days=14, max_items=50, weighted=True, min_relevance=0.5)
WEIGHTED_CONFIG = CONFIG.model_copy(update={"news": WEIGHTED})


def estimate(low: float, high: float) -> WeightEstimate:
    return WeightEstimate(weight=(low + high) / 2, low=low, high=high, n=100)


def table(as_of: datetime, reuters: WeightEstimate) -> StoredTable:
    return StoredTable(
        as_of=as_of,
        n_rows=1000,
        table=WeightTable(
            lambda0=100.0,
            source={"Reuters": reuters, "(unknown)": estimate(-0.001, 0.001)},
            event_type={"earnings": estimate(-0.004, -0.001)},
        ),
    )


TABLES = [table(SEP, estimate(0.001, 0.003)), table(OCT, estimate(-0.003, -0.001))]


def article(n: int, hours_before: float, publisher: str | None = "Reuters") -> NewsItem:
    published = CLOSE - timedelta(hours=hours_before)
    return NewsItem(
        title=f"Chipmaker headline {n}",
        summary=f"Summary of story {n}.",
        source="finnhub",
        publisher=publisher,
        tickers=["NVDA"],
        url=f"https://example.com/story/{n}",
        published_at=published,
        fetched_at=published + timedelta(minutes=5),
    )


def scores(relevance: float, bullish: float = 0.7) -> ItemScores:
    return ItemScores(
        relevance=relevance,
        stance="bullish",
        stance_confidence=0.81,
        stance_probabilities={"bullish": bullish, "bearish": 0.1, "neutral": 0.9 - bullish},
        novelty=0.5,
        rumour=0.1,
        event_type="earnings",
        event_type_confidence=0.93,
        event_type_probabilities={"earnings": 0.93},
    )


@pytest.fixture
def store(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        yield store


@pytest.fixture
def score_store(tmp_path):
    with ScoreStore(tmp_path / "scores.db") as scores:
        yield scores


def ids(store) -> dict[str, int]:
    """Title -> store id."""
    return {item.title: item.id for item in store.query("NVDA", before=CLOSE + timedelta(days=1))}


def stocked(store, score_store, relevances: dict[int, float]) -> dict[int, int]:
    """Articles 1..n, article n the newest; scored where `relevances` says. n -> store id."""
    store.add([article(n, hours_before=10 * (len(relevances) + 1 - n)) for n in relevances])
    by_title = ids(store)
    numbered = {n: by_title[f"Chipmaker headline {n}"] for n in relevances}
    for n, relevance in relevances.items():
        if relevance is not None:
            score_store.put(numbered[n], "test", "deadbeef", scores(relevance))
    return numbered


def weighting(score_store, jev=None, tables=TABLES) -> Weighting:
    return Weighting(scores=score_store, tables=tables, scoring=config(), jev=jev or FakeJev())


def fetch(store, weighting_, news=WEIGHTED):
    return prefetch_weighted_news(store, "NVDA", CLOSE, news, weighting_)


# --- unweighted runs are as they were ---


def test_an_unweighted_run_saves_the_same_state_as_before(store, tmp_path):
    store.add([article(1, 5), article(2, 3)])
    saved = dump_state(go(pipeline(store, tmp_path, FakeModel())))
    assert "signals" not in saved["inputs"]
    assert set(saved["inputs"]["news"][0]) == {
        "ref", "title", "summary", "publisher", "published_at", "session", "reprint_count",
    }  # fmt: skip


def test_the_shipped_configs_differ_only_in_weighting():
    root = Path(__file__).parents[2] / "configs"
    v1, v2 = load_config("v1", root), load_config("v2", root)
    assert not v1.news.weighted
    assert (v2.news.weighted, v2.news.min_relevance, v2.news.scoring_config) == (True, 0.5, "v1")
    assert v1.models == v2.models and v1.debate_rounds == v2.debate_rounds
    assert {r for r in v1.prompts if v1.prompts[r] != v2.prompts[r]} == {"news_analyst"}


def test_a_weighted_config_without_weighting_is_refused(store, tmp_path):
    with pytest.raises(ValueError, match="weights news"):
        prefetch(store, None, None, "NVDA", CLOSE, WEIGHTED_CONFIG)


# --- which weight table ---


def test_the_latest_table_fitted_by_the_decision_time_is_used(store, score_store):
    stocked(store, score_store, {1: 0.9})
    news, signals, gaps = fetch(store, weighting(score_store))
    # Decided on 30 Sep: the October table, which reverses Reuters, is not visible yet.
    assert news[0].source_record == "confirmed"
    assert news[0].type_record == "reversed"
    assert signals[0].weights_as_of == SEP
    assert gaps == []


def test_before_any_table_there_are_no_labels_and_no_signal(store, score_store):
    stocked(store, score_store, {1: 0.9, 2: 0.2})
    news, signals, gaps = fetch(store, weighting(score_store, tables=[TABLES[1]]))
    assert signals == []
    assert [(n.title, n.source_record, n.event_type) for n in news] == [
        ("Chipmaker headline 1", None, "earnings")  # still filtered by relevance
    ]
    assert [g.reason for g in gaps] == ["no weight table fitted by the decision time"]


# --- choosing the items ---


def test_relevance_filters_before_the_cap(store, score_store):
    stocked(store, score_store, {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.1})  # 4 is the newest
    news, _, _ = fetch(store, weighting(score_store), WEIGHTED.model_copy(update={"max_items": 2}))
    assert [n.title for n in news] == ["Chipmaker headline 2", "Chipmaker headline 3"]


def test_relevance_exactly_at_the_threshold_is_kept(store, score_store):
    stocked(store, score_store, {1: 0.5, 2: 0.49})
    news, _, _ = fetch(store, weighting(score_store))
    assert [n.title for n in news] == ["Chipmaker headline 1"]


def test_when_nothing_is_relevant_enough_there_are_no_items_and_a_gap(store, score_store):
    stocked(store, score_store, {1: 0.1})
    news, signals, gaps = fetch(store, weighting(score_store))
    assert news == []
    assert len(signals) == 1  # the signal still covers every scored item
    assert [g.reason for g in gaps] == ["no item reached relevance 0.5"]


# --- items without scores ---


def test_an_unscored_item_is_scored_now_stored_and_not_asked_again(store, score_store):
    numbered = stocked(store, score_store, {1: 0.9, 2: None})
    jev = FakeJev(full_answer())
    _, signals, gaps = fetch(store, weighting(score_store, jev))
    assert gaps == []
    assert signals[0].n_items == 2
    assert score_store.get(numbered[2], "test").relevance == 0.9
    # compared with the titles published before it, as `sluice score` does
    assert "Chipmaker headline 1" in jev.requests[0].state

    fetch(store, weighting(score_store, FakeJev()))  # a second call to Jev would fail


def test_an_item_jev_cannot_score_is_shown_unlabelled_and_left_for_next_time(store, score_store):
    numbered = stocked(store, score_store, {1: 0.9, 2: None})
    jev = FakeJev(JevError("rate limited"))
    news, signals, gaps = fetch(store, weighting(score_store, jev))
    assert [(n.title, n.scored) for n in news] == [
        ("Chipmaker headline 1", True),
        ("Chipmaker headline 2", False),
    ]
    assert news[1].event_type is None and news[1].source_record is None
    assert signals[0].n_items == 1
    assert [g.reason for g in gaps] == ["1 items could not be scored"]
    assert not score_store.done(numbered[2], "test")


# --- the news signal ---


def test_the_signal_covers_every_scored_item_before_the_filter(store, score_store):
    stocked(store, score_store, {1: 0.9, 2: 0.2})
    news, signals, _ = fetch(store, weighting(score_store))
    sep = TABLES[0].table
    items = [
        SignalItem(raw=0.6, source="Reuters", event_type="earnings", relevance=relevance)
        for relevance in (0.9, 0.2)
    ]
    expected = ticker_day_signal(items, sep)
    assert len(news) == 1
    assert (signals[0].ref, signals[0].n_items) == (NEWS_SIGNAL, 2)
    assert signals[0].value == pytest.approx(expected, abs=1e-8)


def test_no_signal_when_no_scored_item_has_any_relevance(store, score_store):
    stocked(store, score_store, {1: 0.0})
    _, signals, gaps = fetch(store, weighting(score_store))
    assert signals == []
    assert [g.reason for g in gaps] == [
        "no news signal: no scored item has any relevance (weights 2026-09-01)",
        "no item reached relevance 0.5",
    ]


# --- what the analyst sees ---


def test_the_prompt_shows_labels_and_the_signal_but_no_raw_numbers(store, score_store):
    stocked(store, score_store, {1: 0.9})
    news, signals, _ = fetch(store, weighting(score_store))
    meta = Meta(
        run_id="r", ticker="NVDA", decision_time=CLOSE, mode="backtest",
        config_version="t", config_digest="d", repeat=0,
    )  # fmt: skip
    prompt = news_prompt(meta, Inputs(news=news, signals=signals))
    assert "type: earnings | source record: confirmed | type record: reversed" in prompt
    assert "[S:news_signal] news signal: " in prompt
    for raw in ("0.7", "0.81", "0.93", "0.9", "0.002", "probab", "confidence", "relevance 0"):
        assert raw not in prompt.replace(f"{signals[0].value:+.2e}", "")


def test_a_weighted_run_cites_the_signal_and_reads_back_from_its_checkpoint(
    store, score_store, tmp_path
):
    stocked(store, score_store, {1: 0.9})
    by_title = ids(store)
    ref = f"N{by_title['Chipmaker headline 1']}"
    model = FakeModel(analyst=(f"Results beat [{ref}]. The flow leans up [S:news_signal].",))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = pipeline(
            store, tmp_path, model, config=WEIGHTED_CONFIG, weighting=weighting(score_store)
        )
        state = go(graph)
        again = go(graph)
    assert state["meta"].status == "ok"
    assert "[S:news_signal]" in state["reports"]["news"].text
    assert again == state
    saved = dump_state(state)
    assert saved["inputs"]["signals"][0]["ref"] == "S:news_signal"
    assert saved["inputs"]["news"][0]["source_record"] == "confirmed"
    assert json.loads(json.dumps(saved)) == saved


# --- the batch checks for weight tables ---


def test_a_batch_needs_a_weight_table_for_every_decision_day(tmp_path):
    from sluice.fundamentals.store import FundamentalsStore
    from sluice.prices.store import PriceStore
    from tests.fundamentals.statements import stock
    from tests.prices.series import series

    statements = FundamentalsStore(tmp_path / "statements")
    stock(statements, "NVDA")
    with PriceStore(tmp_path / "prices.db") as prices:
        prices.replace("NVDA", series(date(2026, 9, 30), 30))
        prices.replace("SPY", series(date(2026, 9, 30), 30))
        cells = grid(["NVDA"], [date(2026, 8, 31), date(2026, 9, 30)])
        assert unprepared(cells, statements, prices, "SPY") == []  # unweighted: not checked
        assert unprepared(cells, statements, prices, "SPY", TABLES) == [
            "no weight table fitted by 2026-08-31 (sluice news fit-weights)"
        ]
        assert unprepared(cells[1:], statements, prices, "SPY", TABLES) == []
