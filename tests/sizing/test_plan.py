from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from sluice.evaluation.decisions import Decision
from sluice.news.models import NewsItem
from sluice.news.signal import signal_items, ticker_day_signal
from sluice.news.store import NewsStore
from sluice.news.weight_tables import StoredTable
from sluice.news.weights import WeightEstimate, WeightTable
from sluice.scoring.reports import ReportScoreStore
from sluice.scoring.store import ItemScores, ScoreStore
from sluice.sizing.config import load_sizing_config
from sluice.sizing.plan import Market, Sources, ablations, formula_schedules, news_score

CONFIG = load_sizing_config("v1", Path(__file__).parents[2] / "configs")
# two stocks cannot sum to 1 under the shipped 20% cap
LOOSE = CONFIG.model_copy(update={"stocks": CONFIG.stocks.model_copy(update={"max_weight": 1.0})})
DAY = date(2026, 3, 3)
CLOSE = datetime(2026, 3, 3, 21, 0, tzinfo=UTC)
TABLE = StoredTable(
    as_of=datetime(2026, 3, 1, tzinfo=UTC),
    n_rows=1000,
    table=WeightTable(
        lambda0=1.0,
        source={"Reuters": WeightEstimate(0.002, 0.001, 0.003, 100)},
        event_type={"earnings": WeightEstimate(-0.001, -0.002, 0.0, 100)},
    ),
)


def scores(bullish: float, relevance: float = 0.9) -> ItemScores:
    return ItemScores(
        relevance=relevance, stance="bullish", stance_confidence=0.8,
        stance_probabilities={"bullish": bullish, "bearish": 0.1, "neutral": 0.9 - bullish},
        novelty=0.5, rumour=0.1, event_type="earnings", event_type_confidence=0.9,
        event_type_probabilities={"earnings": 0.9},
    )  # fmt: skip


@pytest.fixture
def src(tmp_path):
    with (
        NewsStore(tmp_path / "news.db") as news,
        ScoreStore(tmp_path / "scores.db") as score_store,
        ReportScoreStore(tmp_path / "reports.db") as reports,
    ):
        yield Sources(news, score_store, [TABLE], reports, prices=None)


def add(news: NewsStore, n: int, hours_before: float) -> int:
    published = CLOSE - timedelta(hours=hours_before)
    news.add([NewsItem(
        title=f"Headline {n}", source="finnhub", publisher="Reuters", tickers=["NVDA"],
        url=f"https://example.com/{n}", published_at=published, fetched_at=published,
    )])  # fmt: skip
    found = news.query("NVDA", before=CLOSE + timedelta(days=1))
    return next(i.id for i in found if i.title == f"Headline {n}")


def test_the_news_score_uses_scored_items_only_and_never_asks_jev(src):
    scored, unscored = add(src.news, 1, 5), add(src.news, 2, 3)
    src.scores.put(scored, "v1", "x", scores(0.7))
    found = src.news.query("NVDA", before=CLOSE, since=CLOSE - timedelta(days=14))
    expected = ticker_day_signal(signal_items(found, {scored: scores(0.7)}), TABLE.table)
    assert news_score(src, "NVDA", CLOSE, CONFIG) == pytest.approx(expected)
    assert not src.scores.done(unscored, "v1")  # left unscored: Sources has no Jev to ask


def test_no_table_by_the_decision_time_means_no_news_score(src):
    add(src.news, 1, 5)
    early = Sources(src.news, src.scores, [], src.reports, None)
    assert news_score(early, "NVDA", CLOSE, CONFIG) is None


def test_news_outside_the_window_or_after_the_decision_is_left_out(src):
    old, late = add(src.news, 1, 24 * 20), add(src.news, 2, -1)
    for item_id in (old, late):
        src.scores.put(item_id, "v1", "x", scores(0.7))
    assert news_score(src, "NVDA", CLOSE, CONFIG) is None  # nothing scored inside the window


def _market() -> Market:
    return Market(exposure=0.8, vols={"A": 0.2, "B": 0.2}, news={"A": None, "B": None})


def test_formula_schedules_count_missing_ratings_and_lean_towards_the_better_rated(src):
    decisions = [
        Decision(ticker="A", day=DAY, repeat=0, config="v", run_id=None, rating="buy", confidence=0.6),
        Decision(ticker="B", day=DAY, repeat=0, config="v", run_id=None, rating=None, confidence=None),
    ]
    rebalance = date(2026, 3, 4)
    [(repeat, (schedule, missing))] = formula_schedules(
        decisions, {DAY: _market()}, {DAY: rebalance}, src, LOOSE
    ).items()
    assert (repeat, missing) == (0, 1)
    # only A has a rating: one value has no spread, so z is 0 for both and the weights stay equal
    assert schedule[rebalance] == pytest.approx({"A": 0.4, "B": 0.4})


def test_the_ablations_read_no_rating(src):
    rebalance = date(2026, 3, 4)
    out = ablations({DAY: _market()}, {DAY: rebalance}, LOOSE)
    assert set(out) == {"只做总仓位（等权 × E）", "波动率缩放 + 总仓位（k = 0）"}
    assert out["只做总仓位（等权 × E）"][rebalance] == pytest.approx({"A": 0.4, "B": 0.4})
