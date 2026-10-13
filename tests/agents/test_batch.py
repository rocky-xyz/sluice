import json
from datetime import date, timedelta

import pytest

from sluice.agents.batch import (
    RATINGS_FILE,
    Cell,
    CellResult,
    decision_days,
    estimated_cost,
    grid,
    run_batch,
    run_dir,
    state_path,
    unprepared,
    write_ratings,
)
from sluice.agents.cache import CallCache
from sluice.agents.graph import Deps, build_graph, open_checkpointer
from sluice.agents.llm import CachedLLM
from sluice.agents.prefetch import backtest_decision_time
from sluice.fundamentals.store import FundamentalsStore
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore
from tests.agents.test_pipeline import CONFIG, FakeModel, article, verdict
from tests.fundamentals.statements import stock
from tests.prices.series import series

DAYS = [date(2026, 9, 16), date(2026, 9, 30)]
TICKERS = ["NVDA", "AMD"]


@pytest.fixture
def world(tmp_path):
    """News, prices and statements for two tickers on two decision days."""
    with NewsStore(tmp_path / "news.db") as store:
        for n, (day, ticker) in enumerate((d, t) for d in DAYS for t in TICKERS):
            item = article(n, backtest_decision_time(day) - timedelta(hours=3))
            store.add([item.model_copy(update={"tickers": [ticker]})])
    with PriceStore(tmp_path / "prices.db") as prices:
        for ticker in (*TICKERS, "SPY"):
            prices.replace(ticker, series(DAYS[-1], 260))
    statements = FundamentalsStore(tmp_path / "statements")
    for ticker in TICKERS:
        stock(statements, ticker)
    return tmp_path


class Harness:
    """Shared stores plus a graph factory, the way the command line wires a batch."""

    def __init__(self, root, model):
        self.root, self.model = root, model
        self.prices = PriceStore(root / "prices.db")
        self.statements = FundamentalsStore(root / "statements")
        self.llm = CachedLLM(CallCache(root / "llm.db"), model, sleep=lambda _: None)
        self.saver = open_checkpointer(root / "checkpoints.db")
        self.graphs = 0

    def make_graph(self):
        self.graphs += 1
        deps = Deps(NewsStore(self.root / "news.db"), self.prices, self.statements, self.llm, CONFIG)
        return build_graph(deps, self.saver)

    def run(self, cells, workers=1):
        return run_batch(cells, CONFIG, self.make_graph, self.root / "runs", workers)


def any_news_model(**kwargs) -> FakeModel:
    """Cites prices only, so it is valid whatever the news ids of a cell are."""
    defaults = {"analyst": ["News is quiet [N0]."], "manager": [verdict("P:ret_5d")]}
    return FakeModel(**(defaults | kwargs))


class CitingModel(FakeModel):
    """A news analyst that cites the first id it was shown."""

    def complete(self, request):
        if request.messages[0].content == CONFIG.prompts["news_analyst"]:
            ref = request.messages[1].content.split("[N", 1)[1].split("]", 1)[0]
            self.answers["news_analyst"] = [f"Something happened [N{ref}]."]
        return super().complete(request)


# --- the grid ---


def test_decision_days_are_every_tenth_trading_day():
    days = decision_days(date(2026, 1, 2), date(2026, 10, 12))
    assert len(days) == 20
    assert (days[0], days[1], days[-1]) == (date(2026, 1, 2), date(2026, 1, 16), date(2026, 10, 6))
    assert decision_days(date(2026, 1, 1), date(2026, 10, 12)) == days  # New Year's Day is shut
    assert decision_days(date(2026, 1, 2), date(2026, 1, 15)) == [date(2026, 1, 2)]
    assert decision_days(date(2026, 1, 2), date(2026, 1, 9), every=5) == [
        date(2026, 1, 2), date(2026, 1, 9),
    ]  # fmt: skip
    assert decision_days(date(2026, 10, 3), date(2026, 10, 4)) == []


def test_the_grid_is_in_time_order_with_canonical_tickers():
    cells = grid(["nvda", "GOOG", "GOOGL"], reversed(DAYS), repeats=2)
    assert [(c.day, c.ticker, c.repeat) for c in cells] == [
        (day, ticker, repeat) for day in DAYS for ticker in ("NVDA", "GOOGL") for repeat in (0, 1)
    ]


def test_states_of_different_configs_are_kept_apart(tmp_path):
    cell = Cell(ticker="NVDA", day=DAYS[0], repeat=1)
    assert state_path(tmp_path, CONFIG, cell) == (
        tmp_path / "t1-abc12345" / "NVDA" / "2026-09-16_r1.json"
    )
    other = CONFIG.model_copy(update={"digest": "ffff0000"})
    assert run_dir(tmp_path, other) != run_dir(tmp_path, CONFIG)


# --- readiness ---


def test_missing_statements_and_prices_are_reported_before_running(world):
    harness = Harness(world, any_news_model())
    cells = grid([*TICKERS, "KO"], DAYS)
    assert unprepared(grid(TICKERS, DAYS), harness.statements, harness.prices, "SPY") == []
    assert unprepared(cells, harness.statements, harness.prices, "SPY") == [
        "KO: no statements stored (sluice fundamentals fetch --tickers KO)",
        "KO: no prices up to 2026-09-30",
    ]
    assert unprepared(grid(TICKERS, DAYS), harness.statements, harness.prices, "QQQ") == [
        "QQQ: no prices up to 2026-09-30"
    ]
    assert unprepared([], harness.statements, harness.prices, "SPY") == []


# --- running ---


def test_every_cell_is_run_saved_and_listed(world):
    harness = Harness(world, CitingModel(manager=[verdict("P:ret_5d")]))
    results = harness.run(grid(TICKERS, DAYS))

    assert [(r.day, r.ticker, r.status, r.rating) for r in results] == [
        (day, ticker, "ok", "overweight") for day in DAYS for ticker in TICKERS
    ]
    for cell in grid(TICKERS, DAYS):
        saved = json.loads(state_path(world / "runs", CONFIG, cell).read_text())
        assert list(saved) == ["meta", "inputs", "reports", "debate", "decision", "usage"]
        assert saved["meta"]["ticker"] == cell.ticker
        assert saved["meta"]["decision_time"][:10] == cell.day.isoformat()
        assert len(saved["inputs"]["news"]) == 1

    lines = (run_dir(world / "runs", CONFIG) / RATINGS_FILE).read_text().splitlines()
    listed = [json.loads(line) for line in lines]
    assert [(row["day"], row["ticker"]) for row in listed] == [
        (day.isoformat(), ticker) for day in DAYS for ticker in sorted(TICKERS)
    ]
    first = listed[0]
    assert (first["status"], first["rating"], first["confidence"]) == ("ok", "overweight", 0.6)
    assert (first["calls"], first["gaps"]) == (4, 0)
    assert first["tokens"] == {"fast-model": [300, 60], "deep-model": [100, 20]}


def test_each_cell_only_sees_news_from_before_its_own_day(world):
    harness = Harness(world, CitingModel(manager=[verdict("P:ret_5d")]))
    harness.run(grid(["NVDA"], DAYS))
    early = json.loads(state_path(world / "runs", CONFIG, Cell(ticker="NVDA", day=DAYS[0])).read_text())
    late = json.loads(state_path(world / "runs", CONFIG, Cell(ticker="NVDA", day=DAYS[1])).read_text())
    assert len(early["inputs"]["news"]) == 1 and len(late["inputs"]["news"]) == 1
    assert early["inputs"]["news"][0]["ref"] != late["inputs"]["news"][0]["ref"]
    assert early["inputs"]["prices_as_of"] == "2026-09-16"


def test_a_second_batch_makes_no_calls_and_gives_the_same_files(world):
    model = CitingModel(manager=[verdict("P:ret_5d")])
    cells = grid(TICKERS, DAYS)
    first = Harness(world, model).run(cells)
    files = {p: p.read_text() for p in (world / "runs").rglob("*.json*")}
    asked = len(model.requests)

    again = Harness(world, model).run(cells)
    assert len(model.requests) == asked == 16
    assert again == first
    assert {p: p.read_text() for p in (world / "runs").rglob("*.json*")} == files


def test_repeats_are_separate_runs(world):
    model = CitingModel(manager=[verdict("P:ret_5d")])
    results = Harness(world, model).run(grid(["NVDA"], DAYS[:1], repeats=3))
    assert [r.repeat for r in results] == [0, 1, 2]
    assert len({r.run_id for r in results}) == 3
    assert len(model.requests) == 12


def test_workers_do_not_change_the_results(world):
    cells = grid(TICKERS, DAYS, repeats=2)
    harness = Harness(world, CitingModel(manager=[verdict("P:ret_5d")]))
    parallel = harness.run(cells, workers=4)
    assert 1 <= harness.graphs <= 4
    assert [r.status for r in parallel] == ["ok"] * 8

    (world / "serial").mkdir()
    for name in ("news.db", "prices.db"):
        (world / "serial" / name).write_bytes((world / name).read_bytes())
    for ticker in TICKERS:
        stock(FundamentalsStore(world / "serial" / "statements"), ticker)
    serial = Harness(world / "serial", CitingModel(manager=[verdict("P:ret_5d")])).run(cells)
    assert [r.model_dump(exclude={"tokens"}) for r in serial] == [
        r.model_dump(exclude={"tokens"}) for r in parallel
    ]


def test_a_failed_run_is_saved_and_listed_without_a_rating(world):
    harness = Harness(world, CitingModel(manager=["not json"]))
    result = harness.run(grid(["NVDA"], DAYS[:1]))[0]
    assert (result.status, result.rating) == ("failed", None)
    assert result.failure.startswith("research_manager:")
    saved = json.loads(state_path(world / "runs", CONFIG, Cell(ticker="NVDA", day=DAYS[0])).read_text())
    assert saved["decision"] is None
    assert write_ratings(world / "runs", CONFIG)[0].status == "failed"


def test_a_run_that_breaks_off_is_reported_and_the_rest_goes_on(world):
    model = CitingModel(manager=[RuntimeError("network down"), verdict("P:ret_5d")])
    cells = grid(TICKERS, DAYS[:1])
    results = Harness(world, model).run(cells)

    assert [r.status for r in results] == ["error", "ok"]
    assert results[0].failure == "RuntimeError: network down"
    assert not state_path(world / "runs", CONFIG, cells[0]).exists()
    assert len(write_ratings(world / "runs", CONFIG)) == 1

    # rerunning finishes the broken run from its checkpoint: only the manager is asked
    asked = len(model.requests)
    again = Harness(world, model).run(cells)
    assert [r.status for r in again] == ["ok", "ok"]
    assert len(model.requests) == asked + 1
    assert len(write_ratings(world / "runs", CONFIG)) == 2


def test_a_day_without_trading_breaks_off_only_its_own_cell(world):
    harness = Harness(world, CitingModel(manager=[verdict("P:ret_5d")]))
    results = harness.run([Cell(ticker="NVDA", day=date(2026, 9, 26)), Cell(ticker="NVDA", day=DAYS[1])])
    assert [r.status for r in results] == ["error", "ok"]
    assert "not a trading day" in results[0].failure


def test_cost_adds_up_tokens_by_model():
    tokens = {"deepseek-flash": (1_000_000, 500_000), "unknown-model": (5, 5)}
    results = [
        CellResult(ticker="NVDA", day=DAYS[0], repeat=repeat, status="ok", tokens=tokens)
        for repeat in range(2)
    ]
    totals, cost = estimated_cost(results)
    assert totals["deepseek-flash"] == (2_000_000, 1_000_000)
    assert cost == pytest.approx(2 * 0.3 + 1 * 1.2)
