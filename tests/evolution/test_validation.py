import json
from datetime import date

import pytest

from sluice.agents.batch import run_dir
from sluice.agents.config import load_config
from sluice.evaluation.report import calendar
from sluice.evolution.validation import Validation
from sluice.news.store import NewsStore
from sluice.prices.store import PriceStore
from sluice.scoring.reports import ReportScoreStore
from sluice.scoring.store import ScoreStore
from sluice.sizing import plan
from sluice.sizing.config import load_sizing_config
from tests.prices.series import series

TICKERS = ["AAA", "BBB"]
LAST = date(2026, 8, 14)  # past the validation period's last close


def write_runs(folder, cal, rating_of):
    for day in cal.decisions:
        for ticker in TICKERS:
            for repeat in range(3):
                path = folder / ticker / f"{day}_r{repeat}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "meta": {"run_id": f"{ticker}-{day}-{repeat}"},
                    "decision": {"rating": rating_of(ticker), "confidence": 0.5},
                }
                path.write_text(json.dumps(state))


@pytest.fixture
def world(tmp_path):
    cal = calendar("validation")
    with (
        PriceStore(tmp_path / "prices.db") as prices,
        NewsStore(tmp_path / "news.db") as news,
        ScoreStore(tmp_path / "scores.db") as scores,
        ReportScoreStore(tmp_path / "reports.db") as reports,
    ):
        prices.replace("AAA", series(LAST, 400, drift=0.002))
        prices.replace("BBB", series(LAST, 400, drift=-0.001, start=50))
        prices.replace("SPY", series(LAST, 400, drift=0.0005))
        yield cal, prices, plan.Sources(news, scores, [], reports, prices), tmp_path / "runs"


def test_the_portfolio_can_be_built_from_any_pipelines_ratings(world):
    cal, prices, src, runs = world
    sizing = load_sizing_config("v1").model_copy(
        update={"stocks": load_sizing_config("v1").stocks.model_copy(update={"max_weight": 1.0})}
    )  # two stocks cannot sum to 1 under a 20% cap
    v1, v3 = load_config("v1"), load_config("v3")
    write_runs(run_dir(runs, v1), cal, lambda t: "hold")
    write_runs(run_dir(runs, v3), cal, lambda t: "overweight" if t == "AAA" else "underweight")
    by_v1 = Validation(src, prices, TICKERS, runs)(sizing)
    by_v3 = Validation(src, prices, TICKERS, runs, pipeline=v3)(sizing)
    assert sorted(by_v1) == sorted(by_v3) == [0, 1, 2]
    assert len(by_v3[0]) == len(cal.days)
    assert by_v1[0] != by_v3[0]  # leaning to AAA changes the portfolio


def test_a_pipeline_with_runs_missing_is_refused_by_name(world):
    cal, prices, src, runs = world
    v3 = load_config("v3")
    write_runs(run_dir(runs, v3), cal, lambda t: "hold")
    (run_dir(runs, v3) / "AAA" / f"{cal.decisions[0]}_r2.json").unlink()
    with pytest.raises(ValueError, match=f"v3-{v3.digest} repeat 2 has 29 of 30"):
        Validation(src, prices, TICKERS, runs, pipeline=v3)
