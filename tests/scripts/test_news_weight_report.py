import csv
from datetime import UTC, datetime

import pytest

from scripts.news_weight_report import (
    _other_row,
    render_report,
    write_event_type_csv,
    write_source_csv,
)
from sluice.news.walkforward import Refit
from sluice.news.weights import WeightEstimate, WeightTable


def _table() -> WeightTable:
    return WeightTable(
        lambda0=10.0,
        source={
            "Reuters": WeightEstimate(weight=0.02, low=0.01, high=0.03, n=500),
            "Tiny Blog A": WeightEstimate(weight=0.09, low=-0.1, high=0.2, n=3),
            "Tiny Blog B": WeightEstimate(weight=-0.03, low=-0.2, high=0.1, n=5),
        },
        event_type={
            "earnings": WeightEstimate(weight=-0.01, low=-0.02, high=0.0, n=1000),
            "guidance": WeightEstimate(weight=0.002, low=-0.001, high=0.005, n=200),
        },
    )


def test_other_row_aggregates_the_small_sources():
    table = _table()
    small = [e for e in table.source.values() if e.n < 20]
    other = _other_row(small)
    assert other["count"] == 2
    assert other["total_n"] == 8
    # weighted mean = (0.09*3 + -0.03*5) / 8
    assert other["weighted_mean"] == pytest.approx((0.09 * 3 + -0.03 * 5) / 8)
    assert other["min"] == pytest.approx(-0.03)
    assert other["max"] == pytest.approx(0.09)


def test_render_report_merges_sparse_sources_and_lists_the_rest():
    table = _table()
    refits = [
        Refit(as_of=datetime(2026, 1, 1, tzinfo=UTC), n_rows=100, table=None),
        Refit(as_of=datetime(2026, 2, 1, tzinfo=UTC), n_rows=508, table=table),
    ]
    report = render_report(datetime(2026, 2, 1, tzinfo=UTC), table, refits, min_source_n=20)

    assert "Reuters" in report
    assert "Tiny Blog A" not in report
    assert "Tiny Blog B" not in report
    assert "其他（2 个来源合并）" in report
    assert "earnings" in report and "guidance" in report
    assert "2026-01-01" in report and "2026-02-01" in report
    assert "n/a（样本不足）" in report  # the None-table month


def test_csv_writers_round_trip(tmp_path):
    table = _table()
    source_path = tmp_path / "sources.csv"
    type_path = tmp_path / "types.csv"
    write_source_csv(source_path, table)
    write_event_type_csv(type_path, table)

    with source_path.open() as f:
        rows = {row["source"]: row for row in csv.DictReader(f)}
    assert rows["Reuters"]["n"] == "500"
    assert float(rows["Reuters"]["weight"]) == pytest.approx(0.02)

    with type_path.open() as f:
        rows = {row["event_type"]: row for row in csv.DictReader(f)}
    assert rows["guidance"]["n"] == "200"
