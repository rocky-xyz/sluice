from datetime import UTC, datetime

from sluice.news.walkforward import Refit
from sluice.news.weight_tables import (
    StoredTable,
    fit_missing,
    latest_before,
    load_all,
    path_for,
    save,
)
from sluice.news.weights import WeightEstimate, WeightTable

SEP = datetime(2026, 9, 1, tzinfo=UTC)
OCT = datetime(2026, 10, 1, tzinfo=UTC)


def _table(weight: float = 0.01) -> WeightTable:
    return WeightTable(
        lambda0=100.0,
        source={"Reuters": WeightEstimate(weight=weight, low=0.001, high=0.02, n=120)},
        event_type={"earnings": WeightEstimate(weight=-0.002, low=-0.004, high=-0.001, n=80)},
    )


def _stored(as_of: datetime) -> StoredTable:
    return StoredTable(as_of=as_of, n_rows=1000, table=_table())


def test_a_saved_table_loads_back_unchanged(tmp_path):
    save(tmp_path, "v1", SEP, 1234, _table())
    [loaded] = load_all(tmp_path, "v1")
    assert loaded == StoredTable(as_of=SEP, n_rows=1234, table=_table())
    assert path_for(tmp_path, "v1", SEP) == tmp_path / "v1" / "2026-09-01.json"


def test_tables_are_kept_apart_by_scoring_config(tmp_path):
    save(tmp_path, "v1", SEP, 1, _table())
    assert load_all(tmp_path, "v2") == []


def test_load_all_is_oldest_first(tmp_path):
    save(tmp_path, "v1", OCT, 1, _table())
    save(tmp_path, "v1", SEP, 1, _table())
    assert [t.as_of for t in load_all(tmp_path, "v1")] == [SEP, OCT]


def test_latest_before_picks_the_latest_month_start_not_after_the_moment():
    tables = [_stored(SEP), _stored(OCT)]
    assert latest_before(tables, datetime(2026, 9, 30, 20, tzinfo=UTC)).as_of == SEP
    assert latest_before(tables, OCT).as_of == OCT  # fitted as of that very moment: usable
    assert latest_before(tables, datetime(2026, 10, 20, tzinfo=UTC)).as_of == OCT


def test_latest_before_the_first_table_is_none():
    assert latest_before([_stored(SEP)], datetime(2026, 8, 31, tzinfo=UTC)) is None
    assert latest_before([], OCT) is None


def test_fit_missing_writes_new_months_and_skips_too_thin_ones(tmp_path):
    def fit(todo):
        return [Refit(SEP, 100, None), Refit(OCT, 900, _table())]

    result = fit_missing([SEP, OCT], tmp_path, "v1", fit)
    assert (result.written, result.too_few, result.kept) == ([OCT], [SEP], [])
    assert [t.as_of for t in load_all(tmp_path, "v1")] == [OCT]


def test_saved_months_are_frozen_unless_refreshed(tmp_path):
    save(tmp_path, "v1", SEP, 1000, _table(weight=0.01))
    asked = []

    def fit(todo):
        asked.append(list(todo))
        return [Refit(m, 2000, _table(weight=0.05)) for m in todo]

    result = fit_missing([SEP, OCT], tmp_path, "v1", fit)
    assert asked == [[OCT]]  # the saved month is not even refitted
    assert (result.kept, result.written) == ([SEP], [OCT])
    assert load_all(tmp_path, "v1")[0].table.source["Reuters"].weight == 0.01

    fit_missing([SEP, OCT], tmp_path, "v1", fit, refresh=True)
    assert load_all(tmp_path, "v1")[0].table.source["Reuters"].weight == 0.05


def test_nothing_to_fit_does_not_call_fit(tmp_path):
    save(tmp_path, "v1", SEP, 1000, _table())

    def fit(todo):
        raise AssertionError("should not be called")

    assert fit_missing([SEP], tmp_path, "v1", fit).kept == [SEP]
