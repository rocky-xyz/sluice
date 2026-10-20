from datetime import date

from sluice.patterns.prices import BENCHMARK, fetch_all
from sluice.prices.store import HISTORY_START, PriceStore
from tests.prices.series import series

THROUGH = date(2026, 10, 16)


class Yahoo:
    def __init__(self, broken=()):
        self.asked, self.broken = [], set(broken)

    def __call__(self, ticker, start):
        self.asked.append((ticker, start))
        if ticker in self.broken:
            raise ConnectionError("reset by peer")
        if ticker == "NOPE":
            return []
        return series(THROUGH, 30)


def test_every_symbol_and_spy_are_fetched_from_the_chosen_start(tmp_path):
    yahoo = Yahoo()
    store = PriceStore(tmp_path / "p.db", fetch=yahoo, history_start=date(2016, 1, 1))
    result = fetch_all(store, ["AAA", "NOPE", "GOOG"], THROUGH, sleep=lambda s: None)
    assert [t for t, _ in yahoo.asked] == [BENCHMARK, "AAA", "NOPE", "GOOGL"]
    assert {s for _, s in yahoo.asked} == {date(2016, 1, 1)}
    assert (result.fetched, result.empty, result.failed) == (3, ["NOPE"], {})


def test_a_rerun_skips_what_is_current_and_a_failure_does_not_stop_the_list(tmp_path):
    store = PriceStore(tmp_path / "p.db", fetch=Yahoo(broken={"BBB"}))
    first = fetch_all(store, ["AAA", "BBB", "CCC"], THROUGH, sleep=lambda s: None)
    assert first.fetched == 3 and list(first.failed) == ["BBB"]
    again = Yahoo()
    store._fetch = again
    second = fetch_all(store, ["AAA", "BBB", "CCC"], THROUGH, sleep=lambda s: None)
    assert [t for t, _ in again.asked] == ["BBB"] and second.current == 3


def test_the_pipelines_store_still_starts_in_2024(tmp_path):
    yahoo = Yahoo()
    PriceStore(tmp_path / "p.db", fetch=yahoo).history("AAA", THROUGH, 1)
    assert yahoo.asked == [("AAA", HISTORY_START)] and HISTORY_START == date(2024, 1, 1)
