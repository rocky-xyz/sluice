from datetime import UTC, date, datetime

import pytest

from sluice.agents import fundamentals
from sluice.agents.config import FundamentalsConfig
from sluice.agents.fundamentals import (
    compute,
    free_cash_flow,
    quarter_figures,
    valuation,
    visible,
    visible_from,
    year_before,
)
from sluice.agents.prefetch import prefetch_fundamentals
from sluice.fundamentals.store import FundamentalsStore, Quarter
from sluice.prices.store import PriceStore
from tests.fundamentals.statements import ENDS, answers, stock
from tests.prices.series import series

CONFIG = FundamentalsConfig(lag_days=60, quarters=4)
DAY = date(2026, 9, 30)
CLOSE = datetime(2026, 9, 30, 20, 0, tzinfo=UTC)


def quarter(end, revenue=1000.0, **fields) -> Quarter:
    base = {
        "gross_profit": revenue * 0.6,
        "operating_income": revenue * 0.3,
        "net_income": revenue * 0.2,
        "operating_cashflow": revenue * 0.25,
        "capex": revenue * 0.05,
        "shares": 100.0,
        "currency": "USD",
    }
    return Quarter(end=end, revenue=revenue, **(base | fields))


def eight_quarters() -> list[Quarter]:
    """Oldest first; revenue 1000, 1100, ... 1700."""
    return [quarter(end, 1000.0 + 100 * i) for i, end in enumerate(sorted(ENDS)[2:])]


# --- what is visible ---


def test_a_quarter_becomes_visible_sixty_days_after_it_ends():
    june = quarter(date(2026, 6, 30))
    assert visible_from(june, 60) == date(2026, 8, 29)
    assert visible([june], date(2026, 8, 28), 60) == []
    assert visible([june], date(2026, 8, 29), 60) == [june]


def test_visible_quarters_come_oldest_first():
    quarters = eight_quarters()
    seen = visible(list(reversed(quarters)), date(2026, 3, 15), 60)
    assert [q.end for q in seen] == [q.end for q in quarters[:6]]  # up to 2025-12-31


# --- the figures ---


def test_the_same_quarter_a_year_earlier_is_found_by_date():
    quarters = eight_quarters()
    assert year_before(quarters, quarters[-1]) == quarters[-5]
    assert year_before(quarters, quarters[3]) is None
    # fiscal quarters ending on the last Sunday of a month drift by a few days
    drifting = [quarter(date(2025, 7, 27)), quarter(date(2026, 7, 26))]
    assert year_before(drifting, drifting[1]) == drifting[0]


def test_figures_of_one_quarter():
    now = quarter(date(2026, 6, 30), 1200.0, gross_profit=780.0, capex=100.0)
    then = quarter(date(2025, 6, 30), 1000.0)
    figures = quarter_figures(now, then)

    assert figures["revenue"] == 1200.0
    assert figures["gross_margin"] == pytest.approx(0.65)
    assert figures["operating_margin"] == pytest.approx(0.30)
    assert figures["fcf"] == pytest.approx(1200 * 0.25 - 100)
    assert figures["revenue_yoy"] == pytest.approx(0.20)
    assert figures["gross_margin_yoy"] == pytest.approx(0.05)  # 65% - 60%, a difference
    assert figures["operating_margin_yoy"] == pytest.approx(0.0)
    assert figures["net_income_yoy"] == pytest.approx(0.20)
    assert figures["fcf_yoy"] == pytest.approx(200 / 200 - 1)


def test_without_the_year_before_there_is_no_change_and_missing_parts_drop_figures():
    alone = quarter_figures(quarter(date(2026, 6, 30), capex=None), None)
    assert set(alone) == {"revenue", "gross_margin", "operating_margin", "net_income"}
    assert free_cash_flow(quarter(date(2026, 6, 30), operating_cashflow=None)) is None


def test_growth_from_a_loss_is_not_computed():
    now = quarter(date(2026, 6, 30), net_income=50.0)
    then = quarter(date(2025, 6, 30), net_income=-20.0)
    figures = quarter_figures(now, then)
    assert "net_income_yoy" not in figures
    assert "revenue_yoy" in figures


def test_valuation_is_market_value_over_four_quarters():
    last_four = eight_quarters()[-4:]  # revenue 1400 + 1500 + 1600 + 1700 = 6200
    figures = valuation(last_four, close=124.0)  # market value 100 shares * 124
    assert figures["ps_ttm"] == pytest.approx(12400 / 6200)
    assert figures["pe_ttm"] == pytest.approx(12400 / (6200 * 0.2))


def test_valuation_needs_four_dollar_quarters_in_a_row_a_price_and_a_profit():
    quarters = eight_quarters()
    assert valuation(quarters[-3:], 124.0) == {}
    assert valuation(quarters[-4:], None) == {}
    assert valuation([q.model_copy(update={"currency": "TWD"}) for q in quarters[-4:]], 124.0) == {}
    assert valuation([quarters[0], *quarters[-3:]], 124.0) == {}  # a year's gap inside
    losing = [q.model_copy(update={"net_income": -1.0}) for q in quarters[-4:]]
    assert set(valuation(losing, 124.0)) == {"ps_ttm"}


def test_a_full_history_gives_every_figure_in_order():
    figures = compute(eight_quarters(), 124.0, quarters=4)
    assert list(fundamentals.names(4)) == [
        *(f"{name}_q{i}" for i in range(4) for name, _ in fundamentals.QUARTER_FIGURES),
        "pe_ttm",
        "ps_ttm",
    ]
    assert set(figures) == set(fundamentals.names(4))
    assert len(figures) == 42
    assert figures["revenue_q0"][0] == 1700.0 and figures["revenue_q3"][0] == 1400.0
    assert figures["revenue_q0"][1].end == date(2026, 6, 30)
    assert figures["revenue_yoy_q3"][0] == pytest.approx(1400 / 1000 - 1)


def test_a_short_history_gives_fewer_figures():
    figures = compute(eight_quarters()[:2], 124.0, quarters=4)
    assert set(figures) == {
        f"{name}_q{i}"
        for i in range(2)
        for name in ("revenue", "gross_margin", "operating_margin", "net_income", "fcf")
    }


# --- prefetch ---


@pytest.fixture
def prices(tmp_path):
    with PriceStore(tmp_path / "prices.db") as prices:
        prices.replace("NVDA", series(date(2026, 10, 9), 30))
        yield prices


@pytest.fixture
def statements(tmp_path):
    store = FundamentalsStore(tmp_path / "statements")
    stock(store, "NVDA")
    return store


def test_only_published_quarters_are_shown(statements, prices):
    shown, gaps = prefetch_fundamentals(statements, prices, "NVDA", CLOSE, CONFIG)
    assert gaps == []
    assert len(shown) == 42
    assert {item.quarter_end for item in shown} == set(ENDS[:4])
    assert all(item.visible_from <= DAY for item in shown)
    first = shown[0]
    assert (first.ref, first.unit, first.currency) == ("F:revenue_q0", "amount", "USD")
    assert (first.quarter_end, first.visible_from) == (date(2026, 6, 30), date(2026, 8, 29))


def test_the_day_before_a_quarter_is_published_it_is_not_used(statements, prices):
    eve = datetime(2026, 8, 28, 20, 0, tzinfo=UTC)
    shown, _ = prefetch_fundamentals(statements, prices, "NVDA", eve, CONFIG)
    latest = next(item for item in shown if item.name == "revenue_q0")
    assert latest.quarter_end == date(2026, 3, 31)

    on_the_day = datetime(2026, 8, 31, 20, 0, tzinfo=UTC)  # the 29th is a Saturday
    shown, _ = prefetch_fundamentals(statements, prices, "NVDA", on_the_day, CONFIG)
    assert next(i for i in shown if i.name == "revenue_q0").quarter_end == date(2026, 6, 30)


def test_changing_unpublished_quarters_changes_nothing(tmp_path, prices):
    early = datetime(2026, 3, 16, 20, 0, tzinfo=UTC)  # sees quarters up to 2025-12-31
    store = FundamentalsStore(tmp_path / "a")
    stock(store, "NVDA")
    before = prefetch_fundamentals(store, prices, "NVDA", early, CONFIG)

    changed = answers()
    for rows in changed.values():
        for row in rows["quarterlyReports"][:2]:  # 2026-06-30 and 2026-03-31
            for field in row:
                if field not in ("fiscalDateEnding", "reportedCurrency"):
                    row[field] = "1"
    for function, answer in changed.items():
        store.save(function, "NVDA", answer)
    assert prefetch_fundamentals(store, prices, "NVDA", early, CONFIG) == before

    # and the check has teeth: a published quarter does matter
    changed["INCOME_STATEMENT"]["quarterlyReports"][2]["totalRevenue"] = "1"
    store.save("INCOME_STATEMENT", "NVDA", changed["INCOME_STATEMENT"])
    assert prefetch_fundamentals(store, prices, "NVDA", early, CONFIG) != before


def test_valuation_uses_the_unadjusted_close_of_the_last_finished_day(statements, prices):
    shown, _ = prefetch_fundamentals(statements, prices, "NVDA", CLOSE, CONFIG)
    bar = prices.history("NVDA", DAY, limit=1)[0]
    revenue = sum(item.value for item in shown if item.name.startswith("revenue_q"))
    ps = next(item for item in shown if item.name == "ps_ttm")
    assert ps.value == round(1e9 * bar.raw_close / revenue, 4)
    assert (ps.unit, ps.currency, ps.quarter_end) == ("multiple", None, date(2026, 6, 30))


def test_gaps_are_recorded(tmp_path, prices):
    store = FundamentalsStore(tmp_path / "gaps")
    nothing = prefetch_fundamentals(store, prices, "NVDA", CLOSE, CONFIG)
    assert nothing == ([], [nothing[1][0]]) and "no statements stored" in nothing[1][0].reason

    stock(store, "NVDA")
    too_early = datetime(2024, 5, 1, 20, 0, tzinfo=UTC)
    assert prefetch_fundamentals(store, prices, "NVDA", too_early, CONFIG)[0] == []

    stock(store, "TSM", currency="TWD")
    shown, gaps = prefetch_fundamentals(store, prices, "TSM", CLOSE, CONFIG)
    assert len(shown) == 40 and shown[0].currency == "TWD"
    assert [gap.reason for gap in gaps] == [
        "no valuation: statements are in TWD",
        "could not compute: pe_ttm, ps_ttm",
    ]

    stock(store, "NEWCO", ends=ENDS[:2])
    shown, gaps = prefetch_fundamentals(store, prices, "NEWCO", CLOSE, CONFIG)
    assert len(shown) == 10
    assert gaps[0].reason == "only 2 of 4 quarters are published"
