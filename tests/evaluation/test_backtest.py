import math
from datetime import date

import pytest

from sluice.evaluation import metrics
from sluice.evaluation.backtest import PriceBar, simulate
from sluice.evaluation.decisions import Decision
from sluice.evaluation.portfolio import target_weights
from sluice.evaluation.report import calendar, load_prices, rating_schedules
from sluice.evaluation.split import SealedError

D1, D2, D3, D4 = date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4), date(2026, 3, 5)


def bars(*pairs: tuple[float, float]) -> dict[date, PriceBar]:
    days = (D1, D2, D3, D4)
    return {day: PriceBar(open=o, close=c) for day, (o, c) in zip(days, pairs, strict=False)}


# --- ratings to weights ---


def test_all_hold_is_equal_weight():
    assert target_weights({"A": "hold", "B": "hold"}) == {"A": 0.5, "B": 0.5}


def test_multiples_are_normalised():
    weights = target_weights({"A": "buy", "B": "underweight", "C": "sell", "D": "overweight"})
    assert weights == pytest.approx({"A": 0.5, "B": 0.125, "C": 0.0, "D": 0.375})


def test_a_missing_rating_counts_as_hold_and_all_sell_falls_back_to_equal():
    assert target_weights({"A": None, "B": "buy"}) == pytest.approx({"A": 1 / 3, "B": 2 / 3})
    assert target_weights({"A": "sell", "B": "sell"}) == {"A": 0.5, "B": 0.5}


# --- the engine ---


def test_the_first_trade_buys_at_the_open_and_pays_for_it():
    prices = {"A": bars((100, 110))}
    path = simulate({D1: {"A": 1.0}}, [D1], prices, cost=0.001)
    assert path.turnover == [1.0]
    assert path.costs == pytest.approx(0.001)
    assert path.values[-1] == pytest.approx((1 - 0.001) * 1.1)


def test_weights_drift_between_rebalances():
    prices = {"A": bars((100, 100), (100, 120)), "B": bars((100, 100), (100, 90))}
    path = simulate({D1: {"A": 0.5, "B": 0.5}}, [D1, D2], prices, cost=0.0)
    assert path.values[-1] == pytest.approx(0.5 * 1.2 + 0.5 * 0.9)
    assert path.returns[-1] == pytest.approx(0.05)


def test_the_trade_is_at_the_open_so_an_overnight_jump_is_not_captured():
    # B jumps from 100 to 150 overnight into D2; switching into it at the D2 open misses that.
    prices = {"A": bars((100, 100), (100, 100)), "B": bars((100, 100), (150, 150))}
    path = simulate({D1: {"A": 1.0}, D2: {"B": 1.0}}, [D1, D2], prices, cost=0.0)
    assert path.values[-1] == pytest.approx(1.0)


def test_a_rebalance_is_charged_on_the_value_traded_at_the_open():
    prices = {"A": bars((100, 100), (110, 110)), "B": bars((100, 100), (90, 90))}
    path = simulate({D1: {"A": 0.5, "B": 0.5}, D2: {"A": 1.0}}, [D1, D2], prices, cost=0.01)
    at_open = 0.99 * (0.5 * 1.1 + 0.5 * 0.9)  # after the first trade's cost, gapped to the open
    held_a = 0.99 * 0.5 * 1.1 / at_open
    traded = (1.0 - held_a) + (1 - held_a)  # buy A up to 1, sell all of B
    assert path.turnover[1] == pytest.approx(traded)
    assert path.costs == pytest.approx(0.01 + 0.01 * traded * at_open)


def test_between_rebalances_the_value_follows_the_closes():
    prices = {"A": bars((100, 100), (100, 105), (105, 99))}
    path = simulate({D1: {"A": 1.0}}, [D1, D2, D3], prices, cost=0.0)
    assert path.values == pytest.approx([1.0, 1.0, 1.05, 0.99])


def test_the_first_day_must_rebalance():
    with pytest.raises(ValueError):
        simulate({D2: {"A": 1.0}}, [D1, D2], {"A": bars((1, 1), (1, 1))}, cost=0.0)


# --- metrics ---


def test_sharpe_sortino_and_drawdown_by_hand():
    returns = [0.01, -0.02, 0.03]
    mean = 0.02 / 3
    sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / 2)
    assert metrics.sharpe(returns) == pytest.approx(mean / sd * math.sqrt(252))
    assert metrics.sortino(returns) == pytest.approx(mean / math.sqrt(0.02**2 / 3) * math.sqrt(252))
    assert metrics.max_drawdown([1.0, 1.2, 0.9, 1.3, 1.17]) == pytest.approx(0.25)
    assert metrics.total_return([1.0, 1.1]) == pytest.approx(0.1)
    assert metrics.sharpe([0.01, 0.01]) is None  # no spread


# --- the calendar and its inputs ---


def test_the_validation_calendar():
    cal = calendar("validation")
    assert len(cal.decisions) == 15
    assert (cal.decisions[0], cal.decisions[-1]) == (date(2026, 1, 2), date(2026, 7, 27))
    assert all(r > d for d, r in zip(cal.decisions, cal.rebalances, strict=True))
    assert cal.rebalances[0] == date(2026, 1, 5)  # the next trading day after Friday 2 January
    assert cal.end == date(2026, 8, 10)
    assert (cal.days[0], cal.days[-1]) == (cal.rebalances[0], cal.end)


def test_a_test_calendar_is_sealed():
    with pytest.raises(SealedError):
        calendar("test_1")
    assert calendar("test_1", final=True).end == date(2026, 10, 20)


def test_ratings_become_one_schedule_per_repeat_with_holds_filled_in():
    cal = calendar("validation")
    first = cal.decisions[0]
    decisions = [
        Decision(ticker="A", day=first, repeat=0, config="v", run_id=None, rating="buy", confidence=0.6),
        Decision(ticker="B", day=first, repeat=0, config="v", run_id=None, rating=None, confidence=None),
        Decision(ticker="A", day=first, repeat=1, config="v", run_id=None, rating="sell", confidence=0.6),
    ]
    schedules = rating_schedules(decisions, ["A", "B"], cal)
    schedule, neutral = schedules[0]
    assert schedule[cal.rebalances[0]] == pytest.approx({"A": 2 / 3, "B": 1 / 3})
    assert schedule[cal.rebalances[1]] == {"A": 0.5, "B": 0.5}  # no runs at all on the second day
    assert neutral == 1 + 2 * 14  # B's failed run, then both tickers on 14 days without runs
    assert schedules[1][0][cal.rebalances[0]] == pytest.approx({"A": 0.0, "B": 1.0})


def test_all_hold_ratings_follow_the_equal_weight_baseline_exactly():
    cal = calendar("validation")
    decisions = [
        Decision(ticker=t, day=d, repeat=0, config="v", run_id=None, rating="hold", confidence=0.5)
        for d in cal.decisions
        for t in ("A", "B")
    ]
    schedule, neutral = rating_schedules(decisions, ["A", "B"], cal)[0]
    assert neutral == 0
    assert schedule == {day: {"A": 0.5, "B": 0.5} for day in cal.rebalances}


class _Store:
    def __init__(self, days):
        self.days = days

    def history(self, ticker, through, limit):
        from sluice.prices.store import Bar

        return [Bar(day=d, open=1, close=1, raw_close=1, volume=1) for d in self.days]


def test_a_price_gap_is_refused():
    cal = calendar("validation")
    with pytest.raises(ValueError, match="no prices for A"):
        load_prices(_Store(cal.days[:-1]), ["A"], cal)
    assert set(load_prices(_Store(cal.days), ["A"], cal)["A"]) == set(cal.days)
