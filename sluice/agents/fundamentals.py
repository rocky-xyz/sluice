from collections.abc import Sequence
from datetime import date, timedelta

from sluice.fundamentals.store import Quarter

PRICE_CURRENCY = "USD"  # prices are in dollars; statements in another currency cannot be valued
YEAR_TOLERANCE = timedelta(days=20)  # fiscal quarters do not end on the same date every year

# Per quarter, in the order shown. "amount" is in the statement's currency.
QUARTER_FIGURES = (
    ("revenue", "amount"),
    ("revenue_yoy", "ratio"),
    ("gross_margin", "ratio"),
    ("gross_margin_yoy", "points"),
    ("operating_margin", "ratio"),
    ("operating_margin_yoy", "points"),
    ("net_income", "amount"),
    ("net_income_yoy", "ratio"),
    ("fcf", "amount"),
    ("fcf_yoy", "ratio"),
)
VALUATION_FIGURES = ("pe_ttm", "ps_ttm")


def names(quarters: int) -> list[str]:
    """Every figure a full history gives, in the order shown."""
    per_quarter = [f"{name}_q{i}" for i in range(quarters) for name, _ in QUARTER_FIGURES]
    return [*per_quarter, *VALUATION_FIGURES]


def visible_from(quarter: Quarter, lag_days: int) -> date:
    """The first day a quarter may be used: its last day plus a fixed lag.

    The vendor gives no publication date, and a quarter is published weeks after
    it ends, so using it from its end date would leak.
    """
    return quarter.end + timedelta(days=lag_days)


def visible(quarters: Sequence[Quarter], decision_day: date, lag_days: int) -> list[Quarter]:
    """The quarters that count as published on `decision_day`, oldest first."""
    return sorted(
        (q for q in quarters if visible_from(q, lag_days) <= decision_day), key=lambda q: q.end
    )


def year_before(quarters: Sequence[Quarter], quarter: Quarter) -> Quarter | None:
    """The same fiscal quarter one year earlier."""
    target = quarter.end - timedelta(days=365)
    near = [q for q in quarters if abs(q.end - target) <= YEAR_TOLERANCE]
    return min(near, key=lambda q: abs(q.end - target)) if near else None


def _ratio(top: float | None, bottom: float | None) -> float | None:
    if top is None or bottom is None or bottom <= 0:
        return None
    return top / bottom


def _growth(now: float | None, then: float | None) -> float | None:
    """now / then - 1; undefined when the earlier value is not positive."""
    ratio = _ratio(now, then)
    return None if ratio is None else ratio - 1


def _difference(now: float | None, then: float | None) -> float | None:
    return None if now is None or then is None else now - then


def free_cash_flow(quarter: Quarter) -> float | None:
    """Operating cash flow minus capital expenditure."""
    return _difference(quarter.operating_cashflow, quarter.capex)


def quarter_figures(quarter: Quarter, earlier: Quarter | None) -> dict[str, float]:
    """The figures of one quarter; year-on-year ones only with the quarter a year before."""
    gross = _ratio(quarter.gross_profit, quarter.revenue)
    operating = _ratio(quarter.operating_income, quarter.revenue)
    figures = {
        "revenue": quarter.revenue,
        "gross_margin": gross,
        "operating_margin": operating,
        "net_income": quarter.net_income,
        "fcf": free_cash_flow(quarter),
    }
    if earlier is not None:
        figures |= {
            "revenue_yoy": _growth(quarter.revenue, earlier.revenue),
            "gross_margin_yoy": _difference(gross, _ratio(earlier.gross_profit, earlier.revenue)),
            "operating_margin_yoy": _difference(
                operating, _ratio(earlier.operating_income, earlier.revenue)
            ),
            "net_income_yoy": _growth(quarter.net_income, earlier.net_income),
            "fcf_yoy": _growth(free_cash_flow(quarter), free_cash_flow(earlier)),
        }
    return {name: value for name, value in figures.items() if value is not None}


def valuation(last_four: Sequence[Quarter], close: float | None) -> dict[str, float]:
    """Market value over the last four quarters' net income and revenue.

    Market value is the share count of the latest quarter times `close`, the
    unadjusted close of the last finished trading day. Needs four quarters in
    dollars; a multiple of a sum that is not positive is left out.
    """
    if len(last_four) < 4 or close is None:
        return {}
    latest = last_four[-1]
    if latest.end - last_four[0].end > timedelta(days=300):
        return {}  # a quarter is missing in between: the four do not make a year
    if latest.shares is None or any(q.currency != PRICE_CURRENCY for q in last_four):
        return {}
    market_value = latest.shares * close
    figures = {}
    for name, field in (("pe_ttm", "net_income"), ("ps_ttm", "revenue")):
        parts = [getattr(q, field) for q in last_four]
        if None not in parts and (multiple := _ratio(market_value, sum(parts))) is not None:
            figures[name] = multiple
    return figures


def compute(
    seen: Sequence[Quarter], close: float | None, quarters: int
) -> dict[str, tuple[float, Quarter]]:
    """Every figure that can be computed, with the quarter it belongs to.

    `seen` must already be cut to what is visible: nothing here looks at dates
    except to find the same quarter a year earlier among them.
    """
    figures: dict[str, tuple[float, Quarter]] = {}
    for i, quarter in enumerate(reversed(seen[-quarters:])):
        for name, value in quarter_figures(quarter, year_before(seen, quarter)).items():
            figures[f"{name}_q{i}"] = (value, quarter)
    for name, value in valuation(seen[-4:], close).items():
        figures[name] = (value, seen[-1])
    return figures
