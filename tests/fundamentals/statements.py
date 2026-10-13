from datetime import date

from sluice.fundamentals.alphavantage import FUNCTIONS
from sluice.fundamentals.store import FundamentalsStore

# Fiscal quarter ends, newest first, like the vendor's answers.
ENDS = [
    date(2026, 6, 30), date(2026, 3, 31), date(2025, 12, 31), date(2025, 9, 30),
    date(2025, 6, 30), date(2025, 3, 31), date(2024, 12, 31), date(2024, 9, 30),
    date(2024, 6, 30), date(2024, 3, 31),
]  # fmt: skip


def answers(ends=ENDS, currency="USD", shares=1_000_000_000) -> dict[str, dict]:
    """Made-up statements: revenue grows 5% a quarter from 1,000m at the oldest quarter."""
    reports = {function: [] for function in FUNCTIONS}
    for i, end in enumerate(ends):
        revenue = round(1_000_000_000 * 1.05 ** (len(ends) - 1 - i))
        common = {"fiscalDateEnding": end.isoformat(), "reportedCurrency": currency}
        reports["INCOME_STATEMENT"].append(
            common
            | {
                "totalRevenue": str(revenue),
                "grossProfit": str(round(revenue * 0.6)),
                "operatingIncome": str(round(revenue * 0.3)),
                "netIncome": str(round(revenue * 0.2)),
            }
        )
        reports["CASH_FLOW"].append(
            common
            | {
                "operatingCashflow": str(round(revenue * 0.25)),
                "capitalExpenditures": str(round(revenue * 0.05)),
            }
        )
        reports["BALANCE_SHEET"].append(common | {"commonStockSharesOutstanding": str(shares)})
    return {
        function: {"symbol": "X", "quarterlyReports": rows} for function, rows in reports.items()
    }


def stock(store: FundamentalsStore, ticker: str, **kwargs) -> None:
    for function, answer in answers(**kwargs).items():
        store.save(function, ticker, answer)
