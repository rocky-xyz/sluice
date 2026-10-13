import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from sluice.fundamentals.alphavantage import FUNCTIONS, AlphaVantageError, Client, QuotaExceeded
from sluice.universe import canonical

DEFAULT_ROOT = Path("data/local/raw/alphavantage")


class Quarter(BaseModel):
    """One fiscal quarter as the vendor reports it today, later restatements included."""

    end: date
    currency: str
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    operating_cashflow: float | None = None
    capex: float | None = None  # money spent, a positive number
    shares: float | None = None  # on today's split basis


class FetchResult(BaseModel):
    fetched: int = 0
    already_stored: int = 0
    failed: list[str] = []
    stopped: str | None = None  # why the run ended early


# Our field -> (statement, the vendor's field).
_FIELDS = {
    "revenue": ("INCOME_STATEMENT", "totalRevenue"),
    "gross_profit": ("INCOME_STATEMENT", "grossProfit"),
    "operating_income": ("INCOME_STATEMENT", "operatingIncome"),
    "net_income": ("INCOME_STATEMENT", "netIncome"),
    "operating_cashflow": ("CASH_FLOW", "operatingCashflow"),
    "capex": ("CASH_FLOW", "capitalExpenditures"),
    "shares": ("BALANCE_SHEET", "commonStockSharesOutstanding"),
}


def _number(text: object) -> float | None:
    try:
        return float(text)  # the vendor writes missing values as "None"
    except (TypeError, ValueError):
        return None


class FundamentalsStore:
    """The vendor's answers kept as files, one per statement and ticker, and read from there."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def _path(self, function: str, ticker: str) -> Path:
        return self.root / function / f"{canonical(ticker)}.json"

    def has(self, function: str, ticker: str) -> bool:
        return self._path(function, ticker).exists()

    def save(self, function: str, ticker: str, answer: dict) -> None:
        path = self._path(function, ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(answer))

    def quarters(self, ticker: str) -> list[Quarter] | None:
        """Every stored quarter, oldest first; None unless all three statements are stored."""
        if not all(self.has(function, ticker) for function in FUNCTIONS):
            return None
        rows: dict[str, dict[str, dict]] = {}
        for function in FUNCTIONS:
            answer = json.loads(self._path(function, ticker).read_text())
            for report in answer.get("quarterlyReports", []):
                rows.setdefault(report["fiscalDateEnding"], {})[function] = report
        quarters = []
        for end, reports in sorted(rows.items()):
            income = reports.get("INCOME_STATEMENT")
            if income is None:
                continue
            values = {
                name: _number(reports.get(function, {}).get(field))
                for name, (function, field) in _FIELDS.items()
            }
            currency = income.get("reportedCurrency") or "unknown"
            quarters.append(Quarter(end=date.fromisoformat(end), currency=currency, **values))
        return quarters


def fetch(
    tickers: list[str], client: Client, store: FundamentalsStore, refresh: bool = False
) -> FetchResult:
    """Fetch and store the statements that are not stored yet. Stops when the quota runs out."""
    result = FetchResult()
    for ticker in dict.fromkeys(canonical(t) for t in tickers):
        for function in FUNCTIONS:
            if store.has(function, ticker) and not refresh:
                result.already_stored += 1
                continue
            try:
                store.save(function, ticker, client.statement(function, ticker))
            except QuotaExceeded as exc:
                result.stopped = str(exc)
                return result
            except AlphaVantageError as exc:
                result.failed.append(f"{ticker} {function}: {exc}")
                continue
            result.fetched += 1
    return result
