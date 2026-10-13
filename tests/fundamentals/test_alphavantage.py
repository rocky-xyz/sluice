import json

import pytest
import requests

from sluice.fundamentals.alphavantage import (
    FUNCTIONS,
    MIN_INTERVAL,
    AlphaVantageError,
    Client,
    QuotaExceeded,
)
from sluice.fundamentals.store import FundamentalsStore, fetch
from tests.fundamentals.statements import ENDS, answers, stock


class FakeHTTP:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append(kwargs["params"])
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        status, body = answer if isinstance(answer, tuple) else (200, answer)
        response = requests.Response()
        response.status_code = status
        response._content = (body if isinstance(body, str) else json.dumps(body)).encode()
        return response


def client(*answers, clock=None, waits=None):
    http = FakeHTTP(*answers)
    ticks = iter(clock or [float(n * 100) for n in range(1000)])
    sleep = (waits if waits is not None else []).append
    made = Client("secret", get=http, sleep=sleep, clock=lambda: next(ticks))
    return made, http


GOOD = answers()["INCOME_STATEMENT"]
LIMIT = {"Information": "Thank you for using Alpha Vantage! ... 25 requests per day ..."}


def test_a_statement_is_requested_and_returned_as_given():
    made, http = client(GOOD)
    assert made.statement("INCOME_STATEMENT", "NVDA") == GOOD
    assert http.calls == [{"function": "INCOME_STATEMENT", "symbol": "NVDA", "apikey": "secret"}]


@pytest.mark.parametrize("body", [LIMIT, {"Note": "slow down"}])
def test_a_limit_message_is_told_apart_from_other_failures(body):
    made, _ = client(body)
    with pytest.raises(QuotaExceeded):
        made.statement("INCOME_STATEMENT", "NVDA")


@pytest.mark.parametrize(
    "answer",
    [{}, {"Error Message": "Invalid API call"}, (500, "oops"), (200, "not json"), (200, "[]"),
     requests.ConnectionError("https://x/?apikey=secret")],
)  # fmt: skip
def test_other_failures_are_errors_that_never_show_the_key(answer):
    made, _ = client(answer)
    with pytest.raises(AlphaVantageError) as raised:
        made.statement("INCOME_STATEMENT", "NVDA")
    assert not isinstance(raised.value, QuotaExceeded)
    assert "secret" not in str(raised.value)


def test_requests_are_spaced_out():
    waits = []
    made, _ = client(GOOD, clock=[0.0, 0.5, 0.5, 10.0, 10.0], waits=waits)
    for _ in range(3):
        made.statement("INCOME_STATEMENT", "NVDA")
    assert waits == [MIN_INTERVAL - 0.5]


# --- store ---


def test_the_three_statements_are_merged_by_quarter(tmp_path):
    store = FundamentalsStore(tmp_path)
    assert store.quarters("NVDA") is None
    stock(store, "nvda", currency="TWD")

    quarters = store.quarters("NVDA")
    assert [q.end for q in quarters] == sorted(ENDS)
    oldest = quarters[0]
    assert (oldest.revenue, oldest.gross_profit, oldest.operating_income) == (1e9, 6e8, 3e8)
    assert (oldest.net_income, oldest.operating_cashflow, oldest.capex) == (2e8, 2.5e8, 5e7)
    assert (oldest.shares, oldest.currency) == (1e9, "TWD")


def test_incomplete_statements_and_missing_values(tmp_path):
    store = FundamentalsStore(tmp_path)
    given = answers()
    given["CASH_FLOW"]["quarterlyReports"][0]["capitalExpenditures"] = "None"
    del given["BALANCE_SHEET"]["quarterlyReports"][0]
    store.save("INCOME_STATEMENT", "NVDA", given["INCOME_STATEMENT"])
    store.save("CASH_FLOW", "NVDA", given["CASH_FLOW"])
    assert store.quarters("NVDA") is None  # one statement is still missing

    store.save("BALANCE_SHEET", "NVDA", given["BALANCE_SHEET"])
    latest = store.quarters("NVDA")[-1]
    assert (latest.capex, latest.shares) == (None, None)
    assert latest.revenue is not None


# --- fetch ---


def test_fetch_stores_three_statements_a_ticker_and_skips_what_it_has(tmp_path):
    store = FundamentalsStore(tmp_path)
    made, http = client(GOOD)
    first = fetch(["NVDA", "goog", "GOOGL"], made, store)
    assert (first.fetched, first.already_stored, first.stopped) == (6, 0, None)
    assert [(c["symbol"], c["function"]) for c in http.calls] == [
        (ticker, function) for ticker in ("NVDA", "GOOGL") for function in FUNCTIONS
    ]

    again = fetch(["NVDA", "GOOGL"], made, store)
    assert (again.fetched, again.already_stored) == (0, 6)
    assert len(http.calls) == 6
    assert fetch(["NVDA"], made, store, refresh=True).fetched == 3


def test_fetch_stops_at_the_quota_and_continues_next_time(tmp_path):
    store = FundamentalsStore(tmp_path)
    made, http = client(GOOD, GOOD, LIMIT)
    result = fetch(["NVDA", "AMD"], made, store)
    assert (result.fetched, result.stopped) == (2, "rate or daily limit reached")
    assert len(http.calls) == 3
    assert store.quarters("NVDA") is None

    made, http = client(GOOD)
    result = fetch(["NVDA", "AMD"], made, store)
    assert (result.fetched, result.already_stored) == (4, 2)
    assert store.quarters("NVDA") is not None


def test_one_failing_statement_does_not_stop_the_rest(tmp_path):
    store = FundamentalsStore(tmp_path)
    made, _ = client({}, GOOD)
    result = fetch(["SPY", "NVDA"], made, store)
    assert result.failed == ["SPY INCOME_STATEMENT: no INCOME_STATEMENT for SPY"]
    assert result.fetched == 5
