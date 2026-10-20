from datetime import date

import pytest
import requests

from sluice.patterns.universe import HEADERS, SOURCE, fetch, load, parse, save

PAGE = """
<html><body>
<table class="wikitable" id="other"><tr><th>Symbol</th></tr><tr><td>ZZZZ</td></tr></table>
<table class="wikitable sortable" id="constituents">
<tr><th>Symbol</th><th>Security</th></tr>
<tr><td>MMM</td><td>3M</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
<tr><td>GOOGL</td><td>Alphabet (Class A)</td></tr>
<tr><td>GOOG</td><td>Alphabet (Class C)</td></tr>
<tr><td>AAPL</td><td>Apple</td></tr>
</table></body></html>
"""


def test_parse_reads_the_constituents_table_in_yahoo_spelling():
    assert parse(PAGE) == ["AAPL", "BRK-B", "GOOG", "GOOGL", "MMM"]


def test_parse_refuses_a_page_without_the_table_or_with_odd_symbols():
    with pytest.raises(ValueError):
        parse("<table id='other'><tr><th>Symbol</th></tr><tr><td>X</td></tr></table>")
    odd = PAGE.replace("<td>MMM</td>", "<td>3M Company</td>")
    with pytest.raises(ValueError, match="not ticker symbols"):
        parse(odd)


def test_fetch_sends_a_user_agent():
    sent = {}

    def get(url, headers, timeout):
        sent.update(url=url, headers=headers)
        response = requests.Response()
        response.status_code, response._content = 200, PAGE.encode()
        return response

    assert len(fetch(get)) == 5
    assert sent == {"url": SOURCE, "headers": HEADERS}


def test_a_saved_list_says_where_it_is_from_loads_back_and_is_never_overwritten(tmp_path):
    path = save(["AAPL", "MMM"], "sp500-test", date(2026, 10, 20), tmp_path)
    text = path.read_text()
    assert "on 2026-10-20: 2 symbols" in text and "survivorship bias" in text
    assert load("sp500-test", tmp_path) == ["AAPL", "MMM"]
    with pytest.raises(FileExistsError):
        save(["AAPL"], "sp500-test", date(2026, 10, 21), tmp_path)
