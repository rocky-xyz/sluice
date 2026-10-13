import json
from datetime import UTC, datetime

import pytest
import requests

from sluice.news.finnhub import archive, import_archive, parse_item
from sluice.news.finnhub_links import (
    RATE_LIMIT_WAITS,
    load_links,
    pending_links,
    resolve_links,
)
from sluice.news.publishers import publisher_for
from sluice.news.store import NewsStore

FETCHED = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
LATER = datetime(2027, 1, 1, tzinfo=UTC)
FOOL = "https://www.fool.com/investing/2026/01/09/prediction/"
YAHOO = "https://finance.yahoo.com/news/nvidia-ceo-213108932.html"


def item(n: int, source: str = "Yahoo", **overrides) -> dict:
    raw = {
        "datetime": 1768173623,  # 2026-01-11
        "headline": f"Headline {n}",
        "id": n,
        "source": source,
        "summary": "Summary.",
        "url": f"https://finnhub.io/api/news?id={n}",
    }
    raw.update(overrides)
    return raw


def run(raw_dir, resolve, **kwargs):
    kwargs.setdefault("sleep", lambda seconds: None)
    return resolve_links(raw_dir, resolve, now=lambda: FETCHED, **kwargs)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (FOOL, "Motley Fool"),
        ("https://247wallst.com/investing/2026/01/09/x/", "24/7 Wall St."),
        ("https://investorshub.advfn.com/a/1", "investorshub.advfn.com"),
        ("https://www.Axios.com/2026/01/09/x", "axios.com"),
        (YAHOO, None),
        ("https://sg.finance.yahoo.com/news/x.html", None),
        ("https://notyahoo.com/x", "notyahoo.com"),
        (None, None),
        ("", None),
    ],
)
def test_publisher_is_read_from_the_host(url, expected):
    assert publisher_for(url) == expected


def test_aggregated_item_takes_its_publisher_from_where_its_link_leads():
    assert parse_item(item(1), "NVDA", FETCHED).publisher is None
    assert parse_item(item(1), "NVDA", FETCHED, YAHOO).publisher is None
    resolved = parse_item(item(1), "NVDA", FETCHED, FOOL)
    assert resolved.publisher == "Motley Fool"
    assert resolved.url == "https://finnhub.io/api/news?id=1"  # the stored identity does not move
    assert parse_item(item(1, "Benzinga"), "NVDA", FETCHED, FOOL).publisher == "Benzinga"


def test_only_aggregated_items_are_pending_and_each_once(tmp_path):
    archive(tmp_path, "NVDA", FETCHED, [item(1), item(2, "Benzinga"), item(3, url="")])
    archive(tmp_path, "AMD", FETCHED, [item(1), item(4)])
    assert [raw["id"] for raw in pending_links(tmp_path)] == [1, 4]


def test_links_are_followed_recorded_and_not_followed_again(tmp_path):
    archive(tmp_path, "NVDA", FETCHED, [item(1), item(2), item(3)])
    asked = []

    def resolve(url: str):
        asked.append(url)
        return {"1": (302, FOOL), "2": (302, YAHOO), "3": (404, None)}[url[-1]]

    result = run(tmp_path, resolve)
    assert (result.pending, result.resolved, result.dead, result.failed) == (3, 2, 1, 0)
    assert load_links(tmp_path) == {1: FOOL, 2: YAHOO, 3: None}
    line = (tmp_path / "finnhub_links" / "2026-01-11.jsonl").read_text().splitlines()[0]
    assert json.loads(line) == {"id": 1, "status": 302, "location": FOOL, "at": FETCHED.isoformat()}

    asked.clear()
    assert run(tmp_path, resolve).pending == 0
    assert asked == []


def test_unanswered_links_are_tried_again_next_time(tmp_path):
    archive(tmp_path, "NVDA", FETCHED, [item(1), item(2)])

    def flaky(url: str):
        if url.endswith("1"):
            raise requests.ConnectionError("offline")
        return 302, FOOL

    result = run(tmp_path, flaky)
    assert (result.resolved, result.failed) == (1, 1)
    assert [raw["id"] for raw in pending_links(tmp_path)] == [1]


def test_rate_limit_pauses_then_continues(tmp_path):
    archive(tmp_path, "NVDA", FETCHED, [item(1)])
    answers = iter([(429, None), (429, None), (302, FOOL)])
    slept = []
    result = run(tmp_path, lambda url: next(answers), sleep=slept.append)
    assert result.resolved == 1
    assert slept == list(RATE_LIMIT_WAITS[:2])


def test_persistent_rate_limit_stops_the_run_and_keeps_what_was_done(tmp_path):
    archive(tmp_path, "NVDA", FETCHED, [item(n) for n in range(1, 6)])

    def resolve(url: str):
        return (302, FOOL) if int(url.rsplit("=", 1)[1]) <= 3 else (429, None)

    result = run(tmp_path, resolve)
    assert result.aborted
    assert result.resolved == 3
    assert [raw["id"] for raw in pending_links(tmp_path)] == [4, 5]


def test_limit_caps_one_run(tmp_path):
    archive(tmp_path, "NVDA", FETCHED, [item(n) for n in range(1, 6)])
    assert run(tmp_path, lambda url: (302, FOOL), limit=2).resolved == 2
    assert len(pending_links(tmp_path)) == 3


def test_import_fills_in_publishers_once_links_are_followed(tmp_path):
    raw_dir = tmp_path / "raw"
    archive(raw_dir, "NVDA", FETCHED, [item(1), item(2)])
    with NewsStore(tmp_path / "news.db") as store:
        import_archive(raw_dir, store)
        assert {r.publisher for r in store.query("NVDA", before=LATER)} == {None}

        run(raw_dir, lambda url: (302, FOOL) if url.endswith("1") else (302, YAHOO))
        again = import_archive(raw_dir, store)
        assert (again.inserted, again.skipped) == (0, 2)  # same articles, no duplicates
        rows = {r.title: r.publisher for r in store.query("NVDA", before=LATER)}
        assert rows == {"Headline 1": "Motley Fool", "Headline 2": None}
