import json
from datetime import UTC, datetime

import requests

from sluice.news.store import NewsStore
from sluice.news.yahoo_publishers import (
    RATE_LIMIT_WAITS,
    ResolveResult,
    provider_from_html,
    resolve_publishers,
)

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def page(provider: str | None = "GuruFocus.com") -> str:
    ld_json = {
        "@type": "NewsArticle",
        "publisher": {"@type": "Organization", "name": "Yahoo! Finance"},
    }
    if provider is not None:
        ld_json["provider"] = {"@type": "Organization", "name": provider}
    return (
        '<meta property="og:site_name" content="Yahoo Finance">'
        f'<script type="application/ld+json">{json.dumps(ld_json)}</script>'
    )


def run(store, candidates, fetch, raw_dir, **kwargs):
    kwargs.setdefault("sleep", lambda seconds: None)
    return resolve_publishers(store, candidates, raw_dir, fetch, now=lambda: NOW, **kwargs)


# --- provider_from_html ---


def test_extracts_the_true_publisher():
    assert provider_from_html(page("Investor's Business Daily")) == "Investor's Business Daily"


def test_a_provider_equal_to_the_aggregator_is_not_a_find():
    assert provider_from_html(page("Yahoo! Finance")) is None
    assert provider_from_html(page("yahoo finance")) is None


def test_no_ld_json_block_is_not_a_find():
    assert provider_from_html("<html><body>nothing here</body></html>") is None


def test_malformed_json_is_not_a_find():
    html = '<script type="application/ld+json">{not valid json</script>'
    assert provider_from_html(html) is None


def test_a_missing_provider_field_is_not_a_find():
    assert provider_from_html(page(provider=None)) is None


# --- resolve_publishers ---


def test_a_found_provider_fills_in_the_publisher(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(1)])
        article_id = next(iter(store.missing_publisher_urls("finnhub")))
        candidates = {article_id: "https://finnhub.io/api/news?id=1"}

        result = run(store, candidates, lambda url: (200, page("GuruFocus.com")), tmp_path)
        assert result == ResolveResult(pending=1, found=1)
        assert store.missing_publisher_urls("finnhub") == {}


def test_a_page_with_no_usable_provider_is_not_filled_in(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(1)])
        article_id = next(iter(store.missing_publisher_urls("finnhub")))
        result = run(store, {article_id: "u"}, lambda url: (200, page(None)), tmp_path)
        assert result == ResolveResult(pending=1, no_provider=1)
        assert article_id in store.missing_publisher_urls("finnhub")


def test_a_dead_link_is_counted_separately_from_no_provider(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(1)])
        article_id = next(iter(store.missing_publisher_urls("finnhub")))
        result = run(store, {article_id: "u"}, lambda url: (404, ""), tmp_path)
        assert result == ResolveResult(pending=1, dead=1)


def test_a_network_failure_is_left_for_the_next_run(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(1), _item(2)])
        ids = sorted(store.missing_publisher_urls("finnhub"))

        def flaky(url):
            if url == "fails":
                raise requests.ConnectionError("offline")
            return 200, page("Benzinga")

        candidates = {ids[0]: "fails", ids[1]: "ok"}
        result = run(store, candidates, flaky, tmp_path)
        assert (result.failed, result.found) == (1, 1)

        # a later run only tries what failed before
        asked = []
        run(store, candidates, lambda url: (asked.append(url), (200, page("Benzinga")))[1], tmp_path)
        assert asked == ["fails"]


def test_an_already_attempted_id_is_not_fetched_again(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(1)])
        article_id = next(iter(store.missing_publisher_urls("finnhub")))
        candidates = {article_id: "u"}
        run(store, candidates, lambda url: (200, page("Benzinga")), tmp_path)

        asked = []
        run(store, candidates, lambda url: (asked.append(url), (200, page("Benzinga")))[1], tmp_path)
        assert asked == []


def test_rate_limit_pauses_then_continues(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(1)])
        article_id = next(iter(store.missing_publisher_urls("finnhub")))
        answers = iter([(429, ""), (429, ""), (200, page("Benzinga"))])
        slept = []
        result = resolve_publishers(
            store, {article_id: "u"}, tmp_path, lambda url: next(answers),
            now=lambda: NOW, sleep=slept.append,
        )
        assert result.found == 1
        assert slept == list(RATE_LIMIT_WAITS[:2])


def test_persistent_rate_limit_stops_the_run_and_keeps_what_was_done(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(n) for n in range(1, 6)])
        candidates = store.missing_publisher_urls("finnhub")

        def fetch(url):
            return (200, page("Benzinga")) if url.endswith(("1", "2", "3")) else (429, "")

        result = run(store, candidates, fetch, tmp_path)
        assert result.aborted
        assert result.found == 3


def test_limit_caps_one_run(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(n) for n in range(1, 6)])
        candidates = store.missing_publisher_urls("finnhub")
        result = run(store, candidates, lambda url: (200, page("Benzinga")), tmp_path, limit=2)
        assert result.found == 2
        assert len(store.missing_publisher_urls("finnhub")) == 3


def test_concurrent_workers_still_write_every_result_once(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(n) for n in range(1, 21)])
        candidates = store.missing_publisher_urls("finnhub")

        def fetch(url):
            n = int(url.rsplit("=", 1)[1])
            if n % 5 == 0:
                return 404, ""
            return 200, page(f"Publisher {n}")

        result = run(store, candidates, fetch, tmp_path, workers=5)
        assert (result.found, result.dead) == (16, 4)
        assert len(store.missing_publisher_urls("finnhub")) == 4  # only the dead ones remain


def test_concurrency_does_not_corrupt_the_attempt_log(tmp_path):
    with NewsStore(tmp_path / "news.db") as store:
        store.add([_item(n) for n in range(1, 21)])
        candidates = store.missing_publisher_urls("finnhub")
        run(store, candidates, lambda url: (200, page("Benzinga")), tmp_path, workers=5)

        asked = []
        run(
            store, candidates, lambda url: (asked.append(url), (200, page("Benzinga")))[1],
            tmp_path, workers=5,
        )
    assert asked == []  # every id from the first run was logged exactly once


def _item(n: int):
    from sluice.news.models import NewsItem

    return NewsItem(
        title=f"Headline {n}",
        source="finnhub",
        tickers=["NVDA"],
        fetched_at=NOW,
        url=f"https://finnhub.io/api/news?id={n}",
        published_at=NOW,
    )
