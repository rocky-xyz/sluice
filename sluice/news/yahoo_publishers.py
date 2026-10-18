"""Publisher metadata read from the Yahoo Finance page a Finnhub redirect lands on.

Finnhub's "source" for many Yahoo-hosted articles is just the aggregator, not
the original publisher (see `publishers.py`, which only resolves aggregator
redirects by domain -- a redirect that lands back on Yahoo's own renderer
tells it nothing). The page itself embeds the true publisher in a schema.org
NewsArticle block, separate from Yahoo's own `og:site_name`.
"""

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests
from pydantic import BaseModel

from sluice.news.store import NewsStore

LOG_DIR = "yahoo_publishers"
RATE_LIMIT_WAITS = (30.0, 60.0, 120.0)  # seconds; HTTP 429 only

_LD_JSON_RE = re.compile(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', re.DOTALL)
# Yahoo's own name for itself in the same block; a provider equal to this says nothing new.
_AGGREGATOR_NAMES = {"yahoo! finance", "yahoo finance"}

Fetch = Callable[[str], tuple[int, str]]


def fetch(url: str) -> tuple[int, str]:
    """Follow redirects to the final page and return its status and HTML."""
    response = requests.get(
        url, allow_redirects=True, timeout=20, headers={"User-Agent": "Mozilla/5.0"}
    )
    return response.status_code, response.text


def provider_from_html(html: str) -> str | None:
    """The page's NewsArticle JSON-LD `provider.name`, when it names a real publisher."""
    match = _LD_JSON_RE.search(html)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except ValueError:
        return None
    name = (data.get("provider") or {}).get("name")
    if not name or not name.strip() or name.strip().lower() in _AGGREGATOR_NAMES:
        return None
    return name.strip()


class ResolveResult(BaseModel):
    pending: int = 0
    found: int = 0  # a real publisher was read and filled in
    no_provider: int = 0  # the page answered 200, but named no usable publisher
    dead: int = 0  # a non-200 answer (e.g. the page no longer exists)
    failed: int = 0  # no answer at all; left for the next run
    aborted: str | None = None


def _attempted(raw_dir: Path) -> set[int]:
    done: set[int] = set()
    for path in sorted((raw_dir / LOG_DIR).glob("*.jsonl")):
        with path.open() as lines:
            done.update(json.loads(line)["article_id"] for line in lines)
    return done


def _fetch_one(job: tuple[int, str], fetch: Fetch, sleep: Callable[[float], None]) -> tuple:
    """One candidate's outcome: (article_id, category, provider, http status or None)."""
    article_id, url = job
    status: int | None = 429
    html = ""
    try:
        for wait in (*RATE_LIMIT_WAITS, None):
            status, html = fetch(url)
            if status != 429 or wait is None:
                break
            sleep(wait)
    except requests.RequestException:
        return article_id, "failed", None, None
    if status == 429:
        return article_id, "rate_limited", None, status
    if status != 200:
        return article_id, "dead", None, status
    provider = provider_from_html(html)
    return article_id, ("found" if provider else "no_provider"), provider, status


def resolve_publishers(
    store: NewsStore,
    candidates: dict[int, str],
    raw_dir: Path,
    fetch: Fetch = fetch,
    limit: int | None = None,
    workers: int = 1,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    on_step: Callable[[int, int], None] = lambda done, total: None,
) -> ResolveResult:
    """Fetch each candidate's page once and fill in its publisher if the page names one.

    `candidates` is article id -> url (e.g. from `NewsStore.missing_publisher_urls`).
    Fetches run `workers` at a time; every write to `store` happens back on this
    thread once a fetch returns, so `store` itself is never touched concurrently.
    An id already attempted, found or not, is skipped on a later run, so
    interrupting and rerunning only works through what is left. A candidate
    still rate-limited after retrying its own backoff is left unlogged (and so
    retried later) rather than stopping the whole batch.
    """
    done = _attempted(raw_dir)
    todo = [(article_id, url) for article_id, url in candidates.items() if article_id not in done]
    todo = todo[:limit]
    result = ResolveResult(pending=len(todo))
    log_path = raw_dir / LOG_DIR / f"{now():%Y-%m-%d}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, log_path.open("a") as log:
        outcomes = pool.map(lambda job: _fetch_one(job, fetch, sleep), todo)
        for step, (article_id, category, provider, status) in enumerate(outcomes, start=1):
            on_step(step, len(todo))
            if category == "failed":
                result.failed += 1
                continue
            if category == "rate_limited":
                result.aborted = "still rate limited after waiting"
                continue
            if category == "found":
                store.fill_publisher(article_id, provider)
                result.found += 1
            elif category == "no_provider":
                result.no_provider += 1
            else:  # dead
                result.dead += 1
            log.write(
                json.dumps({"article_id": article_id, "status": status, "provider": provider})
                + "\n"
            )
            log.flush()
    return result
