"""Where Finnhub's article links lead. Finnhub gives a redirect, not the article's address."""

import json
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import requests
from pydantic import BaseModel

from sluice.news.archive import archive_files, read_records
from sluice.news.finnhub import AGGREGATORS, SOURCE, _stamped_at

LINKS = "finnhub_links"
SAVE_EVERY = 50  # links between writes; an interrupted run loses at most this many
RATE_LIMIT_WAITS = (30.0, 60.0, 120.0)  # seconds; HTTP 429 only

Resolve = Callable[[str], tuple[int, str | None]]


class ResolveResult(BaseModel):
    pending: int = 0  # links not yet followed when the run started
    resolved: int = 0  # redirects that named a target
    dead: int = 0  # answered, but with no target (the link no longer works)
    failed: int = 0  # no answer; left for the next run
    aborted: str | None = None


def follow(url: str) -> tuple[int, str | None]:
    """Ask where a link leads without going there: status code and redirect target."""
    response = requests.get(
        url, allow_redirects=False, timeout=20, headers={"User-Agent": "Mozilla/5.0"}
    )
    return response.status_code, response.headers.get("Location") or None


def load_links(raw_dir: Path) -> dict[int, str | None]:
    """Article id -> redirect target (None for a dead link), for every link followed so far."""
    return {
        record["id"]: record["location"]
        for path in archive_files(raw_dir, LINKS)
        for record in read_records(path)
    }


def pending_links(raw_dir: Path) -> list[dict]:
    """Archived articles from an aggregator whose link has not been followed, oldest first."""
    done = load_links(raw_dir)
    pending = {}
    for path in archive_files(raw_dir, SOURCE):
        for record in read_records(path):
            raw = record["raw"]
            if raw.get("source") in AGGREGATORS and raw.get("url") and raw["id"] not in done:
                pending.setdefault(raw["id"], raw)
    return list(pending.values())


def _save(raw_dir: Path, answered: list[tuple[dict, int, str | None]], at: datetime) -> None:
    by_file: dict[str, list[dict]] = defaultdict(list)
    for raw, status, location in answered:
        stamped = _stamped_at(raw)
        name = f"{stamped:%Y-%m-%d}" if stamped else "undated"
        by_file[name].append(
            {"id": raw["id"], "status": status, "location": location, "at": at.isoformat()}
        )
    for name, records in by_file.items():
        path = raw_dir / LINKS / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as out:
            out.writelines(json.dumps(record) + "\n" for record in records)


def resolve_links(
    raw_dir: Path,
    resolve: Resolve = follow,
    limit: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    on_step: Callable[[int, int], None] = lambda done, total: None,
) -> ResolveResult:
    """Follow every pending link once and record where it leads. Rerun to continue.

    Links are followed one at a time: Finnhub answers simultaneous requests
    with HTTP 429 even at a low overall rate, while one connection sustained
    about four a second without a refusal (measured 2026-10-13). Links that get
    no answer are not recorded, so the next run tries them again. A rate limit
    pauses the run; one that persists ends it.
    """
    todo = pending_links(raw_dir)[:limit]
    result = ResolveResult(pending=len(todo))
    unsaved: list[tuple[dict, int, str | None]] = []

    def save() -> None:
        _save(raw_dir, unsaved, now())
        unsaved.clear()

    for step, raw in enumerate(todo, start=1):
        on_step(step, len(todo))
        status: int | None = 429
        location = None
        try:
            for wait in (*RATE_LIMIT_WAITS, None):
                status, location = resolve(raw["url"])
                if status != 429 or wait is None:
                    break
                sleep(wait)
        except requests.RequestException:
            result.failed += 1
            continue
        if status == 429:
            result.aborted = "still rate limited after waiting"
            break
        unsaved.append((raw, status, location))
        result.resolved += location is not None
        result.dead += location is None
        if len(unsaved) >= SAVE_EVERY:
            save()
    save()
    return result
