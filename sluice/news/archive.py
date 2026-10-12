"""Raw JSONL archive shared by all collectors: one vendor item per line, exactly as received."""

import json
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from sluice.news.models import NewsItem
from sluice.news.store import NewsStore

Parser = Callable[[dict, str, datetime], NewsItem | None]


class TickerResult(BaseModel):
    ticker: str
    fetched: int = 0
    inserted: int = 0
    skipped: int = 0  # already stored
    dropped: int = 0  # no title, or not linked to any collected ticker
    error: str | None = None


class ImportResult(BaseModel):
    records: int = 0
    inserted: int = 0
    skipped: int = 0
    dropped: int = 0


def raw_id(raw: dict) -> str | int | None:
    return raw.get("id") or raw.get("uuid")


def append_records(path: Path, ticker: str, fetched_at: datetime, items: Iterable[dict]) -> int:
    """Append raw items to one archive file, once per (ticker, item id). Returns lines written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        seen = {(r["ticker"], raw_id(r["raw"])) for r in read_records(path)}
    written = 0
    with path.open("a") as out:
        for raw in items:
            key = (ticker, raw_id(raw))
            if key[1] is not None and key in seen:
                continue
            seen.add(key)
            record = {"fetched_at": fetched_at.isoformat(), "ticker": ticker, "raw": raw}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def read_records(path: Path) -> Iterator[dict]:
    with path.open() as lines:
        for line in lines:
            yield json.loads(line)


def archive_files(raw_dir: Path, source: str) -> list[Path]:
    return sorted((raw_dir / source).glob("*.jsonl"))


def import_files(files: Iterable[Path], parse: Parser, store: NewsStore) -> ImportResult:
    """Parse archived records and add them to the store. Safe to repeat."""
    result = ImportResult()
    for path in files:
        records = list(read_records(path))
        items = [
            parse(r["raw"], r["ticker"], datetime.fromisoformat(r["fetched_at"])) for r in records
        ]
        kept = [item for item in items if item is not None]
        added = store.add(kept)
        result.records += len(records)
        result.dropped += len(items) - len(kept)
        result.inserted += added.inserted
        result.skipped += added.skipped
    return result


def write_run_log(
    run_dir: Path, started_at: datetime, source: str, results: list[TickerResult]
) -> Path:
    """Append one line per run, so gaps can be found long after the CI logs are gone."""
    path = run_dir / f"{started_at:%Y-%m-%d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "started_at": started_at.isoformat(),
        "source": source,
        "results": [r.model_dump() for r in results],
    }
    with path.open("a") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
