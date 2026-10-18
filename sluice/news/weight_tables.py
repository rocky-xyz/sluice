"""Monthly source/event-type weight tables saved to disk for the pipeline (task 5.2).

The pipeline never fits weights while it runs: it reads the table fitted as of
the latest month start at or before its decision time
(docs/specs/07-agent-pipeline.md, "已确认（任务 5.1）"). Tables are frozen once
written. A later import can add news published before a past month start, and
refitting would then quietly change the table a finished run says it used, so
a saved month is only refitted when asked to (`refresh`).

One file per month: `<root>/<scoring config>/<as_of date>.json`.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from sluice.news.walkforward import Refit
from sluice.news.weights import WeightEstimate, WeightTable

DEFAULT_ROOT = Path("data/local/news_weights")


@dataclass(frozen=True)
class StoredTable:
    as_of: datetime
    n_rows: int
    table: WeightTable


class FitResult(BaseModel):
    written: list[datetime] = []
    kept: list[datetime] = []  # already saved; left as they were
    too_few: list[datetime] = []  # not enough resolved rows for a table


def path_for(root: Path, scoring_config: str, as_of: datetime) -> Path:
    return root / scoring_config / f"{as_of:%Y-%m-%d}.json"


def save(root: Path, scoring_config: str, as_of: datetime, n_rows: int, table: WeightTable) -> Path:
    path = path_for(root, scoring_config, as_of)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "as_of": as_of.isoformat(),
        "scoring_config": scoring_config,
        "n_rows": n_rows,
        "lambda0": table.lambda0,
        "source": {name: asdict(e) for name, e in table.source.items()},
        "event_type": {name: asdict(e) for name, e in table.event_type.items()},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return path


def _load(path: Path) -> StoredTable:
    data = json.loads(path.read_text())
    return StoredTable(
        as_of=datetime.fromisoformat(data["as_of"]).astimezone(UTC),
        n_rows=data["n_rows"],
        table=WeightTable(
            lambda0=data["lambda0"],
            source={k: WeightEstimate(**v) for k, v in data["source"].items()},
            event_type={k: WeightEstimate(**v) for k, v in data["event_type"].items()},
        ),
    )


def load_all(root: Path, scoring_config: str) -> list[StoredTable]:
    """Every saved table for `scoring_config`, oldest first."""
    tables = [_load(path) for path in (root / scoring_config).glob("*.json")]
    return sorted(tables, key=lambda t: t.as_of)


def latest_before(tables: list[StoredTable], moment: datetime) -> StoredTable | None:
    """The table fitted as of the latest month start not after `moment`."""
    usable = [t for t in tables if t.as_of <= moment]
    return max(usable, key=lambda t: t.as_of) if usable else None


def fit_missing(
    cutoffs: list[datetime],
    root: Path,
    scoring_config: str,
    fit: Callable[[list[datetime]], list[Refit]],
    refresh: bool = False,
) -> FitResult:
    """Fit and save the months not saved yet (every month with `refresh`).

    `fit` is `walk_forward_fit` with its stores bound; it is only asked for the
    months that will be written.
    """
    result = FitResult()
    todo = []
    for as_of in cutoffs:
        if not refresh and path_for(root, scoring_config, as_of).exists():
            result.kept.append(as_of)
        else:
            todo.append(as_of)
    for refit in fit(todo) if todo else []:
        if refit.table is None:
            result.too_few.append(refit.as_of)
        else:
            save(root, scoring_config, refit.as_of, refit.n_rows, refit.table)
            result.written.append(refit.as_of)
    return result
