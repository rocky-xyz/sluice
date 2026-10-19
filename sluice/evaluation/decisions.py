"""Pipeline decisions read back for evaluation, one segment at a time.

Only single run states are read, picked by the decision day in their file
name; a state of another segment is never opened. The ratings file is not
used: it mixes every segment in one file.
"""

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from sluice.agents.state import Rating
from sluice.evaluation.split import Segment, require_open, segment


class Decision(BaseModel):
    ticker: str
    day: date
    repeat: int
    config: str  # "<version>-<digest>"
    run_id: str | None  # from the state's meta; None for an external file without one
    rating: Rating | None  # None when the run failed
    confidence: float | None


def load_decisions(folder: Path, requested: Segment, final: bool = False) -> list[Decision]:
    """Every run of one config (a run directory) decided in `requested`, in time order."""
    require_open(requested, final)
    decisions = []
    for path in folder.glob("*/*.json"):
        day_text, repeat = path.stem.split("_r")
        day = date.fromisoformat(day_text)
        if segment(day) != requested:
            continue
        state = json.loads(path.read_text())
        verdict = state["decision"] or {}
        decisions.append(
            Decision(
                ticker=path.parent.name,
                day=day,
                repeat=int(repeat),
                config=folder.name,
                run_id=(state.get("meta") or {}).get("run_id"),
                rating=verdict.get("rating"),
                confidence=verdict.get("confidence"),
            )
        )
    return sorted(decisions, key=lambda d: (d.day, d.ticker, d.repeat))
