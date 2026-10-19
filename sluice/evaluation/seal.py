"""The record of opened test segments: each may be opened once (docs/specs/05-evaluation.md).

This guards against accidents, such as looking at test results and then going
back to change the design; it does not stop anyone determined to get around it.
Test segment two may only be opened after segment one, and only for the config
recorded when segment one was opened, so that it checks that config rather
than one changed after seeing segment one.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import AwareDatetime, BaseModel

from sluice.evaluation.split import SEALED, SealedError, Segment

DEFAULT_PATH = Path("data/local/evaluation/seals.json")


class AlreadyOpened(RuntimeError):
    """A test segment that has had its one evaluation was asked to be opened again."""


class Seal(BaseModel):
    segment: Segment
    opened_at: AwareDatetime
    config: str  # "<version>-<digest>", as in the run directory's name
    commit: str  # the code that evaluated it


def opened(path: Path = DEFAULT_PATH) -> list[Seal]:
    if not path.exists():
        return []
    return [Seal.model_validate(entry) for entry in json.loads(path.read_text())]


def open_seal(
    segment: Segment,
    config: str,
    commit: str,
    path: Path = DEFAULT_PATH,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Seal:
    """Record that `segment` is being evaluated now. Refuses a second time."""
    if segment not in SEALED:
        raise ValueError(f"{segment} is not sealed; there is nothing to open")
    seals = opened(path)
    if any(seal.segment == segment for seal in seals):
        raise AlreadyOpened(f"{segment} has already been evaluated once")
    if segment == "test_2":
        first = next((seal for seal in seals if seal.segment == "test_1"), None)
        if first is None:
            raise SealedError("test_2 can only be opened after test_1")
        if first.config != config:
            raise SealedError(f"test_2 may only evaluate {first.config}, frozen at test_1")
    seal = Seal(segment=segment, opened_at=now(), config=config, commit=commit)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [s.model_dump(mode="json") for s in (*seals, seal)]
    path.write_text(json.dumps(entries, indent=2) + "\n")
    return seal
