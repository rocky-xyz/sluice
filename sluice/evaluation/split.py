"""Which segment a decision day belongs to (docs/specs/05-evaluation.md).

Frozen: the evolution layer must not change these dates. Segments are decided
by the decision day, not by the dates of the prices behind its label, so the
last validation decision is still evaluated with prices from early August.
There is no training segment: weights are fitted walk-forward, each table on
news before its month start, and the pipeline has nothing else to train.
"""

from datetime import date
from typing import Literal

Segment = Literal["validation", "test_1", "test_2"]

VALIDATION = (date(2026, 1, 2), date(2026, 7, 27))
TEST_1 = (date(2026, 8, 10), date(2026, 10, 6))  # historical, sealed
TEST_2_FROM = date(2026, 10, 20)  # every decision from here on, as it happens; sealed
SEALED: tuple[Segment, ...] = ("test_1", "test_2")


class SealedError(PermissionError):
    """A sealed test segment was asked for outside the one final evaluation."""


def segment(day: date) -> Segment | None:
    """The segment of a decision day; None for a day that belongs to none."""
    if VALIDATION[0] <= day <= VALIDATION[1]:
        return "validation"
    if TEST_1[0] <= day <= TEST_1[1]:
        return "test_1"
    if day >= TEST_2_FROM:
        return "test_2"
    return None


def require_open(requested: Segment, final: bool = False) -> None:
    """Raise unless `requested` may be read: validation always, a test segment only when final."""
    if requested in SEALED and not final:
        raise SealedError(
            f"{requested} is sealed: only the final evaluation (sluice evaluate --final) reads it"
        )
