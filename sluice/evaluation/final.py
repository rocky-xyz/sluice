"""The one final evaluation of a sealed test segment (docs/specs/05-evaluation.md, task 7.7).

Everything that could stop it is checked before the seal is opened, and the
whole report is worked out in memory first: only once it exists is the opening
recorded and the report shown. A crash on the way leaves the segment sealed and
nothing seen, so it can be run again; once recorded, it cannot.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from sluice.evaluation.decisions import Decision
from sluice.evaluation.report import Calendar
from sluice.evaluation.seal import DEFAULT_PATH, AlreadyOpened, Seal, open_seal, opened
from sluice.evaluation.split import Segment
from sluice.market_calendar import close_of
from sluice.sizing.config import SizingConfig

PIPELINE = "v1"  # the champion's pipeline: unweighted news (docs/specs/06, task 7.1)
EXTERNAL = "tradingagents-1394a3f"  # the original TradingAgents, run outside the repo (6.5)
REPEATS = {PIPELINE: 3, EXTERNAL: 1}


def sealed_config(pipeline: str, sizing: SizingConfig) -> str:
    """What the seal records, and what test_2 may later evaluate."""
    return f"{pipeline} + sizing {sizing.version}-{sizing.digest}"


def missing_runs(
    name: str, decisions: list[Decision], tickers: list[str], cal: Calendar, repeats: int
) -> list[str]:
    """Cells without a run file; a failed run has one and counts as hold."""
    grid = {(day, ticker) for day in cal.decisions for ticker in tickers}
    problems = []
    for repeat in range(repeats):
        have = {(d.day, d.ticker) for d in decisions if d.repeat == repeat}
        if missing := grid - have:
            problems.append(f"{name} repeat {repeat}: {len(missing)} of {len(grid)} runs missing")
    return problems


def too_early(cal: Calendar, now: datetime) -> str | None:
    """The last period ends at the close of `cal.end`; the labels need it too."""
    end = close_of(cal.end)
    if end is None or now <= end:
        return f"the segment's last period ends at the close of {cal.end}; run after that"
    return None


def evaluate_once(
    segment: Segment,
    compute: Callable[[], str],
    config: str,
    commit: str,
    path: Path = DEFAULT_PATH,
) -> tuple[Seal, str]:
    """The report from `compute`, with the opening recorded only once it has succeeded."""
    if any(seal.segment == segment for seal in opened(path)):
        raise AlreadyOpened(f"{segment} has already had its one evaluation")
    report = compute()  # an error here records nothing: the segment stays sealed
    return open_seal(segment, config, commit, path), report
