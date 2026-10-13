import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from sluice.agents.config import PipelineConfig
from sluice.agents.graph import run
from sluice.agents.prefetch import backtest_decision_time
from sluice.agents.state import PipelineState, dump_state
from sluice.fundamentals.store import FundamentalsStore
from sluice.market_calendar import close_of
from sluice.prices.store import PriceStore
from sluice.universe import canonical

DEFAULT_OUT = Path("data/local/runs")
FIRST_DAY = date(2026, 1, 2)  # the first trading day of the study
EVERY = 10  # trading days between decisions: two weeks
RATINGS_FILE = "ratings.jsonl"

# Dollars per million tokens (input, output) at peak hours, from DeepSeek's price list on
# 2026-10-14; off-peak is half. Only for the estimate printed after a batch.
PRICES = {"deepseek-flash": (0.3, 1.2), "deepseek-v4-pro": (1.32, 3.96)}


class Cell(BaseModel):
    """One run of the grid."""

    ticker: str
    day: date
    repeat: int = 0


class CellResult(BaseModel):
    """One line of the ratings file: what a run concluded and what it used."""

    ticker: str
    day: date
    repeat: int
    run_id: str | None = None
    status: Literal["ok", "failed", "error"]  # error: the run broke off and left no state
    rating: str | None = None
    confidence: float | None = None
    failure: str | None = None
    gaps: int = 0
    calls: int = 0
    tokens: dict[str, tuple[int, int]] = {}  # model -> (input, output)


def decision_days(first: date, last: date, every: int = EVERY) -> list[date]:
    """Every `every`-th trading day from the first one on or after `first`, up to `last`."""
    days, day = [], first
    while day <= last:
        if close_of(day) is not None:
            days.append(day)
        day += timedelta(days=1)
    return days[::every]


def grid(tickers: Iterable[str], days: Iterable[date], repeats: int = 1) -> list[Cell]:
    """Every ticker on every day, `repeats` times, in time order."""
    symbols = list(dict.fromkeys(canonical(t) for t in tickers))
    return [
        Cell(ticker=ticker, day=day, repeat=repeat)
        for day in sorted(days)
        for ticker in symbols
        for repeat in range(repeats)
    ]


def unprepared(
    cells: Iterable[Cell], statements: FundamentalsStore, prices: PriceStore, benchmark: str
) -> list[str]:
    """What is missing for a batch to run without avoidable gaps.

    A run that lacks statements or prices still finishes, and a finished run is
    never run again; so the gap would stay in the results for good.
    """
    cells = list(cells)
    if not cells:
        return []
    tickers = list(dict.fromkeys(cell.ticker for cell in cells))
    last = max(cell.day for cell in cells)
    problems = [
        f"{ticker}: no statements stored (sluice fundamentals fetch --tickers {ticker})"
        for ticker in tickers
        if statements.quarters(ticker) is None
    ]
    # Also fetches each series once, here, rather than from several workers at a time.
    for ticker in dict.fromkeys([*tickers, canonical(benchmark)]):
        bars = prices.history(ticker, last, limit=1)
        if not bars or bars[-1].day < last:
            problems.append(f"{ticker}: no prices up to {last}")
    return problems


def run_dir(out_dir: Path, config: PipelineConfig) -> Path:
    """Results of one config live together and apart from every other config's."""
    return out_dir / f"{config.version}-{config.digest}"


def state_path(out_dir: Path, config: PipelineConfig, cell: Cell) -> Path:
    return run_dir(out_dir, config) / cell.ticker / f"{cell.day}_r{cell.repeat}.json"


def _result(cell: Cell, state: dict) -> CellResult:
    """A ratings line from a saved state."""
    meta, decision = state["meta"], state["decision"] or {}
    tokens: dict[str, tuple[int, int]] = {}
    for call in state["usage"]:
        used = tokens.get(call["model"], (0, 0))
        tokens[call["model"]] = (used[0] + call["input_tokens"], used[1] + call["output_tokens"])
    return CellResult(
        ticker=cell.ticker,
        day=cell.day,
        repeat=cell.repeat,
        run_id=meta["run_id"],
        status=meta["status"],
        rating=decision.get("rating"),
        confidence=decision.get("confidence"),
        failure=meta["failure"],
        gaps=len(state["inputs"]["gaps"]),
        calls=len(state["usage"]),
        tokens=tokens,
    )


def run_cell(
    graph: CompiledStateGraph, config: PipelineConfig, cell: Cell, out_dir: Path
) -> CellResult:
    """Run one cell and save its full state. A cell that breaks off is reported, not raised."""
    try:
        decision_time = backtest_decision_time(cell.day)
        state: PipelineState = run(graph, config, cell.ticker, decision_time, repeat=cell.repeat)
    except Exception as exc:  # noqa: BLE001 - one broken cell must not stop the grid
        failure = f"{type(exc).__name__}: {exc}".rstrip(": ")
        return CellResult(**cell.model_dump(), status="error", failure=failure)
    saved = dump_state(state)
    path = state_path(out_dir, config, cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    return _result(cell, saved)


def write_ratings(out_dir: Path, config: PipelineConfig) -> list[CellResult]:
    """Rebuild the ratings file from every saved state of this config, in time order."""
    folder = run_dir(out_dir, config)
    results = []
    for path in folder.glob("*/*.json"):
        day, repeat = path.stem.split("_r")
        cell = Cell(ticker=path.parent.name, day=date.fromisoformat(day), repeat=int(repeat))
        results.append(_result(cell, json.loads(path.read_text())))
    results.sort(key=lambda r: (r.day, r.ticker, r.repeat))
    if results:
        lines = [result.model_dump_json() for result in results]
        (folder / RATINGS_FILE).write_text("\n".join(lines) + "\n")
    return results


def run_batch(
    cells: list[Cell],
    config: PipelineConfig,
    make_graph: Callable[[], CompiledStateGraph],
    out_dir: Path = DEFAULT_OUT,
    workers: int = 4,
    on_done: Callable[[CellResult], None] | None = None,
) -> list[CellResult]:
    """Run every cell, `workers` at a time, and rebuild the ratings file.

    Runs are independent, so their order and the number of workers do not
    change any result. Each worker thread gets its own graph from `make_graph`.
    Safe to repeat: finished runs are returned from their checkpoints.
    """
    local = threading.local()

    def work(cell: Cell) -> CellResult:
        if not hasattr(local, "graph"):
            local.graph = make_graph()
        result = run_cell(local.graph, config, cell, out_dir)
        if on_done is not None:
            on_done(result)
        return result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(work, cells))
    write_ratings(out_dir, config)
    return results


def estimated_cost(results: Iterable[CellResult]) -> tuple[dict[str, tuple[int, int]], float]:
    """Tokens per model over these runs, and their cost in dollars at peak list prices."""
    totals: dict[str, tuple[int, int]] = {}
    for result in results:
        for model, (used_in, used_out) in result.tokens.items():
            have = totals.get(model, (0, 0))
            totals[model] = (have[0] + used_in, have[1] + used_out)
    cost = sum(
        (used_in * PRICES[model][0] + used_out * PRICES[model][1]) / 1_000_000
        for model, (used_in, used_out) in totals.items()
        if model in PRICES
    )
    return totals, cost
