from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from sluice import __version__
from sluice.universe import COMPANIES, UNIVERSE

app = typer.Typer(help="Multi-agent equity research pipeline.", no_args_is_help=True)
news_app = typer.Typer(help="News collection and storage.", no_args_is_help=True)
app.add_typer(news_app, name="news")
fundamentals_app = typer.Typer(help="Financial statements.", no_args_is_help=True)
app.add_typer(fundamentals_app, name="fundamentals")
calibration_app = typer.Typer(help="Hand-labeling Jev's scores against a sample.", no_args_is_help=True)
app.add_typer(calibration_app, name="calibration")
evolve_app = typer.Typer(help="The evolution loop over sizing configs.", no_args_is_help=True)
app.add_typer(evolve_app, name="evolve")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"sluice {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    load_dotenv()  # API keys live in a local .env; real environment variables win


@contextmanager
def _weighting(config, cache) -> Iterator:
    """Scores, Jev and weight tables for a config that weights news; None when it does not."""
    import os

    if not config.news.weighted:
        yield None
        return
    from sluice.agents.prefetch import Weighting
    from sluice.news.weight_tables import DEFAULT_ROOT, load_all
    from sluice.scoring.client import ENV_KEY, CachedJev, JevClient
    from sluice.scoring.config import load_scoring_config
    from sluice.scoring.store import ScoreStore

    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        reason = f"config {config.version} weights news"
        typer.echo(f"{ENV_KEY} is not set (put it in .env): {reason}.", err=True)
        raise typer.Exit(code=2)
    scoring = load_scoring_config(config.news.scoring_config)
    with ScoreStore() as scores:
        yield Weighting(
            scores=scores,
            tables=load_all(DEFAULT_ROOT, scoring.version),
            scoring=scoring,
            jev=CachedJev(cache, JevClient(key)),
        )


@app.command("analyze")
def analyze(
    ticker: Annotated[str, typer.Argument(help="Ticker to analyse.")],
    day: Annotated[
        str | None,
        typer.Option(
            "--date",
            help="Decision day (YYYY-MM-DD): a backtest run deciding at its close. "
            "Without it the run is ad hoc: it decides now and first collects the ticker's news.",
        ),
    ] = None,
    config_version: Annotated[
        str, typer.Option("--config", help="Pipeline config version under configs/pipeline/.")
    ] = "v1",
    repeat: Annotated[int, typer.Option("--repeat", help="Which repeat of this run.")] = 0,
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the full run state to this JSON file.")
    ] = None,
) -> None:
    """Run the research pipeline for one ticker, as of a past close or as of now."""
    import json
    import os
    from datetime import UTC, date, datetime

    from sluice.agents.cache import CallCache
    from sluice.agents.config import load_config
    from sluice.agents.graph import Deps, build_graph, open_checkpointer, run
    from sluice.agents.llm import ENV_KEY, CachedLLM, DeepSeekClient
    from sluice.agents.prefetch import backtest_decision_time
    from sluice.agents.state import dump_state
    from sluice.fundamentals import alphavantage
    from sluice.fundamentals import store as statements_store
    from sluice.fundamentals.store import FundamentalsStore
    from sluice.news import yahoo
    from sluice.news.store import NewsStore
    from sluice.prices.store import PriceStore
    from sluice.prices.yahoo import fetch_bars

    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        typer.echo(f"{ENV_KEY} is not set (put it in .env).", err=True)
        raise typer.Exit(code=2)
    if day is None:
        mode, decision_time = "adhoc", datetime.now(UTC)
    else:
        mode = "backtest"
        try:
            decision_time = backtest_decision_time(date.fromisoformat(day))
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--date") from exc
    config = load_config(config_version)
    statements = FundamentalsStore()

    checkpointer = open_checkpointer()
    try:
        with (
            NewsStore(db) as store,
            PriceStore(fetch=fetch_bars) as prices,
            CallCache() as cache,
            _weighting(config, cache) as weighting,
        ):

            def collect_news(symbol: str) -> str | None:
                return yahoo.collect([symbol], store, raw_dir=Path("data/local/raw"))[0].error

            def fetch_statements(symbol: str) -> str | None:
                vendor_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
                if not vendor_key:
                    return "ALPHA_VANTAGE_API_KEY is not set"
                result = statements_store.fetch(
                    [symbol], alphavantage.Client(vendor_key), statements
                )
                return result.stopped or next(iter(result.failed), None)

            llm = CachedLLM(cache, DeepSeekClient(key))
            deps = Deps(
                store, prices, statements, llm, config, collect_news, fetch_statements, weighting
            )
            graph = build_graph(deps, checkpointer)
            state = run(graph, config, ticker, decision_time, repeat=repeat, mode=mode)
    finally:
        checkpointer.conn.close()

    saved = dump_state(state)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    meta, decision = state["meta"], state.get("decision")
    console = Console()
    inputs = state["inputs"]
    console.print(
        f"{meta.run_id}  price indicators: {len(inputs.prices)} (as of {inputs.prices_as_of}), "
        f"news items: {len(inputs.news)}, fundamentals: {len(inputs.fundamentals)}"
    )
    for signal in inputs.signals:
        console.print(
            f"  {signal.ref}: {signal.value:+.2e} over {signal.n_items} scored items "
            f"(weights as of {signal.weights_as_of:%Y-%m-%d})"
        )
    for gap in inputs.gaps:
        console.print(f"  gap in {gap.kind}: {gap.reason}", markup=False)
    for call in state.get("usage", []):
        source = "cache" if call.cached else "call"
        console.print(
            f"  {call.role} ({call.model}, {source}): "
            f"{call.input_tokens} in, {call.output_tokens} out"
        )
    if decision is None:
        typer.echo(f"Failed: {meta.failure}", err=True)
        raise typer.Exit(code=1)
    console.print(f"[bold]{decision.rating}[/bold]  confidence {decision.confidence:.2f}")
    for evidence in decision.evidence:
        console.print(f"  {evidence.direction}: {evidence.claim} {evidence.refs}", markup=False)
    console.print(decision.rationale, markup=False)
    for note in decision.notes:
        console.print(f"note: {note}", markup=False)


@app.command("batch")
def batch(
    tickers: Annotated[
        str, typer.Option("--tickers", help="Comma-separated tickers to run.")
    ] = ",".join(COMPANIES),
    start: Annotated[
        str | None, typer.Option("--from", help="First decision day or later (YYYY-MM-DD).")
    ] = None,
    end: Annotated[
        str | None,
        typer.Option(
            "--to", help="Last decision day or earlier. Default: the last closed session."
        ),
    ] = None,
    every: Annotated[
        int, typer.Option("--every", help="Trading days between decision days.")
    ] = 10,
    repeats: Annotated[int, typer.Option("--repeats", help="Runs per ticker and day.")] = 1,
    workers: Annotated[int, typer.Option("--workers", help="Runs going on at a time.")] = 4,
    config_version: Annotated[
        str, typer.Option("--config", help="Pipeline config version under configs/pipeline/.")
    ] = "v1",
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where run states and the ratings file go.")
    ] = Path("data/local/runs"),
    allow_gaps: Annotated[
        bool,
        typer.Option(
            "--allow-gaps", help="Run even when statements, prices or weight tables are missing."
        ),
    ] = False,
) -> None:
    """Backtest a grid of tickers and decision days. Interrupt and rerun freely: it resumes."""
    import os
    from datetime import UTC, date, datetime

    from rich.progress import Progress

    from sluice.agents import batch as batching
    from sluice.agents.cache import CallCache
    from sluice.agents.config import load_config
    from sluice.agents.graph import Deps, build_graph, open_checkpointer
    from sluice.agents.llm import ENV_KEY, CachedLLM, DeepSeekClient
    from sluice.fundamentals.store import FundamentalsStore
    from sluice.market_calendar import last_closed_session
    from sluice.news.store import NewsStore
    from sluice.prices.store import PriceStore
    from sluice.prices.yahoo import fetch_bars

    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        typer.echo(f"{ENV_KEY} is not set (put it in .env).", err=True)
        raise typer.Exit(code=2)
    config = load_config(config_version)
    first = date.fromisoformat(start) if start else batching.FIRST_DAY
    last = date.fromisoformat(end) if end else last_closed_session(datetime.now(UTC))
    days = batching.decision_days(first, last, every)
    cells = batching.grid([t for t in tickers.split(",") if t.strip()], days, repeats)
    if not cells:
        typer.echo("Nothing to run: no decision day in that range.", err=True)
        raise typer.Exit(code=2)

    statements = FundamentalsStore()
    checkpointer = open_checkpointer()
    opened: list[NewsStore] = []
    console = Console()
    try:
        with (
            PriceStore(fetch=fetch_bars) as prices,
            CallCache() as cache,
            _weighting(config, cache) as weighting,
        ):
            tables = weighting.tables if weighting else None
            problems = batching.unprepared(
                cells, statements, prices, config.prices.benchmark, tables
            )
            for problem in problems:
                typer.echo(f"Not ready: {problem}", err=True)
            if problems and not allow_gaps:
                typer.echo("Fix these or pass --allow-gaps.", err=True)
                raise typer.Exit(code=2)
            llm = CachedLLM(cache, DeepSeekClient(key))

            def make_graph():
                # One news store per worker: its connection serves one run at a time.
                store = NewsStore(db)
                opened.append(store)
                deps = Deps(store, prices, statements, llm, config, weighting=weighting)
                return build_graph(deps, checkpointer)

            console.print(
                f"{len(cells)} runs: {len({c.ticker for c in cells})} tickers, {len(days)} days "
                f"({days[0]} .. {days[-1]}), config {config.version}-{config.digest}"
            )
            with Progress(console=console) as bar:
                task = bar.add_task("running", total=len(cells))

                def on_done(result: batching.CellResult) -> None:
                    if result.status != "ok":
                        bar.console.print(
                            f"{result.status}: {result.ticker} {result.day} r{result.repeat}: "
                            f"{result.failure}",
                            markup=False,
                        )
                    bar.advance(task)

                results = batching.run_batch(cells, config, make_graph, out_dir, workers, on_done)
    finally:
        for store in opened:
            store.close()
        checkpointer.conn.close()

    counts = {
        status: sum(r.status == status for r in results) for status in ("ok", "failed", "error")
    }
    totals, cost = batching.estimated_cost(results)
    console.print(f"ok {counts['ok']}, failed {counts['failed']}, broke off {counts['error']}")
    for model, (used_in, used_out) in sorted(totals.items()):
        console.print(f"  {model}: {used_in:,} in, {used_out:,} out")
    console.print(f"  about ${cost:.2f} at peak list prices if nothing had been cached")
    console.print(f"results in {batching.run_dir(out_dir, config)}")
    if counts["error"]:
        typer.echo("Some runs broke off. Rerun to continue.", err=True)
    if counts["error"] or counts["failed"]:
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate(
    configs: Annotated[
        list[str] | None,
        typer.Option("--config", help="Pipeline config to evaluate; may be given more than once."),
    ] = None,
    tickers: Annotated[
        str, typer.Option("--tickers", help="Comma-separated tickers in the portfolio.")
    ] = ",".join(COMPANIES),
    runs: Annotated[
        Path, typer.Option("--runs", help="Where the batch run states are.")
    ] = Path("data/local/runs"),
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the report here as Markdown (default by segment)."),
    ] = None,
    sizing: Annotated[
        str | None,
        typer.Option("--sizing", help="Also size with this config under configs/sizing/."),
    ] = None,
    final: Annotated[
        str | None,
        typer.Option(
            "--final",
            help="The one evaluation of a sealed segment (test_1, later test_2): the champion "
            "against every baseline. Opens the seal; cannot be repeated.",
        ),
    ] = None,
) -> None:
    """Portfolio backtest of pipeline ratings on the validation period, against the baselines."""
    from datetime import UTC, datetime

    from sluice.agents.batch import run_dir
    from sluice.agents.config import load_config
    from sluice.evaluation.report import build
    from sluice.news.store import NewsStore
    from sluice.news.weight_tables import DEFAULT_ROOT, load_all
    from sluice.prices.store import PriceStore
    from sluice.prices.yahoo import fetch_bars
    from sluice.scoring.reports import ReportScoreStore
    from sluice.scoring.store import ScoreStore
    from sluice.sizing import plan
    from sluice.sizing.config import load_sizing_config

    symbols = [t.strip() for t in tickers.split(",") if t.strip()]
    if final:
        _evaluate_final(final, symbols, runs, out)
        return
    folders = [run_dir(runs, load_config(name)) for name in configs or ["v1", "v2"]]
    if absent := [str(f) for f in folders if not f.is_dir()]:
        typer.echo(f"No runs in {', '.join(absent)} (run sluice batch first).", err=True)
        raise typer.Exit(code=2)
    out = out or Path("data/local/evaluation/validation.md")
    with (
        PriceStore(fetch=fetch_bars) as prices,
        NewsStore(Path("data/news.db")) as news,
        ScoreStore() as scores,
        ReportScoreStore() as report_scores,
    ):
        try:
            config = load_sizing_config(sizing) if sizing else None
            src = None
            if config:
                tables = load_all(DEFAULT_ROOT, config.inputs.scoring_config)
                src = plan.Sources(news, scores, tables, report_scores, prices)
            report = build(folders, folders, symbols, prices, src, config, datetime.now(UTC))
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    Console().print(report, markup=False)
    Console().print(f"written to {out}")


def _evaluate_final(segment: str, symbols: list[str], runs: Path, out: Path | None) -> None:
    """Check everything, score the segment's reports, work the report out, then open the seal."""
    import os
    from datetime import UTC, datetime

    from sluice.agents.batch import run_dir
    from sluice.agents.cache import CallCache
    from sluice.agents.config import load_config
    from sluice.evaluation import final as fin
    from sluice.evaluation.decisions import load_decisions
    from sluice.evaluation.report import build, calendar
    from sluice.evaluation.seal import AlreadyOpened, opened
    from sluice.evaluation.split import SEALED, SealedError
    from sluice.evolution.library import load_champion
    from sluice.news.store import NewsStore
    from sluice.news.weight_tables import DEFAULT_ROOT, load_all
    from sluice.prices.store import PriceStore
    from sluice.prices.yahoo import fetch_bars
    from sluice.scoring.client import ENV_KEY, CachedJev, JevClient
    from sluice.scoring.config import load_report_scoring_config
    from sluice.scoring.reports import ReportScoreStore, load_reports, score_reports
    from sluice.scoring.store import ScoreStore
    from sluice.sizing import plan

    def refuse(message: str) -> None:
        typer.echo(f"{message}. The seal stays closed.", err=True)
        raise typer.Exit(code=2)

    if segment not in SEALED:
        refuse(f"{segment} is not a sealed segment")
    if any(seal.segment == segment for seal in opened()):
        refuse(f"{segment} has already had its one evaluation")
    commit = _commit(paths=("sluice", "configs"))
    champion = load_champion()
    pipeline = run_dir(runs, load_config(fin.PIPELINE))
    external = runs / fin.EXTERNAL
    now = datetime.now(UTC)
    try:
        cal = calendar(segment, final=True)
    except ValueError as exc:
        refuse(str(exc))
    if late := fin.too_early(cal, now):
        refuse(late)
    problems = []
    for name, folder in ((fin.PIPELINE, pipeline), (fin.EXTERNAL, external)):
        decisions = load_decisions(folder, segment, final=True) if folder.is_dir() else []
        problems += fin.missing_runs(name, decisions, symbols, cal, fin.REPEATS[name])
    if problems:
        refuse("; ".join(problems))
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        refuse(f"{ENV_KEY} is not set: the segment's analyst reports need scoring")

    with (
        PriceStore(fetch=fetch_bars) as prices,
        NewsStore(Path("data/news.db")) as news,
        ScoreStore() as scores,
        ReportScoreStore() as report_scores,
        CallCache() as cache,
    ):
        spec = load_report_scoring_config(champion.inputs.report_spec)
        items = load_reports(pipeline, segment, final=True)
        scored = score_reports(items, report_scores, spec, CachedJev(cache, JevClient(key)), 4)
        if scored.failed:
            refuse(f"{scored.failed} reports could not be scored; run again")
        tables = load_all(DEFAULT_ROOT, champion.inputs.scoring_config)
        src = plan.Sources(news, scores, tables, report_scores, prices)

        def compute() -> str:
            return build(
                [pipeline, external], [pipeline], symbols, prices, src, champion, now,
                requested=segment, final=True,
            )  # fmt: skip

        try:
            seal, report = fin.evaluate_once(
                segment, compute, fin.sealed_config(pipeline.name, champion), commit
            )
        except (ValueError, SealedError, AlreadyOpened) as exc:
            refuse(str(exc))
    header = (
        f"> 最终评估，只此一次：{seal.segment} 于 {seal.opened_at:%Y-%m-%d %H:%M} UTC 开封；"
        f"配置 {seal.config}；提交 {seal.commit}。\n\n"
    )
    out = out or Path(f"data/local/evaluation/{segment}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    Console().print(header + report, markup=False)  # shown first: the seal is already open
    out.write_text(header + report)
    Console().print(f"written to {out}")


@app.command("score-reports")
def score_reports_command(
    configs: Annotated[
        list[str] | None,
        typer.Option("--config", help="Pipeline config whose runs to score; repeatable."),
    ] = None,
    spec: Annotated[
        str, typer.Option("--spec", help="Jev problem spec for reports, under configs/scoring/.")
    ] = "reports-v1",
    runs: Annotated[
        Path, typer.Option("--runs", help="Where the batch run states are.")
    ] = Path("data/local/runs"),
    workers: Annotated[int, typer.Option("--workers", help="Jev calls going on at a time.")] = 4,
) -> None:
    """Jev's stance on each analyst report of the validation runs. Resumable."""
    import os

    from rich.progress import Progress

    from sluice.agents.batch import run_dir
    from sluice.agents.cache import CallCache
    from sluice.agents.config import load_config
    from sluice.scoring.client import ENV_KEY, PRICE_PER_MILLION_INPUT, CachedJev, JevClient
    from sluice.scoring.config import load_report_scoring_config
    from sluice.scoring.reports import ReportScoreStore, load_reports, score_reports

    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        typer.echo(f"{ENV_KEY} is not set (put it in .env).", err=True)
        raise typer.Exit(code=2)
    config = load_report_scoring_config(spec)
    items = [
        item
        for name in configs or ["v1", "v2"]
        for item in load_reports(run_dir(runs, load_config(name)), "validation")
    ]
    console = Console()
    with ReportScoreStore() as store, CallCache() as cache, Progress(console=console) as bar:
        task = bar.add_task("reports", total=len(items))
        result = score_reports(
            items, store, config, CachedJev(cache, JevClient(key)), workers,
            on_step=lambda: bar.advance(task),
        )  # fmt: skip
    cost = result.input_tokens / 1e6 * PRICE_PER_MILLION_INPUT
    console.print(
        f"{len(items)} reports: scored {result.scored}, already done {result.done}, "
        f"too large {result.skipped}, failed {result.failed}; about ${cost:.3f}"
    )
    if result.failed:
        typer.echo("Some reports failed. Rerun to continue.", err=True)
        raise typer.Exit(code=1)


@app.command("score")
def score(
    tickers: Annotated[
        str, typer.Option("--tickers", help="Comma-separated tickers to score.")
    ] = ",".join(COMPANIES),
    config_version: Annotated[
        str, typer.Option("--config", help="Scoring config version under configs/scoring/.")
    ] = "v1",
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
    scores_db: Annotated[
        Path, typer.Option("--scores-db", help="SQLite score store.")
    ] = Path("data/local/scores.db"),
    workers: Annotated[int, typer.Option("--workers", help="Jev calls going on at a time.")] = 4,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Score at most this many new items.")
    ] = None,
) -> None:
    """Score stored news with Jev. Interrupt and rerun freely: it resumes."""
    import os

    from rich.progress import Progress

    from sluice.agents.cache import CallCache
    from sluice.news.store import NewsStore
    from sluice.scoring.client import ENV_KEY, PRICE_PER_MILLION_INPUT, CachedJev, JevClient
    from sluice.scoring.config import load_scoring_config
    from sluice.scoring.score import score_all
    from sluice.scoring.store import ScoreStore

    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        typer.echo(f"{ENV_KEY} is not set (put it in .env).", err=True)
        raise typer.Exit(code=2)

    config = load_scoring_config(config_version)
    symbols = [t for t in tickers.split(",") if t.strip()]
    console = Console()
    counts = {"scored": 0, "skipped": 0, "done": 0, "error": 0}
    tokens_in = tokens_out = 0

    with (
        NewsStore(db) as store,
        ScoreStore(scores_db) as scores,
        CallCache() as cache,
        Progress(console=console) as bar,
    ):
        jev = CachedJev(cache, JevClient(key))
        task = bar.add_task("scoring", total=None)

        def on_done(result) -> None:
            nonlocal tokens_in, tokens_out
            counts[result.status] += 1
            tokens_in += result.input_tokens
            tokens_out += result.output_tokens
            if result.status == "error":
                bar.console.print(
                    f"error: news {result.news_id} ({result.ticker}): {result.detail}",
                    markup=False,
                )
            bar.advance(task)

        score_all(symbols, store, scores, config, jev, workers=workers, limit=limit, on_done=on_done)

    console.print(
        f"scored {counts['scored']}, skipped {counts['skipped']}, "
        f"already done {counts['done']}, errors {counts['error']}"
    )
    console.print(
        f"  {tokens_in:,} in, {tokens_out:,} out, "
        f"about ${tokens_in * PRICE_PER_MILLION_INPUT / 1_000_000:.4f} at peak list prices"
    )
    if counts["error"]:
        typer.echo("Some items failed. Rerun to continue.", err=True)
        raise typer.Exit(code=1)


@calibration_app.command("sample")
def calibration_sample(
    tickers: Annotated[
        str, typer.Option("--tickers", help="Comma-separated tickers to sample from.")
    ] = ",".join(COMPANIES),
    config_version: Annotated[
        str, typer.Option("--config", help="Scoring config version under configs/scoring/.")
    ] = "v1",
    n: Annotated[int, typer.Option("--n", help="How many items to sample.")] = 50,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = 0,
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
    scores_db: Annotated[
        Path, typer.Option("--scores-db", help="SQLite score store.")
    ] = Path("data/local/scores.db"),
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where the review and answer files go.")
    ] = Path("data/local/calibration"),
) -> None:
    """Sample scored news for hand-labeling, spread across Jev's event types."""
    import json

    from sluice.news.store import NewsStore
    from sluice.scoring.calibration import build_review, candidates, stratified_sample
    from sluice.scoring.config import load_scoring_config
    from sluice.scoring.store import ScoreStore

    config = load_scoring_config(config_version)
    symbols = [t for t in tickers.split(",") if t.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    review_path = out_dir / f"{config_version}_review.md"
    answers_path = out_dir / f"{config_version}_answers.json"

    with NewsStore(db) as store, ScoreStore(scores_db) as scores:
        pool = candidates(symbols, store, scores, config)
        chosen = stratified_sample(pool, n, seed)
        text, answers = build_review(chosen)

    review_path.write_text(text)
    answers_path.write_text(
        json.dumps({str(i): s.model_dump() for i, s in answers.items()}, indent=2) + "\n"
    )
    Console().print(
        f"sampled {len(chosen)} of {len(pool)} scored items\n"
        f"fill in the blanks in {review_path}\n"
        f"do not open {answers_path} until you are done -- it holds Jev's own scores"
    )


@calibration_app.command("report")
def calibration_report(
    config_version: Annotated[
        str, typer.Option("--config", help="Scoring config version the sample was drawn from.")
    ] = "v1",
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where the review and answer files are.")
    ] = Path("data/local/calibration"),
    bins: Annotated[int, typer.Option("--bins", help="Reliability-diagram buckets.")] = 5,
) -> None:
    """Compare your filled-in labels against Jev's scores."""
    import json

    from sluice.scoring.calibration import choice_bins, noul_bins, parse_review
    from sluice.scoring.store import ItemScores

    review_path = out_dir / f"{config_version}_review.md"
    answers_path = out_dir / f"{config_version}_answers.json"
    labels = parse_review(review_path.read_text())
    answers = {
        int(i): ItemScores(**s) for i, s in json.loads(answers_path.read_text()).items()
    }

    console = Console()

    def noul_table(title: str, pairs: list[tuple[float, bool]]) -> None:
        table = Table(title=title)
        table.add_column("Jev's mean score")
        table.add_column("human-observed rate")
        table.add_column("n")
        for predicted, observed, count in noul_bins(pairs, bins):
            table.add_row(f"{predicted:.2f}", f"{observed:.2f}", str(count))
        console.print(table)

    def choice_table(title: str, triples: list[tuple[bool, float]]) -> None:
        table = Table(title=title)
        table.add_column("Jev's mean confidence")
        table.add_column("accuracy")
        table.add_column("n")
        for confidence, accuracy, count in choice_bins(triples, bins):
            table.add_row(f"{confidence:.2f}", f"{accuracy:.2f}", str(count))
        console.print(table)

    noul_table(
        "relevance", [(answers[i].relevance, labels[i].relevant) for i in labels]
    )
    noul_table("novelty", [(answers[i].novelty, labels[i].novel) for i in labels])
    noul_table("rumour", [(answers[i].rumour, labels[i].rumour) for i in labels])
    choice_table(
        "stance",
        [
            (answers[i].stance == labels[i].stance, answers[i].stance_confidence)
            for i in labels
        ],
    )
    choice_table(
        "event_type",
        [
            (answers[i].event_type == labels[i].event_type, answers[i].event_type_confidence)
            for i in labels
        ],
    )


@fundamentals_app.command("fetch")
def fundamentals_fetch(
    tickers: Annotated[
        str, typer.Option("--tickers", help="Comma-separated tickers to fetch.")
    ] = ",".join(COMPANIES),
    raw_dir: Annotated[
        Path, typer.Option("--raw-dir", help="Where the vendor's answers are kept.")
    ] = Path("data/local/raw/alphavantage"),
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Fetch again what is already stored.")
    ] = False,
) -> None:
    """Fetch quarterly statements from Alpha Vantage: three calls a ticker, 25 a day. Resumable."""
    import os

    from sluice.fundamentals.alphavantage import Client
    from sluice.fundamentals.store import FundamentalsStore, fetch

    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    if not key:
        typer.echo("ALPHA_VANTAGE_API_KEY is not set (put it in .env).", err=True)
        raise typer.Exit(code=2)
    symbols = [t for t in tickers.split(",") if t.strip()]
    result = fetch(symbols, Client(key), FundamentalsStore(raw_dir), refresh=refresh)
    Console().print(f"fetched {result.fetched}, already stored {result.already_stored}")
    for failure in result.failed:
        typer.echo(f"Failed: {failure}", err=True)
    if result.stopped:
        typer.echo(f"Stopped: {result.stopped}. Rerun tomorrow to continue.", err=True)
    if result.stopped or result.failed:
        raise typer.Exit(code=1)


@news_app.command("collect")
def news_collect(
    tickers: Annotated[
        str, typer.Option("--tickers", help="Comma-separated tickers to collect.")
    ] = ",".join(UNIVERSE),
    source: Annotated[
        str, typer.Option("--source", help="yahoo, finnhub or all. finnhub needs FINNHUB_API_KEY.")
    ] = "yahoo",
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
    raw_dir: Annotated[
        Path, typer.Option("--raw-dir", help="Where raw items are archived as JSONL.")
    ] = Path("data/local/raw"),
    run_log: Annotated[
        Path | None, typer.Option("--run-log", help="Directory for per-run result records.")
    ] = None,
) -> None:
    """Fetch recent news and add it to the store."""
    import os
    from datetime import UTC, datetime

    from sluice.news import finnhub, yahoo
    from sluice.news.archive import write_run_log
    from sluice.news.store import NewsStore

    if source not in ("yahoo", "finnhub", "all"):
        raise typer.BadParameter("choose yahoo, finnhub or all", param_hint="--source")
    symbols = [t for t in tickers.split(",") if t.strip()]
    token = os.environ.get("FINNHUB_API_KEY", "").strip()
    failed = False
    table = Table("source", "ticker", "fetched", "new", "known", "dropped", "error")

    with NewsStore(db) as store:
        runs = []
        if source in ("yahoo", "all"):
            runs.append((yahoo.SOURCE, lambda: yahoo.collect(symbols, store, raw_dir=raw_dir)))
        if source in ("finnhub", "all"):
            if token:
                fetch = finnhub.Client(token).company_news
                runs.append(
                    (finnhub.SOURCE, lambda: finnhub.collect(symbols, store, fetch, raw_dir))
                )
            else:
                # Yahoo still runs: a missing key must not cost the data that needs none.
                typer.echo("FINNHUB_API_KEY is not set; Finnhub was not collected.", err=True)
                failed = True
        for name, run in runs:
            started_at = datetime.now(UTC)
            results = run()
            if run_log is not None:
                write_run_log(run_log, started_at, name, results)
            for r in results:
                failed = failed or r.error is not None
                table.add_row(
                    name,
                    r.ticker,
                    str(r.fetched),
                    str(r.inserted),
                    str(r.skipped),
                    str(r.dropped),
                    r.error or "",
                )
        store.rebuild_clusters()
    Console().print(table)
    if failed:
        raise typer.Exit(code=1)


@news_app.command("import")
def news_import(
    raw_dir: Annotated[
        list[Path] | None,
        typer.Option("--raw-dir", help="Raw JSONL archive to load; may be given more than once."),
    ] = None,
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
) -> None:
    """Load archived raw items from every source into the store. Safe to repeat."""
    from sluice.news import finnhub, yahoo
    from sluice.news.store import NewsStore

    console = Console()
    with NewsStore(db) as store:
        for directory in raw_dir or [Path("data/raw"), Path("data/local/raw")]:
            for module in (yahoo, finnhub):
                r = module.import_archive(directory, store)
                console.print(
                    f"{directory / module.SOURCE}: {r.records} records, {r.inserted} new, "
                    f"{r.skipped} known, {r.dropped} dropped"
                )
        console.print(f"{store.rebuild_clusters()} articles are reprints of an earlier one")


@news_app.command("resolve-links")
def news_resolve_links(
    raw_dir: Annotated[
        Path, typer.Option("--raw-dir", help="Raw JSONL archive holding the Finnhub items.")
    ] = Path("data/raw"),
    limit: Annotated[
        int | None, typer.Option("--limit", help="Follow at most this many links.")
    ] = None,
) -> None:
    """Record where Finnhub's article links lead. Interrupt and rerun freely: it resumes."""
    from rich.progress import Progress

    from sluice.news.finnhub_links import resolve_links

    with Progress() as bar:
        task = bar.add_task("links", total=None)
        result = resolve_links(
            raw_dir,
            limit=limit,
            on_step=lambda done, total: bar.update(task, completed=done, total=total),
        )
    Console().print(
        f"pending {result.pending}, resolved {result.resolved}, dead {result.dead}, "
        f"failed {result.failed}"
    )
    if result.aborted:
        typer.echo(f"Stopped: {result.aborted}. Rerun to continue.", err=True)
        raise typer.Exit(code=1)


@news_app.command("resolve-publishers")
def news_resolve_publishers(
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
    raw_dir: Annotated[
        Path,
        typer.Option("--raw-dir", help="Raw archive the attempt log is kept in; import reads it."),
    ] = Path("data/raw"),
    limit: Annotated[
        int | None, typer.Option("--limit", help="Fetch at most this many pages.")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", help="Pages fetched at a time.")] = 1,
) -> None:
    """Read Yahoo-hosted articles' own pages for their true publisher. Resumable."""
    from rich.progress import Progress

    from sluice.news.store import NewsStore
    from sluice.news.yahoo_publishers import resolve_publishers

    with NewsStore(db) as store:
        candidates = store.missing_publisher_urls("finnhub")
        with Progress() as bar:
            task = bar.add_task("pages", total=None)
            result = resolve_publishers(
                store,
                candidates,
                raw_dir,
                limit=limit,
                workers=workers,
                on_step=lambda done, total: bar.update(task, completed=done, total=total),
            )
    Console().print(
        f"pending {result.pending}, found {result.found}, no provider {result.no_provider}, "
        f"dead {result.dead}, failed {result.failed}"
    )
    if result.aborted:
        typer.echo(f"Stopped: {result.aborted}. Rerun to continue.", err=True)
        raise typer.Exit(code=1)


@news_app.command("fit-weights")
def news_fit_weights(
    start: Annotated[
        str, typer.Option("--from", help="First month to fit, YYYY-MM-DD (its month counts).")
    ] = "2025-10-01",
    scoring_config: Annotated[
        str, typer.Option("--scoring-config", help="Jev scoring config the weights are fitted on.")
    ] = "v1",
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
    scores_db: Annotated[
        Path, typer.Option("--scores-db", help="SQLite score store.")
    ] = Path("data/local/scores.db"),
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where the monthly tables are saved.")
    ] = Path("data/local/news_weights"),
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Refit months already saved, too.")
    ] = False,
) -> None:
    """Fit and save one weight table per month start. Saved months are kept as they are."""
    from datetime import UTC, date, datetime

    from sluice.news.store import NewsStore
    from sluice.news.walkforward import monthly_cutoffs, walk_forward_fit
    from sluice.news.weight_tables import fit_missing
    from sluice.prices.store import PriceStore
    from sluice.prices.yahoo import fetch_bars
    from sluice.scoring.store import ScoreStore

    first = datetime.combine(date.fromisoformat(start), datetime.min.time(), tzinfo=UTC)
    cutoffs = monthly_cutoffs(first, datetime.now(UTC))
    with (
        NewsStore(db) as news,
        ScoreStore(scores_db) as scores,
        PriceStore(fetch=fetch_bars) as prices,
    ):
        result = fit_missing(
            cutoffs,
            out_dir,
            scoring_config,
            lambda todo: walk_forward_fit(
                news, scores, prices, list(COMPANIES), todo, config_version=scoring_config
            ),
            refresh=refresh,
        )

    def days(moments: list[datetime]) -> str:
        return ", ".join(f"{m:%Y-%m-%d}" for m in moments) or "-"

    console = Console()
    console.print(f"written: {days(result.written)}")
    console.print(f"kept: {days(result.kept)}")
    console.print(f"too few rows: {days(result.too_few)}")


@news_app.command("backfill")
def news_backfill(
    start: Annotated[
        str | None, typer.Option("--from", help="First day, YYYY-MM-DD. Default: one year ago.")
    ] = None,
    end: Annotated[
        str | None, typer.Option("--to", help="Last day, YYYY-MM-DD. Default: yesterday (UTC).")
    ] = None,
    tickers: Annotated[
        str, typer.Option("--tickers", help="Comma-separated tickers to backfill.")
    ] = ",".join(UNIVERSE),
    raw_dir: Annotated[
        Path, typer.Option("--raw-dir", help="Where raw items are archived as JSONL.")
    ] = Path("data/raw"),
    progress: Annotated[
        Path, typer.Option("--progress", help="Record of finished (ticker, day) pairs.")
    ] = Path("data/backfill/finnhub.jsonl"),
) -> None:
    """Archive a year of Finnhub company news. Interrupt and rerun freely: it resumes."""
    import os
    from datetime import UTC, date, datetime, timedelta

    from rich.progress import Progress

    from sluice.news.finnhub import Client, backfill

    token = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not token:
        typer.echo("FINNHUB_API_KEY is not set (put it in .env).", err=True)
        raise typer.Exit(code=2)
    today = datetime.now(UTC).date()
    first = date.fromisoformat(start) if start else today - timedelta(days=365)
    last = date.fromisoformat(end) if end else today - timedelta(days=1)

    with Progress() as bar:
        task = bar.add_task(f"{first} .. {last}", total=None)
        result = backfill(
            [t for t in tickers.split(",") if t.strip()],
            first,
            last,
            Client(token).company_news,
            raw_dir,
            progress,
            on_step=lambda done, total: bar.update(task, completed=done, total=total),
        )
    Console().print(
        f"requested {result.requested}, already done {result.already_done}, "
        f"items {result.items}, newly archived {result.archived}, "
        f"capped days {result.capped}, failed {result.failed}"
    )
    if result.aborted:
        typer.echo(f"Stopped: {result.aborted}. Rerun to continue.", err=True)
    if result.aborted or result.failed:
        raise typer.Exit(code=1)


def _commit(paths: tuple[str, ...] = ("sluice",)) -> str:
    """HEAD, refusing uncommitted changes under `paths`: the recorded commit must reproduce."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    if git("status", "--porcelain", "--", *paths).strip():
        listed = ", ".join(f"{p}/" for p in paths)
        typer.echo(f"{listed} has uncommitted changes; commit them first.", err=True)
        raise typer.Exit(code=2)
    return git("rev-parse", "--short", "HEAD").strip()


@contextmanager
def _gate_backtest(runs: Path) -> Iterator:
    from sluice.evolution.library import load_champion
    from sluice.evolution.validation import Validation
    from sluice.news.store import NewsStore
    from sluice.news.weight_tables import DEFAULT_ROOT, load_all
    from sluice.prices.store import PriceStore
    from sluice.prices.yahoo import fetch_bars
    from sluice.scoring.reports import ReportScoreStore
    from sluice.scoring.store import ScoreStore
    from sluice.sizing import plan

    champion = load_champion()
    with (
        PriceStore(fetch=fetch_bars) as prices,
        NewsStore(Path("data/news.db")) as news,
        ScoreStore() as scores,
        ReportScoreStore() as report_scores,
    ):
        tables = load_all(DEFAULT_ROOT, champion.inputs.scoring_config)
        src = plan.Sources(news, scores, tables, report_scores, prices)
        try:
            yield Validation(src, prices, list(COMPANIES), runs)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc


def _show(outcome) -> None:
    entry = outcome.entry
    lines = [f"{outcome.status}" + (f" (trial {entry.n})" if entry.n else "")]
    lines += [f"  problem: {p}" for p in entry.problems]
    if e := entry.evidence:
        pairs = ", ".join(
            f"{c:.2f} vs {h:.2f}"
            for c, h in zip(e.candidate_sharpes, e.champion_sharpes, strict=True)
        )
        lines.append(f"  Sharpe per repeat (candidate vs champion): {pairs}")
        lines.append(
            f"  averaged: {e.candidate_sharpe:.2f} vs {e.champion_sharpe:.2f}, "
            f"rho {e.rho:.3f}, z {e.z:+.2f} over {e.days} days"
        )
    if outcome.version:
        lines.append(f"  champion is now {outcome.version}")
    Console().print("\n".join(lines), markup=False)


RUNS_OPTION = typer.Option("--runs", help="Where the batch run states are.")


@evolve_app.command("seed")
def evolve_seed(runs: Annotated[Path, RUNS_OPTION] = Path("data/local/runs")) -> None:
    """Log the three sizing configs 6.12 saw: v1, then k = 0 and exposure alone through the gate."""
    from sluice.evolution.gate import seed

    commit = _commit()
    with _gate_backtest(runs) as backtest:
        for outcome in seed(backtest, commit):
            _show(outcome)
    evolve_status()


@evolve_app.command("try")
def evolve_try(
    changes: Annotated[
        list[str], typer.Option("--set", help="key=value, e.g. k=0.3 or stocks.vol_scaling=false.")
    ],
    rationale: Annotated[str, typer.Option("--rationale", help="Why, for the log.")] = "",
    runs: Annotated[Path, RUNS_OPTION] = Path("data/local/runs"),
) -> None:
    """Put one hand-written proposal through the gate. Uses a trial if it gets evaluated."""
    import tomllib

    from sluice.evolution.gate import propose

    parsed = {}
    for item in changes:
        key, sep, value = item.partition("=")
        try:
            if not sep:
                raise ValueError("no '='")
            parsed[key.strip()] = tomllib.loads(f"v = {value.strip()}")["v"]
        except ValueError as exc:
            typer.echo(f"--set {item}: expected key=value ({exc})", err=True)
            raise typer.Exit(code=2) from exc
    commit = _commit()
    with _gate_backtest(runs) as backtest:
        _show(propose(parsed, "manual", backtest, commit, rationale))


@evolve_app.command("status")
def evolve_status() -> None:
    """The champion, the trials left and the experiment log."""
    from sluice.evolution.experiments import ExperimentLog
    from sluice.evolution.library import load_champion

    champion = load_champion()
    log = ExperimentLog()
    console = Console()
    console.print(
        f"champion {champion.version}-{champion.digest}; "
        f"trials used {log.used()}, left {log.remaining()}",
        markup=False,
    )
    table = Table("n", "status", "source", "against", "changes", "z", "version")
    for e in log.entries():
        changes = ", ".join(f"{k}={v}" for k, v in e.changes.items()) or "-"
        z = f"{e.evidence.z:+.2f}" if e.evidence else "-"
        table.add_row(
            str(e.n or "-"), e.status, e.source, e.parent or "-", changes, z, e.version or "-"
        )
    console.print(table)


@evolve_app.command("rollback")
def evolve_rollback(version: str) -> None:
    """Point CHAMPION at an existing version. Commit configs/sizing/CHAMPION afterwards."""
    from sluice.evolution.library import champion, set_champion

    before = champion()
    try:
        set_champion(version)
    except FileNotFoundError as exc:
        typer.echo(f"no configs/sizing/{version}.toml", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"champion {before} -> {version}")


@evolve_app.command("propose")
def evolve_propose(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the proposal only: no gate, no trial used.")
    ] = False,
    runs: Annotated[Path, RUNS_OPTION] = Path("data/local/runs"),
) -> None:
    """Ask the LLM proposer for one sizing change and put it through the gate."""
    import os

    from sluice.agents.cache import CallCache
    from sluice.agents.llm import ENV_KEY, CachedLLM, DeepSeekClient, LLMError
    from sluice.evolution.experiments import ExperimentLog, TrialBudgetExhausted
    from sluice.evolution.gate import propose
    from sluice.evolution.library import load_champion
    from sluice.evolution.proposer import ask, build_request

    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        typer.echo(f"{ENV_KEY} is not set (put it in .env).", err=True)
        raise typer.Exit(code=2)
    log = ExperimentLog()
    try:
        log.require_budget()  # before paying for a proposal that could not be judged
    except TrialBudgetExhausted as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    commit = "" if dry_run else _commit()
    champion = load_champion()
    text = (Path("configs/sizing") / f"{champion.version}.toml").read_text()
    with _gate_backtest(runs) as validation, CallCache() as cache:
        request = build_request(
            champion, text, log.remaining(), log.entries(), validation.periods(champion),
            validation(champion), list(COMPANIES),
        )  # fmt: skip
        try:
            proposal, answers = ask(CachedLLM(cache, DeepSeekClient(key)), request)
        except LLMError as exc:
            typer.echo(f"no proposal: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        paid = [a for a in answers if not a.cached]
        console = Console()
        console.print(
            f"proposal: {proposal.changes or '(none)'}\nrationale: {proposal.rationale}\n"
            f"answers {len(answers)}, new calls {len(paid)}, tokens in "
            f"{sum(a.input_tokens for a in paid)}, out {sum(a.output_tokens for a in paid)}",
            markup=False,
        )
        if dry_run:
            console.print("dry run: not put through the gate, no trial used", markup=False)
            return
        _show(propose(proposal.changes, "llm", validation, commit, proposal.rationale))
