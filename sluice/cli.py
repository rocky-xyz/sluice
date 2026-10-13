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


@app.command("analyze")
def analyze(
    ticker: Annotated[str, typer.Argument(help="Ticker to analyse.")],
    day: Annotated[
        str, typer.Option("--date", help="Decision day (YYYY-MM-DD); decides at its close.")
    ],
    config_version: Annotated[
        str, typer.Option("--config", help="Pipeline config version under configs/pipeline/.")
    ] = "v1",
    repeat: Annotated[int, typer.Option("--repeat", help="Which repeat of this run.")] = 0,
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the full run state to this JSON file.")
    ] = None,
) -> None:
    """Run the research pipeline for one ticker as of one trading day's close."""
    import json
    import os
    from datetime import date

    from sluice.agents.cache import CallCache
    from sluice.agents.config import load_config
    from sluice.agents.graph import Deps, build_graph, open_checkpointer, run
    from sluice.agents.llm import ENV_KEY, CachedLLM, DeepSeekClient
    from sluice.agents.prefetch import backtest_decision_time
    from sluice.agents.state import dump_state
    from sluice.fundamentals.store import FundamentalsStore
    from sluice.news.store import NewsStore
    from sluice.prices.store import PriceStore
    from sluice.prices.yahoo import fetch_bars

    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        typer.echo(f"{ENV_KEY} is not set (put it in .env).", err=True)
        raise typer.Exit(code=2)
    try:
        decision_time = backtest_decision_time(date.fromisoformat(day))
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--date") from exc
    config = load_config(config_version)

    checkpointer = open_checkpointer()
    try:
        with NewsStore(db) as store, PriceStore(fetch=fetch_bars) as prices, CallCache() as cache:
            llm = CachedLLM(cache, DeepSeekClient(key))
            deps = Deps(store, prices, FundamentalsStore(), llm, config)
            graph = build_graph(deps, checkpointer)
            state = run(graph, config, ticker, decision_time, repeat=repeat)
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
