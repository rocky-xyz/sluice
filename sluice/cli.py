from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from sluice import __version__
from sluice.universe import UNIVERSE

app = typer.Typer(help="Multi-agent equity research pipeline.", no_args_is_help=True)
news_app = typer.Typer(help="News collection and storage.", no_args_is_help=True)
app.add_typer(news_app, name="news")


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


@news_app.command("collect")
def news_collect(
    tickers: Annotated[
        str, typer.Option("--tickers", help="Comma-separated tickers to collect.")
    ] = ",".join(UNIVERSE),
    db: Annotated[Path, typer.Option("--db", help="SQLite news store.")] = Path("data/news.db"),
    raw_dir: Annotated[
        Path, typer.Option("--raw-dir", help="Where raw items are archived as JSONL.")
    ] = Path("data/local/raw"),
    run_log: Annotated[
        Path | None, typer.Option("--run-log", help="Directory for per-run result records.")
    ] = None,
) -> None:
    """Fetch recent Yahoo Finance news and add it to the store."""
    from datetime import UTC, datetime

    from sluice.news.archive import write_run_log
    from sluice.news.store import NewsStore
    from sluice.news.yahoo import SOURCE, collect

    started_at = datetime.now(UTC)
    with NewsStore(db) as store:
        results = collect([t for t in tickers.split(",") if t.strip()], store, raw_dir=raw_dir)
    if run_log is not None:
        write_run_log(run_log, started_at, SOURCE, results)

    table = Table("ticker", "fetched", "new", "known", "dropped", "error")
    for r in results:
        table.add_row(
            r.ticker, str(r.fetched), str(r.inserted), str(r.skipped), str(r.dropped), r.error or ""
        )
    Console().print(table)
    if any(r.error for r in results):
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
