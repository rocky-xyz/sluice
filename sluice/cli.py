from pathlib import Path
from typing import Annotated

import typer
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
    pass


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

    from sluice.news.store import NewsStore
    from sluice.news.yahoo import collect, write_run_log

    started_at = datetime.now(UTC)
    with NewsStore(db) as store:
        results = collect([t for t in tickers.split(",") if t.strip()], store, raw_dir=raw_dir)
    if run_log is not None:
        write_run_log(run_log, started_at, results)

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
    """Load archived raw items into the store. Safe to repeat."""
    from sluice.news.store import NewsStore
    from sluice.news.yahoo import import_archive

    console = Console()
    with NewsStore(db) as store:
        for directory in raw_dir or [Path("data/raw"), Path("data/local/raw")]:
            r = import_archive(directory, store)
            console.print(
                f"{directory}: {r.records} records, {r.inserted} new, "
                f"{r.skipped} known, {r.dropped} dropped"
            )
