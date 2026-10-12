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
    ] = Path("data/raw"),
) -> None:
    """Fetch recent Yahoo Finance news and add it to the store."""
    from sluice.news.store import NewsStore
    from sluice.news.yahoo import collect

    with NewsStore(db) as store:
        results = collect([t for t in tickers.split(",") if t.strip()], store, raw_dir=raw_dir)

    table = Table("ticker", "fetched", "new", "known", "dropped", "error")
    for r in results:
        table.add_row(
            r.ticker, str(r.fetched), str(r.inserted), str(r.skipped), str(r.dropped), r.error or ""
        )
    Console().print(table)
    if any(r.error for r in results):
        raise typer.Exit(code=1)
