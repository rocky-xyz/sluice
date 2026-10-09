import typer

from sluice import __version__

app = typer.Typer(help="Multi-agent equity research pipeline.", no_args_is_help=True)


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
