import re

from typer.testing import CliRunner

from sluice import cli

runner = CliRunner()

# Typer colours its help whenever GITHUB_ACTIONS is set, so CI output carries
# ANSI codes between the words; the help tests read it with them removed.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def unstyle(text: str) -> str:
    return ANSI.sub("", text)


def test_sluice_alone_without_a_terminal_shows_the_help():
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0
    assert "Run `sluice` alone in a terminal" in unstyle(result.output)


def test_the_developer_session_without_a_terminal_also_shows_the_help():
    result = runner.invoke(cli.app, ["--dev"])
    assert result.exit_code == 0
    assert "sluice --dev" in unstyle(result.output)


def test_the_help_groups_the_other_commands():
    result = runner.invoke(cli.app, ["--help"], env={"COLUMNS": "120"})
    assert result.exit_code == 0
    output = unstyle(result.output)
    research = output.index("Research")
    data = output.index("Data")
    for name in ("batch", "evaluate", "score-reports", "evolve", "patterns", "calibration"):
        assert research < output.index(f" {name} ") < data, name
    for name in ("news", "fundamentals", "score "):
        assert output.index(f" {name}", data) > data, name


def test_analyze_and_configure_are_gone():
    for name in ("analyze", "configure"):
        result = runner.invoke(cli.app, [name])
        assert result.exit_code == 2
        assert "No such command" in result.output


def test_the_version_still_answers():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0 and result.output.startswith("sluice ")
