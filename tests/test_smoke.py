from typer.testing import CliRunner

import sluice
from sluice.cli import app


def test_version_flag():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert sluice.__version__ in result.output
