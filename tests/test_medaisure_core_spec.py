from typer.testing import CliRunner
import os

import bench.cli_typer as cli
from bench.data.datasets import medaisure_core as core

runner = CliRunner()


def test_medaisure_core_validate_ok():
    # Should not raise when registry entry is consistent
    core.validate()


def test_list_datasets_with_composition_plain_text():
    # Disable Rich to force plain-text fallback for deterministic assertions
    env = {**os.environ, "MEDAISURE_NO_RICH": "1"}
    result = runner.invoke(cli.app, ["list-datasets", "--with-composition"], env=env)
    assert result.exit_code == 0
    out = result.stdout
    # Should contain medaisure-core id and a composition line
    assert "medaisure-core" in out
    assert (
        'composition : {"diagnostics": 100, "summarization": 50, "communication": 50}'
        in out
    )
