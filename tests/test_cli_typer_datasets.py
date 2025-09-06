import json
from pathlib import Path

from typer.testing import CliRunner

import bench.cli_typer as cli

runner = CliRunner()


def test_list_datasets_json_uses_default_registry(tmp_path: Path):
    # Use default packaged registry
    result = runner.invoke(cli.app, ["list-datasets", "--json"])  # no path
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    ids = [d.get("id") for d in data]
    assert "medaisure-core" in ids


def test_show_dataset_core_json(tmp_path: Path):
    result = runner.invoke(cli.app, ["show-dataset", "medaisure-core", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload.get("id") == "medaisure-core"
    # Ensure composition integrity in output
    comp = payload.get("composition", {})
    size = payload.get("size")
    if size is not None and comp:
        assert sum(comp.values()) == size
