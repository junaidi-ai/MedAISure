import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import bench.cli_typer as cli

runner = CliRunner()


def set_tmp_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    reg = tmp_path / "registry.json"
    # Ensure directory exists
    reg.parent.mkdir(parents=True, exist_ok=True)
    # Point CLI to temp registry
    monkeypatch.setattr(cli, "REGISTRY_FILE", reg, raising=False)
    return reg


def test_list_models_empty_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    set_tmp_registry(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["list-models", "--json"])  # no registry file
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data == {}


def test_list_models_empty_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    set_tmp_registry(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["list-models"])  # pretty output
    assert result.exit_code == 0
    assert "No models registered" in result.stdout


def test_list_models_populated_json_masking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    reg = set_tmp_registry(monkeypatch, tmp_path)
    # Write a sample registry with local and api entries
    registry = {
        "local-bert": {
            "type": "local",
            "path": "/models/bert.bin",
            "module": "mypkg.model_loader",
            "load_func": "load_model",
        },
        "remote-api": {
            "type": "api",
            "endpoint": "https://api.example/v1/predict",
            "api_key": "secret-key",
            "timeout": 15.0,
        },
    }
    reg.write_text(json.dumps(registry))

    # Default: masked
    res_masked = runner.invoke(cli.app, ["list-models", "--json"])
    assert res_masked.exit_code == 0
    data_masked = json.loads(res_masked.stdout)
    assert "local-bert" in data_masked
    assert "remote-api" in data_masked
    assert "api_key" not in data_masked["remote-api"]

    # With secrets
    res_full = runner.invoke(cli.app, ["list-models", "--json", "--show-secrets"])
    assert res_full.exit_code == 0
    data_full = json.loads(res_full.stdout)
    assert data_full["remote-api"]["api_key"] == "secret-key"
