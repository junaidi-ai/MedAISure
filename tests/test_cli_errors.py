from pathlib import Path

from typer.testing import CliRunner

from bench.cli_typer import app


runner = CliRunner()


def test_generate_report_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.json"
    result = runner.invoke(app, ["generate-report", str(missing), "--format", "md"])
    # Typer Exit with code 1 due to _ensure_exists
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower()


def test_list_tasks_missing_dir(tmp_path: Path):
    missing_dir = tmp_path / "tasks_dir"
    result = runner.invoke(app, ["list-tasks", "--tasks-dir", str(missing_dir)])
    assert result.exit_code != 0
    assert "tasks directory" in result.stdout.lower()


def test_register_model_local_missing_module(tmp_path: Path):
    # Create a dummy model file to satisfy path check
    model_path = tmp_path / "dummy-model.bin"
    model_path.write_text("dummy")
    # Missing module_path should trigger a BadParameter
    result = runner.invoke(
        app,
        [
            "register-model",
            str(model_path),
            "--model-type",
            "local",
            # no --module-path
        ],
    )
    assert result.exit_code != 0
    assert "module_path is required" in result.stdout.lower()


def test_register_model_api_missing_fields():
    # Missing endpoint and api_key
    result = runner.invoke(app, ["register-model", "--model-type", "api"])
    assert result.exit_code != 0
    assert (
        "endpoint is required" in result.stdout.lower()
        or "api_key is required" in result.stdout.lower()
    )
