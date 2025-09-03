from pathlib import Path

import pytest
from typer.testing import CliRunner

import bench.cli_typer as cli

runner = CliRunner()


def _write_minimal_task(dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    task_path = dest_dir / "mini_task.yaml"
    yaml_text = """
name: "Minimal QA Task"
description: "Tiny QA task for E2E smoke test"
task_type: "qa"
input_schema:
  type: "object"
  required: ["question"]
  properties:
    question:
      type: "string"
output_schema:
  type: "object"
  required: ["answer"]
  properties:
    answer:
      type: "string"
metrics:
  - "accuracy"
inputs:
  - question: "What is BP?"
  - question: "What is HR?"
expected_outputs:
  - answer: "blood pressure"
  - answer: "heart rate"
dataset:
  - input:
      question: "What is BP?"
    output:
      answer: "blood pressure"
  - input:
      question: "What is HR?"
    output:
      answer: "heart rate"
"""
    task_path.write_text(yaml_text.strip() + "\n")
    return task_path.stem


@pytest.mark.integration
def test_e2e_evaluate_then_generate_report_md_and_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Use an isolated registry file
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(cli, "REGISTRY_FILE", registry, raising=False)

    # Prepare one tiny task
    tasks_dir = tmp_path / "tasks"
    task_id = _write_minimal_task(tasks_dir)

    # Dummy model file (local loader is used)
    model_file = tmp_path / "dummy.bin"
    model_file.write_bytes(b"")

    env = {
        "RICH_NO_COLOR": "1",
        "TERM": "dumb",
        "RICH_FORCE_TERMINAL": "0",
        "MEDAISURE_NO_RICH": "1",
    }

    # Register a simple local model fixture
    reg_res = runner.invoke(
        cli.app,
        [
            "register-model",
            str(model_file),
            "--model-type",
            "local",
            "--module-path",
            "tests.fixtures.simple_local_model",
            "--load-func",
            "load_model",
            "--model-id",
            "test-local",
        ],
        env=env,
    )
    assert reg_res.exit_code == 0, reg_res.stdout

    # Evaluate and save JSON results
    out_dir = tmp_path / "results"
    eval_res = runner.invoke(
        cli.app,
        [
            "evaluate",
            "test-local",
            "--tasks",
            task_id,
            "--tasks-dir",
            str(tasks_dir),
            "--model-type",
            "local",
            "--output-dir",
            str(out_dir),
            "--format",
            "json",
            "--batch-size",
            "2",
            "--save-results",
        ],
        env=env,
    )
    assert eval_res.exit_code == 0, eval_res.stdout

    # Find results file
    files = list(out_dir.glob("*.json"))
    assert files, "no results file produced"
    results_file = files[0]

    # Generate Markdown
    md_out = tmp_path / "report.md"
    md_res = runner.invoke(
        cli.app,
        [
            "generate-report",
            str(results_file),
            "--format",
            "md",
            "--output-file",
            str(md_out),
        ],
        env=env,
    )
    assert md_res.exit_code == 0, md_res.stdout
    assert md_out.exists() and "#" in md_out.read_text()

    # Generate CSV (single-file)
    csv_out = tmp_path / "report.csv"
    csv_res = runner.invoke(
        cli.app,
        [
            "generate-report",
            str(results_file),
            "--format",
            "csv",
            "--output-file",
            str(csv_out),
        ],
        env=env,
    )
    assert csv_res.exit_code == 0, csv_res.stdout
    assert csv_out.exists()
    head = csv_out.read_text().splitlines()[0]
    assert head.strip().split(",") == ["task_id", "metric", "score"]
