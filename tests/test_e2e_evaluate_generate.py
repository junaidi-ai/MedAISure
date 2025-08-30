from pathlib import Path

import pytest
from typer.testing import CliRunner

import bench.cli_typer as cli

runner = CliRunner()


def _write_minimal_task(dest_dir: Path) -> str:
    """Create a minimal valid QA task with inputs/expected_outputs.

    Returns the task_id (file stem).
    """
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
    return task_path.stem  # "mini_task"


@pytest.mark.integration
def test_e2e_evaluate_then_generate_report_html(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Use a temp registry so we don't touch the repo
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(cli, "REGISTRY_FILE", registry, raising=False)

    # Prepare a tiny tasks dir with one task
    tasks_dir = tmp_path / "tasks"
    task_id = _write_minimal_task(tasks_dir)

    # Create a dummy model path file and register a simple local model
    model_file = tmp_path / "dummy.bin"
    model_file.write_bytes(b"")

    env = {
        "RICH_NO_COLOR": "1",
        "TERM": "dumb",
        "RICH_FORCE_TERMINAL": "0",
        "MEDAISURE_NO_RICH": "1",
    }

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

    # Run evaluation on the single task, output JSON to an output dir
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

    # Find the produced results file (*.json) in out_dir
    files = list(out_dir.glob("*.json"))
    assert files, "no results file produced"
    results_file = files[0]

    # Generate an HTML report from the results file
    html_out = tmp_path / "report.html"
    gen_res = runner.invoke(
        cli.app,
        [
            "generate-report",
            str(results_file),
            "--format",
            "html",
            "--output-file",
            str(html_out),
        ],
        env=env,
    )
    assert gen_res.exit_code == 0, gen_res.stdout
    assert html_out.exists()
    content = html_out.read_text()
    assert "<html" in content and "MedAISure Benchmark Report" in content
