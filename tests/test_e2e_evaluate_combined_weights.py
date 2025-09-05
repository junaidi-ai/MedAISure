from pathlib import Path

import json
import pytest
from typer.testing import CliRunner

import bench.cli_typer as cli

runner = CliRunner()


def _write_minimal_task_with_accuracy(dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    task_path = dest_dir / "mini_task_acc.yaml"
    yaml_text = """
name: "Minimal QA Task (Accuracy)"
description: "Tiny QA task for E2E combined score"
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
def test_e2e_evaluate_with_combined_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Use a temp registry
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(cli, "REGISTRY_FILE", registry, raising=False)

    # Task
    tasks_dir = tmp_path / "tasks"
    task_id = _write_minimal_task_with_accuracy(tasks_dir)

    # Dummy model and register local model fixture
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

    out_dir = tmp_path / "results"
    weights = (
        "accuracy=1.0"  # ensures combined_score is computable for the minimal task
    )

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
            "--combined-weights",
            weights,
            "--combined-metric-name",
            "combined_score",
        ],
        env=env,
    )
    assert eval_res.exit_code == 0, eval_res.stdout

    files = list(out_dir.glob("*.json"))
    assert files, "no results file produced"
    results_file = files[0]
    payload = json.loads(results_file.read_text())

    # Presence checks
    assert "combined_score" in payload.get("overall_scores", {})
    assert any(
        ("combined_score" in metrics)
        for metrics in payload.get("task_scores", {}).values()
    )
