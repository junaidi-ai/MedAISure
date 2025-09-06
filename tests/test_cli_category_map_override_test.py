from pathlib import Path
import json
import pytest
from typer.testing import CliRunner

import bench.cli_typer as cli

runner = CliRunner()


def _write_minimal_task(dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    task_path = dest_dir / "mini_task_metrics.yaml"
    yaml_text = """
name: "Minimal QA Task (Metrics)"
description: "Tiny QA task for category map override smoke test"
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
  - "rouge_l"
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
def test_cli_category_map_inline_and_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Use temp registry
    registry = tmp_path / "registry.json"
    monkeypatch.setattr(cli, "REGISTRY_FILE", registry, raising=False)

    # Task
    tasks_dir = tmp_path / "tasks"
    task_id = _write_minimal_task(tasks_dir)

    # Dummy model file and register a local model fixture
    model_file = tmp_path / "dummy.bin"
    model_file.write_bytes(b"")

    env = {
        "RICH_NO_COLOR": "1",
        "TERM": "dumb",
        "RICH_FORCE_TERMINAL": "0",
        "MEDAISURE_NO_RICH": "1",
    }

    # Register local model
    res_reg = runner.invoke(
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
    assert res_reg.exit_code == 0, res_reg.stdout

    out_dir = tmp_path / "results"

    # 1) Inline JSON category map: map diagnostics->accuracy only so combined uses accuracy
    inline_map = json.dumps({"diagnostics": ["accuracy"]})
    res_eval_inline = runner.invoke(
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
            "--combined-weights",
            "diagnostics=1.0",
            "--combined-metric-name",
            "combined_score",
            "--category-map",
            inline_map,
        ],
        env=env,
    )
    assert res_eval_inline.exit_code == 0, res_eval_inline.stdout
    files = list(out_dir.glob("*.json"))
    assert files, "no results file produced for inline map"
    payload = json.loads(files[0].read_text())
    # Ensure diagnostics category present and combined_score computed
    assert "diagnostics" in payload.get("overall_scores", {})
    assert "combined_score" in payload.get("overall_scores", {})

    # 2) File-based category map: map summarization->rouge_l only
    cat_file = tmp_path / "catmap.json"
    cat_file.write_text(json.dumps({"summarization": ["rouge_l"]}))
    out_dir2 = tmp_path / "results2"
    res_eval_file = runner.invoke(
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
            str(out_dir2),
            "--format",
            "json",
            "--combined-weights",
            "summarization=1.0",
            "--combined-metric-name",
            "combined_score",
            "--category-map-file",
            str(cat_file),
        ],
        env=env,
    )
    assert res_eval_file.exit_code == 0, res_eval_file.stdout
    files2 = list(out_dir2.glob("*.json"))
    assert files2, "no results file produced for file map"
    payload2 = json.loads(files2[0].read_text())
    assert "summarization" in payload2.get("overall_scores", {})
    assert "combined_score" in payload2.get("overall_scores", {})
