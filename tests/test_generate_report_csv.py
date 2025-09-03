from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import bench.cli_typer as cli

runner = CliRunner()


def _sample_report_json(tmp_path: Path) -> Path:
    data = {
        "schema_version": 1,
        "model_id": "test-model",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "overall_scores": {"accuracy": 0.9, "f1": 0.8},
        "task_scores": {
            "task_a": {"accuracy": 0.92, "f1": 0.82},
            "task_b": {"accuracy": 0.88, "f1": 0.78},
        },
        "detailed_results": [],
        "metadata": {"run_id": "run-123"},
    }
    p = tmp_path / "report.json"
    p.write_text(json.dumps(data))
    return p


def test_generate_report_csv_single_file(tmp_path: Path) -> None:
    src = _sample_report_json(tmp_path)
    out_csv = tmp_path / "out.csv"
    result = runner.invoke(
        cli.app,
        [
            "generate-report",
            str(src),
            "--format",
            "csv",
            "--output-file",
            str(out_csv),
        ],
        env={"MEDAISURE_NO_RICH": "1"},
    )
    assert result.exit_code == 0, result.stdout
    assert out_csv.exists()
    head = out_csv.read_text().splitlines()[0]
    assert head.strip().split(",") == ["task_id", "metric", "score"]


def test_generate_report_csv_directory(tmp_path: Path) -> None:
    src = _sample_report_json(tmp_path)
    out_dir = tmp_path / "csv_dir"
    result = runner.invoke(
        cli.app,
        [
            "generate-report",
            str(src),
            "--format",
            "csv",
            "--output-file",
            str(out_dir),
        ],
        env={"MEDAISURE_NO_RICH": "1"},
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "overall_scores.csv").exists()
    assert (out_dir / "task_scores.csv").exists()
    assert (out_dir / "detailed_metrics.csv").exists()
    assert (out_dir / "detailed_inputs.csv").exists()
    assert (out_dir / "detailed_outputs.csv").exists()
