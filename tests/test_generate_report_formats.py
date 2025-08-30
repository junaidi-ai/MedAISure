import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import bench.cli_typer as cli

runner = CliRunner()


@pytest.fixture()
def sample_report_json(tmp_path: Path) -> Path:
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


def test_generate_report_html(tmp_path: Path, sample_report_json: Path):
    out = tmp_path / "out.html"
    result = runner.invoke(
        cli.app,
        [
            "generate-report",
            str(sample_report_json),
            "--format",
            "html",
            "--output-file",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists()
    content = out.read_text()
    assert "<html" in content and "MedAISure Benchmark Report" in content


def test_generate_report_pdf_if_available(tmp_path: Path, sample_report_json: Path):
    out = tmp_path / "out.pdf"
    result = runner.invoke(
        cli.app,
        [
            "generate-report",
            str(sample_report_json),
            "--format",
            "pdf",
            "--output-file",
            str(out),
        ],
    )
    if result.exit_code != 0:
        # Expect a clear message when weasyprint is missing
        assert "requires 'weasyprint'" in result.stdout
        return
    # weasyprint installed; verify file exists and non-empty
    assert out.exists()
    assert out.stat().st_size > 0
