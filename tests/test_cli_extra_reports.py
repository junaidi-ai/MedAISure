from pathlib import Path

from typer.testing import CliRunner
import bench.cli_typer as cli
from bench.models.benchmark_report import BenchmarkReport

runner = CliRunner()


def test_cli_evaluate_exports_extra_reports(tmp_path, monkeypatch):
    # Prepare a minimal report
    report = BenchmarkReport(
        model_id="cli-model",
        overall_scores={"accuracy": 0.5},
        task_scores={"t1": {"accuracy": 0.5}},
        detailed_results=[],
        metadata={"run_id": "abcd1234"},
    )

    # Monkeypatch EvaluationHarness.evaluate to avoid running models
    def fake_eval(self, *args, **kwargs):  # noqa: ANN001, ANN002
        # Ensure the harness writes the base json when save_results is True
        save_results = kwargs.get("save_results", True)
        results_dir = Path(getattr(self, "results_dir"))
        results_dir.mkdir(parents=True, exist_ok=True)
        if save_results:
            (results_dir / f"{report.metadata['run_id']}.json").write_text(
                report.to_json()
            )
        return report

    monkeypatch.setattr(cli.EvaluationHarness, "evaluate", fake_eval, raising=True)

    out_dir = tmp_path / "results"
    rep_dir = tmp_path / "reports"

    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "model-x",
            "--tasks",
            "dummy-task",
            "--output-dir",
            str(out_dir),
            "--format",
            "json",
            "--extra-report",
            "html",
            "--extra-report",
            "md",
            "--report-dir",
            str(rep_dir),
            "--tasks-dir",
            str(tmp_path),  # empty dir is fine; evaluate is mocked
        ],
        env={"MEDAISURE_NO_RICH": "1"},
    )

    assert result.exit_code == 0, result.stdout

    # Verify extra reports were written
    html_file = rep_dir / "abcd1234.html"
    md_file = rep_dir / "abcd1234.md"
    assert html_file.exists() and md_file.exists()
    assert "<html" in html_file.read_text().lower()
    assert "cli-model" in md_file.read_text()
