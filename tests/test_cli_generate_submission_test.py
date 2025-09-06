import json
from pathlib import Path

from typer.testing import CliRunner

from bench.cli_typer import app
from bench.evaluation.result_aggregator import ResultAggregator
from bench.models.evaluation_result import EvaluationResult
from bench.leaderboard.submission import validate_submission


runner = CliRunner()


def _make_and_save_report(dirpath: Path, run_id: str, filename: str | None = None):
    ra = ResultAggregator(output_dir=str(dirpath))
    er = EvaluationResult(
        model_id="unit-model",
        task_id="task-1",
        inputs=[{"id": "i1"}, {"id": "i2"}],
        model_outputs=[{"y": 1, "reasoning": "because"}, {"y": 2}],
        metrics_results={"accuracy": 1.0},
    )
    ra.add_evaluation_result(er, run_id=run_id)
    report = ra.get_report(run_id)
    # Save using either <run_id>.json or a custom file name
    if filename is None:
        out = dirpath / f"{run_id}.json"
    else:
        out = dirpath / filename
    report.save(out)
    return out


def test_generate_submission_by_exact_filename(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run_id = "r-1234"
    _make_and_save_report(results_dir, run_id)  # writes r-1234.json

    out_file = tmp_path / "submission.json"
    res = runner.invoke(
        app,
        [
            "generate-submission",
            "--run-id",
            run_id,
            "--results-dir",
            str(results_dir),
            "--out",
            str(out_file),
            "--include-reasoning",
        ],
        env={"MEDAISURE_NO_RICH": "1"},
    )
    assert res.exit_code == 0, res.output
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    validate_submission(data)
    # Reasoning should be included for first item
    items = data["submissions"][0]["items"]
    assert "reasoning" in items[0]


def test_generate_submission_by_scan_fallback_and_no_reasoning(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run_id = "r-9876"
    # Save with a non-matching file name to force scan fallback
    _make_and_save_report(results_dir, run_id, filename="custom_name.json")

    out_file = tmp_path / "submission2.json"
    res = runner.invoke(
        app,
        [
            "generate-submission",
            "--run-id",
            run_id,
            "--results-dir",
            str(results_dir),
            "--out",
            str(out_file),
            "--no-include-reasoning",
        ],
        env={"MEDAISURE_NO_RICH": "1"},
    )
    assert res.exit_code == 0, res.output
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    validate_submission(data)
    # Reasoning should not be present when disabled
    items = data["submissions"][0]["items"]
    assert "reasoning" not in items[0]
