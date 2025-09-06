import json

from typer.testing import CliRunner

from bench.cli_typer import app
from bench.leaderboard.submission import validate_submission

runner = CliRunner()


def test_evaluate_export_submission_monkeypatched(tmp_path, monkeypatch):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    out_file = tmp_path / "sub.json"

    # Build a dummy report to be returned by the patched EvaluationHarness
    from bench.models.benchmark_report import BenchmarkReport
    from bench.models.evaluation_result import EvaluationResult

    def _dummy_eval(self, **kwargs):
        er = EvaluationResult(
            model_id="dummy-model",
            task_id="t1",
            inputs=[{"id": "1"}],
            model_outputs=[{"prediction": "yes", "reasoning": "r1"}],
            metrics_results={"accuracy": 1.0},
        )
        br = BenchmarkReport(model_id="dummy-model")
        br.add_evaluation_result(er)
        br.metadata["run_id"] = "run-from-dummy"
        return br

    class DummyHarness:
        def __init__(self, tasks_dir: str, results_dir: str, on_progress=None):
            self.tasks_dir = tasks_dir
            self.results_dir = results_dir
            self.on_progress = on_progress

        def evaluate(self, *args, **kwargs):
            return _dummy_eval(self, **kwargs)

    # Patch the EvaluationHarness used inside the CLI module
    import bench.cli_typer as cli

    monkeypatch.setattr(cli, "EvaluationHarness", DummyHarness, raising=True)

    res = runner.invoke(
        app,
        [
            "evaluate",
            "dummy-model",
            "--tasks",
            "t1",
            "--tasks-dir",
            str(tasks_dir),
            "--model-type",
            "local",
            "--output-dir",
            str(results_dir),
            "--format",
            "json",
            "--save-results",
            "--export-submission",
            str(out_file),
            "--export-submission-include-reasoning",
        ],
        env={"MEDAISURE_NO_RICH": "1"},
    )

    assert res.exit_code == 0, res.output
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    validate_submission(data)
    assert data["run_id"] == "run-from-dummy"
    assert data["model_id"] == "dummy-model"
    items = data["submissions"][0]["items"]
    assert items and items[0]["input_id"] == "1"
    assert "reasoning" in items[0]
