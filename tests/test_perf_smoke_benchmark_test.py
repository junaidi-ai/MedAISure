from pathlib import Path
from typing import Any, Dict, List

import pytest

from bench.evaluation.harness import EvaluationHarness
from bench.evaluation.result_aggregator import ResultAggregator
from bench.models.evaluation_result import EvaluationResult

# Skip whole module if pytest-benchmark is not installed (after imports for E402 compliance)
pytest.importorskip("pytest_benchmark")


class _FastRunner:
    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}

    def load_model(
        self, model_name: str, model_type: str = "local", **kwargs: Any
    ) -> None:
        self._models[model_name] = object()

    def run_model(
        self, model_id: str, inputs: List[Dict[str, Any]], batch_size: int = 8
    ):
        # Return trivial predictions quickly
        return [{"label": inp.get("label", 0)} for inp in inputs]

    def unload_model(self, model_id: str) -> None:
        self._models.pop(model_id, None)


def _write_synthetic_task(tmp_path: Path, n: int = 50) -> str:
    task_id = "synthetic_acc"
    task_file = tmp_path / f"{task_id}.yaml"
    dataset = [{"text": f"x{i}", "label": i % 2} for i in range(n)]
    content = {
        "task_id": task_id,
        "name": "Synthetic Accuracy",
        "description": "Tiny task for perf smoke",
        "task_type": "qa",
        # Minimal, non-empty inputs to satisfy model validation
        "inputs": [{"text": "example", "label": 0}],
        "expected_outputs": [],
        # Metrics must be a list of strings after normalization
        "metrics": ["accuracy"],
        "dataset": dataset,
    }
    import yaml

    task_file.write_text(yaml.safe_dump(content))
    return task_id


def test_benchmark_harness_evaluate_smoke(tmp_path: Path, benchmark):
    tasks_dir = tmp_path / "tasks"
    results_dir = tmp_path / "results"
    tasks_dir.mkdir()
    results_dir.mkdir()

    task_id = _write_synthetic_task(tasks_dir, n=64)

    h = EvaluationHarness(tasks_dir=str(tasks_dir), results_dir=str(results_dir))
    # Inject fast runner to avoid heavyweight model work
    h.model_runner = _FastRunner()  # type: ignore[assignment]

    def _run():
        return h.evaluate(
            model_id="fast-model",
            task_ids=[task_id],
            model_type="local",
            batch_size=16,
            use_cache=False,
            save_results=False,
        )

    report = benchmark(_run)
    assert report is not None
    assert report.task_scores


def _make_agg_with_small_report(tmp_path: Path) -> tuple[ResultAggregator, str]:
    agg = ResultAggregator(output_dir=tmp_path)
    er = EvaluationResult(
        model_id="m1",
        task_id="t1",
        inputs=[{"x": 1}],
        model_outputs=[{"y": 1}],
        metrics_results={"accuracy": 1.0},
    )
    run_id = "r1"
    agg.add_evaluation_result(er, run_id=run_id)
    return agg, run_id


def test_benchmark_export_json(tmp_path: Path, benchmark):
    agg, run_id = _make_agg_with_small_report(tmp_path)
    json_path = tmp_path / "rep.json"
    _ = agg.get_report(run_id)
    benchmark(lambda: agg.export_report_json(run_id, json_path))
    assert json_path.exists()


def test_benchmark_export_csv(tmp_path: Path, benchmark):
    agg, run_id = _make_agg_with_small_report(tmp_path)
    csv_path = tmp_path / "rep.csv"
    _ = agg.get_report(run_id)
    benchmark(lambda: agg.export_report_csv(run_id, csv_path))
    assert csv_path.exists()


def test_benchmark_export_markdown(tmp_path: Path, benchmark):
    agg, run_id = _make_agg_with_small_report(tmp_path)
    md_path = tmp_path / "rep.md"
    _ = agg.get_report(run_id)
    benchmark(lambda: agg.export_report_markdown(run_id, md_path))
    assert md_path.exists()


def test_benchmark_export_html(tmp_path: Path, benchmark):
    agg, run_id = _make_agg_with_small_report(tmp_path)
    html_path = tmp_path / "rep.html"
    _ = agg.get_report(run_id)
    benchmark(lambda: agg.export_report_html(run_id, html_path))
    assert html_path.exists()
