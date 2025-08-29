"""Performance-oriented tests using pytest-benchmark.

Focus areas:
- Adding many EvaluationResult items to BenchmarkReport (aggregation speed)
- JSON/YAML serialization throughput of models

Note: These are smoke-level perf tests to catch regressions, not strict SLAs.
"""

from __future__ import annotations

from typing import List

import pytest

from bench.models import BenchmarkReport, EvaluationResult


@pytest.mark.benchmark(group="benchmark_report")
def test_add_many_results_benchmark(benchmark):
    report = BenchmarkReport(model_id="perf_model")

    # Prepare synthetic results
    def make_results(n: int) -> List[EvaluationResult]:
        return [
            EvaluationResult(
                model_id="perf_model",
                task_id=f"task_{i % 10}",  # recycle small set of tasks to form averages
                inputs=[{"q": str(i)}],
                model_outputs=[{"a": str(i)}],
                metrics_results={"accuracy": (i % 100) / 100.0, "f1": (i % 80) / 80.0},
            )
            for i in range(n)
        ]

    results = make_results(1000)

    def run():
        for r in results:
            report.add_evaluation_result(r)
        return report

    out = benchmark(run)
    # Basic sanity: overall scores should be present
    assert "accuracy" in out.overall_scores
    assert "f1" in out.overall_scores


@pytest.mark.benchmark(group="serialization")
def test_serialization_throughput_json(benchmark):
    report = BenchmarkReport(
        model_id="m1",
        detailed_results=[
            EvaluationResult(
                model_id="m1",
                task_id=f"t{i}",
                inputs=[{"q": str(i)}],
                model_outputs=[{"a": str(i)}],
                metrics_results={"accuracy": 0.5, "f1": 0.5},
            )
            for i in range(50)
        ],
    )

    json_text = benchmark(report.to_json)
    assert isinstance(json_text, str)


@pytest.mark.benchmark(group="serialization")
def test_serialization_throughput_yaml(benchmark):
    report = BenchmarkReport(
        model_id="m1",
        detailed_results=[
            EvaluationResult(
                model_id="m1",
                task_id=f"t{i}",
                inputs=[{"q": str(i)}],
                model_outputs=[{"a": str(i)}],
                metrics_results={"accuracy": 0.5, "f1": 0.5},
            )
            for i in range(50)
        ],
    )

    yaml_text = benchmark(report.to_yaml)
    assert isinstance(yaml_text, str)
