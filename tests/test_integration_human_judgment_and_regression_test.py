"""Integration-like tests for human judgment comparison and regression diffs.

These tests operate at the ResultAggregator level to avoid heavy model dependencies
while still exercising end-to-end aggregation and run comparison logic.
"""

from datetime import datetime, timezone

import pytest

from bench.evaluation.result_aggregator import ResultAggregator
from bench.models.evaluation_result import EvaluationResult


@pytest.fixture
def agg(tmp_path):
    return ResultAggregator(output_dir=tmp_path)


def _mk_result(task_id: str, metrics: dict[str, float]) -> EvaluationResult:
    ts = datetime.now(timezone.utc)
    return EvaluationResult(
        model_id="model-x",
        task_id=task_id,
        inputs=[{"i": 1}],
        model_outputs=[{"o": 1}],
        metrics_results=metrics,
        metadata={},
        timestamp=ts,
    )


def test_human_judgment_comparison_via_compare_runs(agg: ResultAggregator):
    # Two runs with a custom 'human_judgment' metric (1-5 scale)
    # Run A (baseline)
    run_a = "human_judgment_run_A"
    agg.add_evaluation_result(
        _mk_result("task_consult_1", {"human_judgment": 4.0}), run_a
    )
    agg.add_evaluation_result(
        _mk_result("task_consult_2", {"human_judgment": 3.0}), run_a
    )

    # Run B (candidate) with improvements
    run_b = "human_judgment_run_B"
    agg.add_evaluation_result(
        _mk_result("task_consult_1", {"human_judgment": 4.8}), run_b
    )
    agg.add_evaluation_result(
        _mk_result("task_consult_2", {"human_judgment": 3.2}), run_b
    )

    # Compare absolute differences for the custom metric
    diff = agg.compare_runs(run_a, run_b, metrics=["human_judgment"], relative=False)

    # Overall: mean_A = (4.0+3.0)/2 = 3.5; mean_B = (4.8+3.2)/2 = 4.0; diff = 0.5
    assert diff["overall"]["human_judgment"] == pytest.approx(0.5)

    # Per-task diffs
    assert diff["per_task"]["task_consult_1"]["human_judgment"] == pytest.approx(0.8)
    assert diff["per_task"]["task_consult_2"]["human_judgment"] == pytest.approx(0.2)


def test_regression_detection_relative_diff(agg: ResultAggregator):
    # Baseline vs current run with accuracy changes to test relative diffs
    run_a = "baseline_run"
    agg.add_evaluation_result(_mk_result("t1", {"accuracy": 0.80}), run_a)
    agg.add_evaluation_result(_mk_result("t2", {"accuracy": 0.90}), run_a)

    run_b = "current_run"
    agg.add_evaluation_result(_mk_result("t1", {"accuracy": 0.88}), run_b)  # +0.08
    agg.add_evaluation_result(_mk_result("t2", {"accuracy": 0.81}), run_b)  # -0.09

    # Relative diff = (b - a) / (|a| + eps)
    diff = agg.compare_runs(run_a, run_b, metrics=["accuracy"], relative=True)

    # Overall baseline mean = 0.85, current mean = 0.845 -> absolute -0.005
    # Relative overall ~ (-0.005 / 0.85) ≈ -0.005882...
    assert diff["overall"]["accuracy"] == pytest.approx(-0.005882, abs=1e-6)

    # Per-task relative diffs
    # t1: (0.88 - 0.80) / 0.80 = 0.10
    assert diff["per_task"]["t1"]["accuracy"] == pytest.approx(0.10, abs=1e-9)
    # t2: (0.81 - 0.90) / 0.90 ≈ -0.10
    assert diff["per_task"]["t2"]["accuracy"] == pytest.approx(-0.10, abs=1e-9)
