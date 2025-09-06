"""Tests for category aggregation in ResultAggregator."""

from datetime import datetime, timezone
import pytest

from bench.evaluation.result_aggregator import ResultAggregator
from bench.models.evaluation_result import EvaluationResult


def _make_eval(
    model_id: str, task_id: str, metrics: dict[str, float]
) -> EvaluationResult:
    return EvaluationResult(
        model_id=model_id,
        task_id=task_id,
        inputs=[{"i": 1}],
        model_outputs=[{"o": 1}],
        metrics_results=metrics,
        metadata={},
        timestamp=datetime.now(timezone.utc),
    )


def test_add_category_aggregates_mean_per_task_and_overall(tmp_path):
    ra = ResultAggregator(output_dir=tmp_path)
    run = "run-cat"

    # Task t1 exposes mixed metrics across categories
    ra.add_evaluation_result(
        _make_eval(
            "m",
            "t1",
            metrics={
                # diagnostics
                "accuracy": 0.8,
                "diagnostic_accuracy": 0.6,
                # summarization
                "rouge_l": 0.5,
                "clinical_relevance": 0.7,
                # communication
                "reasoning_quality": 0.9,
                # safety
                "safety": 0.4,
            },
        ),
        run_id=run,
    )

    # Task t2 exposes a different subset
    ra.add_evaluation_result(
        _make_eval(
            "m",
            "t2",
            metrics={
                # diagnostics: only accuracy
                "accuracy": 0.4,
                # summarization: only rouge_l
                "rouge_l": 0.3,
                # communication missing
                # safety missing
            },
        ),
        run_id=run,
    )

    # Aggregate into canonical categories
    ra.add_category_aggregates(run)

    report = ra.get_report(run)

    # Per-task expectations (means over present metrics)
    # t1 diagnostics mean over accuracy (0.8), diagnostic_accuracy (0.6) => 0.7
    assert report.task_scores["t1"]["diagnostics"] == pytest.approx(0.7)
    # t1 summarization mean over rouge_l (0.5), clinical_relevance (0.7) => 0.6
    assert report.task_scores["t1"]["summarization"] == pytest.approx(0.6)
    # t1 communication from reasoning_quality => 0.9
    assert report.task_scores["t1"]["communication"] == pytest.approx(0.9)
    # t1 safety from safety => 0.4
    assert report.task_scores["t1"]["safety"] == pytest.approx(0.4)

    # t2 diagnostics from accuracy => 0.4
    assert report.task_scores["t2"]["diagnostics"] == pytest.approx(0.4)
    # t2 summarization from rouge_l => 0.3
    assert report.task_scores["t2"]["summarization"] == pytest.approx(0.3)
    # t2 has no communication/safety categories
    assert "communication" not in report.task_scores["t2"]
    assert "safety" not in report.task_scores["t2"]

    # Overall category means computed across tasks where present
    # diagnostics overall: mean of [0.7 (t1), 0.4 (t2)] => 0.55
    assert report.overall_scores["diagnostics"] == pytest.approx(0.55)
    # summarization overall: mean of [0.6 (t1), 0.3 (t2)] => 0.45
    assert report.overall_scores["summarization"] == pytest.approx(0.45)
    # communication overall: only t1 present => 0.9
    assert report.overall_scores["communication"] == pytest.approx(0.9)
    # safety overall: only t1 present => 0.4
    assert report.overall_scores["safety"] == pytest.approx(0.4)
