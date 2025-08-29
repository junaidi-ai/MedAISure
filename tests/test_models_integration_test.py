"""Integration tests covering relationships between models.

Validates end-to-end flows:
- MedicalTask schema + dataset consistency
- EvaluationResult.validate_against_task
- BenchmarkReport.validate_against_tasks
- Aggregation correctness with mixed tasks
"""

from __future__ import annotations

from typing import Dict

from bench.models import BenchmarkReport, EvaluationResult, MedicalTask, TaskType


def build_task(task_id: str) -> MedicalTask:
    return MedicalTask(
        task_id=task_id,
        name=f"Task {task_id}",
        task_type=TaskType.QA,
        description="demo",
        inputs=[{"question": "Q?"}],
        expected_outputs=[{"answer": "A!"}],
        metrics=["accuracy", "f1"],
        input_schema={"required": ["question"]},
        output_schema={"required": ["answer"]},
    )


def test_result_validates_against_task_and_report_validation():
    # Create two tasks
    t1 = build_task("t1")
    t2 = build_task("t2")
    tasks: Dict[str, MedicalTask] = {t1.task_id: t1, t2.task_id: t2}

    # Create results that comply with schemas and metrics
    r1 = EvaluationResult(
        model_id="m1",
        task_id="t1",
        inputs=[{"question": "Q1"}],
        model_outputs=[{"answer": "A1"}],
        metrics_results={"accuracy": 1.0, "f1": 0.5},
    )
    r2 = EvaluationResult(
        model_id="m1",
        task_id="t2",
        inputs=[{"question": "Q2"}],
        model_outputs=[{"answer": "A2"}],
        metrics_results={"accuracy": 0.0, "f1": 0.5},
    )

    # Individual validation
    r1.validate_against_task(t1)
    r2.validate_against_task(t2)

    # Report-level aggregation and validation
    report = BenchmarkReport(model_id="m1")
    report.add_evaluation_result(r1)
    report.add_evaluation_result(r2)
    report.validate_against_tasks(tasks)

    # Check overall scores average across tasks for each metric
    assert report.overall_scores["accuracy"] == 0.5
    assert report.overall_scores["f1"] == 0.5


def test_report_validation_catches_unknown_metrics():
    t = build_task("t1")
    r = EvaluationResult(
        model_id="m1",
        task_id="t1",
        inputs=[{"question": "Q"}],
        model_outputs=[{"answer": "A"}],
        metrics_results={"unknown_metric": 1.0},
    )

    # Direct relation check should fail because metric not defined in task
    try:
        r.validate_against_task(t)
        assert False, "Expected ValueError for unknown metrics"
    except ValueError:
        pass
