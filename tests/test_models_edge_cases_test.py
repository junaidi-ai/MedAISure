"""Edge case tests for MedAISure data models.

Covers invalid inputs, boundary conditions, and timezone normalization.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bench.models import BenchmarkReport, EvaluationResult, MedicalTask, TaskType


class TestMedicalTaskEdgeCases:
    def test_empty_task_id(self):
        with pytest.raises(ValueError):
            MedicalTask(
                task_id=" ",
                task_type=TaskType.QA,
                inputs=[{"q": "?"}],
                metrics=["acc"],
            )

    def test_empty_inputs(self):
        with pytest.raises(ValueError):
            MedicalTask(
                task_id="t",
                task_type=TaskType.QA,
                inputs=[],
                metrics=["acc"],
            )

    def test_duplicate_metrics(self):
        with pytest.raises(ValueError):
            MedicalTask(
                task_id="t",
                task_type=TaskType.QA,
                inputs=[{"q": "?"}],
                metrics=["acc", "acc"],
            )

    def test_input_output_schema_required_lists(self):
        # invalid 'required' types
        with pytest.raises(ValueError):
            MedicalTask(
                task_id="t",
                task_type=TaskType.QA,
                inputs=[{"q": "?"}],
                metrics=["acc"],
                input_schema={"required": "q"},
            )
        with pytest.raises(ValueError):
            MedicalTask(
                task_id="t",
                task_type=TaskType.QA,
                inputs=[{"q": "?"}],
                metrics=["acc"],
                output_schema={"required": ["a", 3]},
            )


class TestEvaluationResultEdgeCases:
    def test_non_numeric_metric_value(self):
        with pytest.raises(ValueError):
            EvaluationResult(
                model_id="m",
                task_id="t",
                inputs=[{"q": "?"}],
                model_outputs=[{"a": "!"}],
                metrics_results={"acc": "nan"},
            )

    def test_mismatched_inputs_outputs_length(self):
        with pytest.raises(ValueError):
            EvaluationResult(
                model_id="m",
                task_id="t",
                inputs=[{"q": "?"}, {"q": "2"}],
                model_outputs=[{"a": "!"}],
            )

    def test_timezone_normalization(self):
        # naive datetime should become timezone-aware UTC
        naive = datetime.now(UTC).replace(tzinfo=None)
        obj = EvaluationResult(model_id="m", task_id="t", timestamp=naive)
        assert obj.timestamp.tzinfo is not None
        assert obj.timestamp.utcoffset() is not None


class TestBenchmarkReportEdgeCases:
    def test_model_id_mismatch_in_detailed_results(self):
        with pytest.raises(ValueError):
            BenchmarkReport(
                model_id="m1",
                detailed_results=[
                    EvaluationResult(
                        model_id="m2",
                        task_id="t1",
                        inputs=[{"q": "?"}],
                        model_outputs=[{"a": "!"}],
                    )
                ],
            )

    def test_validate_against_unknown_task(self):
        report = BenchmarkReport(model_id="m1")
        report.add_evaluation_result(
            EvaluationResult(
                model_id="m1", task_id="unknown", inputs=[], model_outputs=[]
            )
        )
        with pytest.raises(ValueError):
            report.validate_against_tasks(tasks={})
