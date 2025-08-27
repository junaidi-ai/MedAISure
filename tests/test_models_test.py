"""Unit tests for MedAISure benchmark data models."""

from datetime import datetime, timezone

import pytest

from bench.models import BenchmarkReport, EvaluationResult, MedicalTask, TaskType


class TestTaskType:
    """Tests for the TaskType enum."""

    def test_task_type_values(self):
        """Test that TaskType has the expected values."""
        assert TaskType.DIAGNOSTIC_REASONING == "diagnostic_reasoning"
        assert TaskType.QA == "qa"
        assert TaskType.SUMMARIZATION == "summarization"
        assert TaskType.COMMUNICATION == "communication"


class TestMedicalTask:
    """Tests for the MedicalTask model."""

    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing."""
        return {
            "task_id": "test_task_1",
            "name": "Test Medical QA Task",
            "task_type": "qa",
            "description": "Test medical QA task",
            "inputs": [{"question": "What is the treatment for a cold?"}],
            "expected_outputs": [{"answer": "Rest and hydration"}],
            "metrics": ["accuracy", "f1_score"],
            "input_schema": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
            },
            "output_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
            "dataset": [
                {
                    "question": "What is the treatment for a cold?",
                    "answer": "Rest and hydration",
                }
            ],
        }

    def test_create_medical_task(self, sample_task_data):
        """Test creating a MedicalTask instance."""
        task = MedicalTask(**sample_task_data)
        assert task.task_id == "test_task_1"
        assert task.task_type == TaskType.QA
        assert len(task.inputs) == 1
        assert len(task.expected_outputs) == 1
        assert len(task.metrics) == 2

    def test_task_validation(self, sample_task_data):
        """Test validation of task data."""
        # Test invalid task type
        with pytest.raises(ValueError):
            data = sample_task_data.copy()
            data["task_type"] = "invalid_type"
            MedicalTask(**data)

        # Test empty inputs
        with pytest.raises(ValueError):
            data = sample_task_data.copy()
            data["inputs"] = []
            MedicalTask(**data)

        # Test empty metrics
        with pytest.raises(ValueError):
            data = sample_task_data.copy()
            data["metrics"] = [""]
            MedicalTask(**data)

    def test_serialization(self, sample_task_data):
        """Test JSON serialization/deserialization."""
        task = MedicalTask.model_validate(sample_task_data)
        json_str = task.model_dump_json()
        loaded_task = MedicalTask.model_validate_json(json_str)
        # Compare dict representations to avoid float precision issues
        assert loaded_task.model_dump() == task.model_dump()


class TestEvaluationResult:
    """Tests for the EvaluationResult model."""

    @pytest.fixture
    def sample_eval_data(self):
        """Sample evaluation data for testing."""
        return {
            "model_id": "test_model_1",
            "task_id": "test_task_1",
            "inputs": [{"question": "What is the treatment for a cold?"}],
            "model_outputs": [{"answer": "Rest and hydration"}],
            "metrics_results": {"accuracy": 0.9, "f1_score": 0.85},
            "metadata": {"model_version": "1.0.0"},
        }

    def test_create_evaluation_result(self, sample_eval_data):
        """Test creating an EvaluationResult instance."""
        result = EvaluationResult(**sample_eval_data)
        assert result.model_id == "test_model_1"
        assert result.task_id == "test_task_1"
        assert len(result.inputs) == 1
        assert len(result.model_outputs) == 1
        assert "accuracy" in result.metrics_results
        assert result.metadata["model_version"] == "1.0.0"
        assert isinstance(result.timestamp, datetime)

    def test_validation(self, sample_eval_data):
        """Test validation of evaluation results."""
        # Test mismatched input/output lengths
        with pytest.raises(ValueError):
            data = sample_eval_data.copy()
            data["inputs"].append({"question": "Another question?"})
            EvaluationResult(**data)

        # Test invalid metric values
        with pytest.raises(ValueError):
            data = sample_eval_data.copy()
            data["metrics_results"]["invalid"] = "not a number"
            EvaluationResult(**data)

    def test_serialization(self, sample_eval_data):
        """Test JSON serialization/deserialization."""
        result = EvaluationResult.model_validate(sample_eval_data)
        json_str = result.model_dump_json()
        loaded_result = EvaluationResult.model_validate_json(json_str)
        # Compare dict representations to avoid float precision issues
        assert loaded_result.model_dump() == result.model_dump()


class TestBenchmarkReport:
    """Tests for the BenchmarkReport model."""

    @pytest.fixture
    def sample_report_data(self):
        """Sample report data for testing."""
        return {
            "model_id": "test_model_1",
            "timestamp": datetime.now(timezone.utc),
            "overall_scores": {"accuracy": 0.9, "f1_score": 0.85},
            "task_scores": {
                "task_1": {"accuracy": 0.9, "f1_score": 0.85},
                "task_2": {"accuracy": 0.8, "f1_score": 0.75},
            },
            "detailed_results": [
                {
                    "model_id": "test_model_1",
                    "task_id": "task_1",
                    "inputs": [{"question": "Q1?"}],
                    "model_outputs": [{"answer": "A1"}],
                    "metrics_results": {"accuracy": 0.9, "f1_score": 0.85},
                },
                {
                    "model_id": "test_model_1",
                    "task_id": "task_2",
                    "inputs": [{"question": "Q2?"}],
                    "model_outputs": [{"answer": "A2"}],
                    "metrics_results": {"accuracy": 0.8, "f1_score": 0.75},
                },
            ],
            "metadata": {"run_id": "test_run_1"},
        }

    def test_create_benchmark_report(self, sample_report_data):
        """Test creating a BenchmarkReport instance."""
        report = BenchmarkReport(**sample_report_data)
        assert report.model_id == "test_model_1"
        assert len(report.task_scores) == 2
        assert len(report.detailed_results) == 2
        assert report.overall_scores["accuracy"] == 0.9
        assert report.metadata["run_id"] == "test_run_1"

    def test_add_evaluation_result(self, sample_report_data):
        """Test adding evaluation results to a report."""
        report = BenchmarkReport(
            model_id="test_model_1",
            timestamp=datetime.now(timezone.utc),
            overall_scores={},
            task_scores={},
            detailed_results=[],
            metadata={"run_id": "test_run_1"},
        )

        # Add first evaluation result
        result1 = EvaluationResult(
            model_id="test_model_1",
            task_id="task_1",
            inputs=[{"question": "Q1?"}],
            model_outputs=[{"answer": "A1"}],
            metrics_results={"accuracy": 0.9, "f1_score": 0.85},
        )
        report.add_evaluation_result(result1)

        # Add second evaluation result for a different task
        result2 = EvaluationResult(
            model_id="test_model_1",
            task_id="task_2",
            inputs=[{"question": "Q2?"}],
            model_outputs=[{"answer": "A2"}],
            metrics_results={"accuracy": 0.8, "f1_score": 0.75},
        )
        report.add_evaluation_result(result2)

        # Verify the report was updated correctly
        assert len(report.detailed_results) == 2
        assert len(report.task_scores) == 2
        assert "accuracy" in report.overall_scores

        # Use pytest.approx for float comparison to handle floating point precision
        assert report.overall_scores["accuracy"] == pytest.approx(0.85)  # Average of 0.9 and 0.8

    def test_serialization(self, sample_report_data):
        """Test JSON serialization/deserialization."""
        report = BenchmarkReport.model_validate(sample_report_data)
        json_str = report.model_dump_json()
        loaded_report = BenchmarkReport.model_validate_json(json_str)
        # Compare dict representations to avoid float precision issues
        assert loaded_report.model_dump(exclude={"timestamp"}) == report.model_dump(exclude={"timestamp"})
