"""Tests for the ResultAggregator component."""

import json
from datetime import datetime, timezone

import pytest

from bench.evaluation.result_aggregator import ResultAggregator
from bench.models.evaluation_result import EvaluationResult


@pytest.fixture
def result_aggregator(tmp_path):
    """Create a ResultAggregator instance for testing."""
    return ResultAggregator(output_dir=tmp_path)


@pytest.fixture
def sample_evaluation_results():
    """Create sample evaluation results for testing."""
    timestamp = datetime.now(timezone.utc)
    return [
        EvaluationResult(
            model_id="test_model",
            task_id="task1",
            inputs=[{"input": "test input 1"}],
            model_outputs=[{"output": "test output 1"}],
            metrics_results={"accuracy": 0.9, "f1": 0.85},
            metadata={"num_examples": 100},
            timestamp=timestamp,
        ),
        EvaluationResult(
            model_id="test_model",
            task_id="task2",
            inputs=[{"input": "test input 2"}],
            model_outputs=[{"output": "test output 2"}],
            metrics_results={"accuracy": 0.8, "f1": 0.75},
            metadata={"num_examples": 200},
            timestamp=timestamp,
        ),
    ]


def test_add_result(result_aggregator):
    """Test adding evaluation results to the aggregator."""
    run_id = "test_run"
    result = EvaluationResult(
        model_id="test_model",
        task_id="test_task",
        inputs=[{"input": "test input"}],
        model_outputs=[{"output": "test output"}],
        metrics_results={"accuracy": 0.9},
        metadata={"num_examples": 100},
    )

    result_aggregator.add_evaluation_result(result, run_id)
    report = result_aggregator.get_report(run_id)

    assert report is not None
    assert report.model_id == "test_model"
    assert "test_task" in report.task_scores
    assert report.task_scores["test_task"]["accuracy"] == 0.9


def test_generate_report(result_aggregator, sample_evaluation_results):
    """Test generating a benchmark report."""
    run_id = "test_run"

    # Add multiple evaluation results
    for result in sample_evaluation_results:
        result_aggregator.add_evaluation_result(result, run_id)

    # Generate and get report
    report = result_aggregator.get_report(run_id)

    # Verify report structure
    assert report is not None
    assert report.model_id == "test_model"
    assert len(report.task_scores) == 2
    assert "task1" in report.task_scores
    assert "task2" in report.task_scores

    # Verify metrics aggregation (simple average across tasks)
    assert report.overall_scores["accuracy"] == pytest.approx(0.85)  # (0.9 + 0.8) / 2
    assert report.overall_scores["f1"] == pytest.approx(0.8)  # (0.85 + 0.75) / 2

    # Verify detailed results
    assert len(report.detailed_results) == 2


def test_save_report(result_aggregator, sample_evaluation_results, tmp_path):
    """Test saving a benchmark report to disk."""
    run_id = "test_run"

    # Add some results
    for result in sample_evaluation_results:
        result_aggregator.add_evaluation_result(result, run_id)

    # Get the report and save it
    report = result_aggregator.get_report(run_id)
    output_file = tmp_path / "test_report.json"
    result_aggregator.save_report(report, output_file)

    # Verify file was created
    assert output_file.exists()

    # Load and verify the saved data
    with open(output_file, "r") as f:
        data = json.load(f)

    assert data["model_id"] == "test_model"
    assert len(data["task_scores"]) == 2
    assert "task1" in data["task_scores"]
    assert "task2" in data["task_scores"]
    assert "overall_scores" in data
    assert "detailed_results" in data


def test_generate_run_id():
    """Test generating a deterministic run ID."""
    model_name = "test_model"
    task_ids = ["task1", "task2"]
    timestamp = "2023-01-01T00:00:00"

    # Test with timestamp
    run_id1 = ResultAggregator.generate_run_id(
        model_name, task_ids, timestamp=timestamp
    )
    assert isinstance(run_id1, str)
    assert len(run_id1) <= 32  # Default max length

    # Test with different timestamp
    run_id2 = ResultAggregator.generate_run_id(
        model_name, task_ids, timestamp="2023-01-02T00:00:00"
    )
    assert run_id1 != run_id2

    # Test with different model name
    run_id3 = ResultAggregator.generate_run_id(
        "another_model", task_ids, timestamp=timestamp
    )
    assert run_id1 != run_id3

    # Test with different task IDs
    run_id4 = ResultAggregator.generate_run_id(
        model_name, ["task3"], timestamp=timestamp
    )
    assert run_id1 != run_id4

    # Test custom max length
    run_id5 = ResultAggregator.generate_run_id(model_name, task_ids, max_length=16)
    assert len(run_id5) <= 16
