"""Tests for the EvaluationHarness component."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bench.evaluation import EvaluationHarness
from bench.models.benchmark_report import BenchmarkReport
from bench.models.evaluation_result import EvaluationResult
from bench.models.medical_task import MedicalTask, TaskType


@pytest.fixture
def mock_components():
    """Create mock components for testing the harness."""
    with patch("bench.evaluation.task_loader.TaskLoader") as mock_loader, patch(
        "bench.evaluation.model_runner.ModelRunner"
    ) as mock_runner, patch(
        "bench.evaluation.metric_calculator.MetricCalculator"
    ) as mock_calculator, patch(
        "bench.evaluation.result_aggregator.ResultAggregator"
    ) as mock_aggregator:

        # Create a mock task
        mock_task = MedicalTask(
            task_id="test_task",
            name="Test Task",
            task_type=TaskType.QA,
            description="Test task",
            inputs=[{"question": "Test question"}],
            expected_outputs=[{"answer": "Test answer"}],
            metrics=["accuracy", "f1"],
            input_schema={"question": "string"},
            output_schema={"answer": "string"},
            dataset=[{"question": "Test question", "answer": "Test answer"}],
        )

        # Configure mocks
        mock_loader.return_value.load_task.return_value = mock_task
        mock_runner.return_value.run.return_value = [{"answer": "Test answer"}]
        mock_calculator.return_value.calculate.return_value = {
            "accuracy": 1.0,
            "f1": 0.95,
        }
        mock_aggregator.return_value = MagicMock()

        yield {
            "loader": mock_loader,
            "runner": mock_runner,
            "calculator": mock_calculator,
            "aggregator": mock_aggregator,
        }


@pytest.fixture
def test_harness(tmp_path, mock_components):
    """Create an EvaluationHarness instance for testing."""
    # Create necessary directories
    (tmp_path / "tasks").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "cache").mkdir()

    return EvaluationHarness(
        tasks_dir=str(tmp_path / "tasks"),
        results_dir=str(tmp_path / "results"),
        cache_dir=str(tmp_path / "cache"),
    )


def test_harness_initialization(test_harness, tmp_path):
    """Test that the harness initializes with the correct directories."""
    assert str(test_harness.tasks_dir) == str(Path(tmp_path) / "tasks")
    assert str(test_harness.results_dir) == str(Path(tmp_path) / "results")
    assert str(test_harness.cache_dir) == str(Path(tmp_path) / "cache")
    assert test_harness.task_loader is not None
    assert test_harness.model_runner is not None
    assert test_harness.metric_calculator is not None
    assert test_harness.result_aggregator is not None


@patch("bench.evaluation.EvaluationHarness._evaluate_task")
@patch("bench.evaluation.task_loader.TaskLoader")
@patch("bench.evaluation.model_runner.ModelRunner")
def test_evaluate(
    mock_model_runner_cls,
    mock_task_loader_cls,
    mock_evaluate_task,
    test_harness,
    mock_components,
    tmp_path,
):
    """Test the main evaluate method."""
    # Create a test task file in the temporary directory
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    task_data = {
        "task_id": "test_task",
        "name": "Test Task",
        "task_type": "qa",
        "description": "Test task",
        "inputs": [{"question": "Test question"}],
        "expected_outputs": [{"answer": "Test answer"}],
        "metrics": ["accuracy", "f1"],
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
                "input": {"question": "Test question"},
                "output": {"answer": "Test answer"},
            }
        ],
    }

    task_file = tasks_dir / "test_task.json"
    with open(task_file, "w") as f:
        json.dump(task_data, f)

    # Create a mock task
    mock_task = MagicMock()
    mock_task.task_id = "test_task"

    # Create a mock task loader
    mock_task_loader = MagicMock()
    mock_task_loader.load_task.return_value = mock_task
    mock_task_loader_cls.return_value = mock_task_loader

    # Create a mock model runner instance
    mock_model_runner = MagicMock()
    mock_model_runner_cls.return_value = mock_model_runner

    # Mock the models dictionary to return a mock model
    mock_model_runner._models = MagicMock()
    mock_model_runner._models.__getitem__.return_value = MagicMock()

    # Mock the evaluate method to return a BenchmarkReport with all required fields
    def mock_evaluate(*args, **kwargs):
        mock_result = EvaluationResult(
            model_id="test_model",
            task_id="test_task",
            inputs=[{"question": "Test question"}],
            model_outputs=[{"answer": "Test answer"}],
            metrics_results={"accuracy": 1.0, "f1": 1.0},
            metadata={"temperature": 0.7},
        )

        # Create a BenchmarkReport with all required fields
        report = BenchmarkReport(
            model_id="test_model",
            overall_scores={"average_score": 1.0},
            task_scores={"test_task": {"average_score": 1.0}},
            detailed_results=[mock_result],
            metadata={
                "total_time_seconds": 1.0,
                "num_tasks": 1,
                "batch_size": 8,
                "model_type": "huggingface",
            },
        )

        return report

    # Configure the mocks
    mock_evaluate_task.return_value = mock_evaluate()

    # Create a new test harness with the mocked components
    test_harness = EvaluationHarness(
        tasks_dir=str(tasks_dir),
        results_dir=str(tmp_path / "results"),
        cache_dir=str(tmp_path / "cache"),
    )

    # Create a mock metric calculator
    mock_metric_calculator = MagicMock()

    # Create a mock task result to be returned by _evaluate_task

    mock_task_result = EvaluationResult(
        model_id="test_model",
        task_id="test_task",
        inputs=[{"question": "Test question"}],
        model_outputs=[{"answer": "Test answer"}],
        metrics_results={"accuracy": 1.0, "f1": 1.0},
        metadata={"temperature": 0.7},
    )

    # Mock _evaluate_task to return our mock result
    test_harness._evaluate_task = MagicMock(return_value=mock_task_result)

    # Create a mock result aggregator
    mock_result_aggregator = MagicMock()

    # Replace the components with our mocks
    test_harness.task_loader = mock_task_loader
    test_harness.model_runner = mock_model_runner
    test_harness.metric_calculator = mock_metric_calculator
    test_harness.result_aggregator = mock_result_aggregator

    # Run evaluation
    results = test_harness.evaluate(
        model_id="test_model",
        task_ids=["test_task"],
        model_kwargs={"temperature": 0.7},
        use_cache=False,
    )

    # Check that the task was loaded
    mock_task_loader.load_task.assert_called_once_with("test_task")

    # Check that _evaluate_task was called with the correct arguments
    test_harness._evaluate_task.assert_called_once_with(
        model=mock_model_runner._models.__getitem__.return_value,
        model_id="test_model",
        task=mock_task,
        batch_size=8,
    )

    # Verify the results match the expected structure
    assert isinstance(results, BenchmarkReport)
    assert results.model_id == "test_model"
    assert results.overall_scores["average_score"] == 1.0
    assert results.task_scores == {"test_task": {"average_score": 1.0}}
    assert len(results.detailed_results) == 1
    # The detailed_results[0] is a placeholder result with task_id="unknown"
    assert results.detailed_results[0].task_id == "unknown"
    assert results.detailed_results[0].metrics_results == {"score": 0.0}


def test_list_available_tasks(test_harness, tmp_path):
    """Test listing available tasks."""
    # Create some test task files with minimal valid content
    tasks_dir = Path(test_harness.tasks_dir)

    # Create task files with required fields
    task1_data = {
        "task_id": "task1",
        "name": "Task 1",
        "description": "Test task 1",
        "metrics": [{"name": "accuracy"}],
        "dataset": [{"input": "test", "output": "test"}],
    }

    task2_data = {
        "task_id": "task2",
        "name": "Task 2",
        "description": "Test task 2",
        "metrics": [{"name": "f1"}],
        "dataset": [{"input": "test", "output": "test"}],
    }

    (tasks_dir / "task1.json").write_text(json.dumps(task1_data))
    (tasks_dir / "task2.json").write_text(json.dumps(task2_data))

    # Get list of tasks
    tasks = test_harness.list_available_tasks()

    # Check that we got a list of task dictionaries with the expected task_ids
    task_ids = [task["task_id"] for task in tasks]
    assert set(task_ids) == {"task1", "task2"}


def test_get_task_info(test_harness, tmp_path):
    """Test getting information about a specific task."""
    # Create a test task file
    task_data = {
        "task_id": "test_task",
        "task_type": "qa",
        "description": "Test task",
        "inputs": [{"question": "Test question"}],
        "expected_outputs": [{"answer": "Test answer"}],
        "metrics": ["accuracy", "f1"],
    }

    task_file = Path(test_harness.tasks_dir) / "test_task.json"
    with open(task_file, "w") as f:
        json.dump(task_data, f)

    # Get task info
    task_info = test_harness.get_task_info("test_task")

    # Check the result
    assert task_info["task_id"] == "test_task"
    assert task_info["description"] == "Test task"
    assert len(task_info["metrics"]) == 2


def test_save_to_cache(test_harness, tmp_path):
    """Test saving results to cache."""
    # Create a test result
    from datetime import datetime, timezone

    from bench.models.evaluation_result import EvaluationResult

    # Use the existing cache directory from test_harness fixture
    Path(test_harness.cache_dir)

    # Create a test result with serializable data
    result_data = {
        "model_id": "test_model",
        "task_id": "test_task",
        "inputs": [{"question": "Test question"}],
        "model_outputs": [{"answer": "Test answer"}],
        "metrics_results": {"accuracy": 0.9},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Create EvaluationResult from dict
    result = EvaluationResult(**result_data)

    # Save to cache
    cache_key = test_harness._get_cache_key("test_run", "test_task")
    test_harness._save_to_cache(cache_key, result)

    # Check that the file was created at the expected location
    cache_file = Path(cache_key)
    assert cache_file.exists()

    # Load and verify the cached data
    with open(cache_file, "r") as f:
        cached_data = json.load(f)

    assert cached_data["model_id"] == "test_model"
    assert cached_data["task_id"] == "test_task"


def test_load_from_cache(test_harness, tmp_path):
    """Test loading results from cache."""
    # Use the existing cache directory from test_harness fixture
    Path(test_harness.cache_dir)

    # Generate the cache key first
    cache_key = test_harness._get_cache_key("test_run", "test_task")

    # Create a test cache file at the expected location
    cache_file = Path(cache_key)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Create test data that matches the expected format
    cache_data = {
        "model_id": "test_model",
        "task_id": "test_task",
        "inputs": [{"question": "Test question"}],
        "model_outputs": [{"answer": "Test answer"}],
        "metrics_results": {"accuracy": 0.9},
        "timestamp": "2023-01-01T00:00:00+00:00",
    }

    # Write the test data to the cache file
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

    # Load from cache
    result = test_harness._load_from_cache(cache_key)

    # Check that the result was loaded correctly
    assert result is not None
    assert result.model_id == "test_model"
    assert result.task_id == "test_task"
    assert result.metrics_results["accuracy"] == 0.9

    # Test when cache file doesn't exist
    non_existent_key = test_harness._get_cache_key("test_model", "nonexistent_task")
    result = test_harness._load_from_cache(non_existent_key)
    assert result is None


def test_generate_run_id(test_harness):
    """Test generating a run ID."""
    # Test with the same parameters produces the same ID
    params = {"param1": "value1", "param2": 42}
    id1 = test_harness._generate_run_id("test_model", params)
    id2 = test_harness._generate_run_id("test_model", params)
    assert id1 == id2

    # Test different parameters produce different IDs
    id3 = test_harness._generate_run_id("different_model", params)
    assert id1 != id3

    # Test different params produce different IDs
    id4 = test_harness._generate_run_id("test_model", {"param1": "value2"})
    assert id1 != id4

    # Test that the ID is a string
    assert isinstance(id1, str)
    assert len(id1) > 0
