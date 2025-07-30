"""Tests for the EvaluationHarness component."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bench.evaluation import (
    EvaluationHarness,
    MetricCalculator,
    ModelRunner,
    ResultAggregator,
    TaskLoader,
)


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

        # Configure mocks
        mock_loader.return_value.load_task.return_value = MagicMock(
            task_id="test_task",
            dataset=[{"text": "test input"}],
            metrics=[{"name": "accuracy"}],
        )

        mock_runner.return_value.run_model.return_value = [
            {"label": "test", "score": 0.9}
        ]

        mock_calculator.return_value.calculate_metrics.return_value = {
            "accuracy": MagicMock(value=0.9, metadata={})
        }

        yield {
            "loader": mock_loader,
            "runner": mock_runner,
            "calculator": mock_calculator,
            "aggregator": mock_aggregator,
        }


@pytest.fixture
def test_harness(tmp_path):
    """Create an EvaluationHarness instance for testing."""
    # Create test directories
    tasks_dir = tmp_path / "tasks"
    results_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"

    tasks_dir.mkdir()
    results_dir.mkdir()
    cache_dir.mkdir()

    # Create a test task file
    task_file = tasks_dir / "test_task.yaml"
    task_file.write_text(
        """
    name: "Test Task"
    description: "A test task"
    input_schema:
      type: "object"
      properties:
        text: {type: "string"}
      required: ["text"]
    output_schema:
      type: "object"
      properties:
        label: {type: "string"}
        score: {type: "number"}
      required: ["label"]
    metrics:
      - name: "accuracy"
    dataset:
      - text: "test input"
        label: "test"
    """
    )

    return EvaluationHarness(
        tasks_dir=str(tasks_dir),
        results_dir=str(results_dir),
        cache_dir=str(cache_dir),
        log_level="ERROR",
    )


def test_harness_initialization(test_harness, tmp_path):
    """Test that the harness initializes with the correct directories."""
    assert test_harness.tasks_dir == Path(tmp_path) / "tasks"
    assert test_harness.results_dir == Path(tmp_path) / "results"
    assert test_harness.cache_dir == Path(tmp_path) / "cache"
    assert isinstance(test_harness.task_loader, TaskLoader)
    assert isinstance(test_harness.model_runner, ModelRunner)
    assert isinstance(test_harness.metric_calculator, MetricCalculator)
    assert isinstance(test_harness.result_aggregator, ResultAggregator)


@patch("bench.evaluation.EvaluationHarness._evaluate_task")
def test_evaluate(mock_evaluate_task, test_harness, tmp_path):
    """Test the main evaluate method."""
    # Setup mocks
    mock_evaluate_task.return_value = MagicMock(
        task_id="test_task", metrics={"accuracy": 0.9}, num_examples=1
    )

    # Create a dummy model file
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    (model_path / "pytorch_model.bin").write_text("")

    # Create a mock model
    mock_model = MagicMock()

    # Patch the ModelRunner to return our mock model
    with patch.object(test_harness.model_runner, "load_model") as mock_load_model:
        # Set up the mock to add the model to the _models dictionary
        def mock_load_model_side_effect(model_name, **kwargs):
            test_harness.model_runner._models[model_name] = mock_model

        mock_load_model.side_effect = mock_load_model_side_effect

        # Run evaluation
        model_id = "test_model"
        task_ids = ["test_task"]

        report = test_harness.evaluate(
            model_id=model_id,
            task_ids=task_ids,
            model_type="local",
            model_path=str(model_path),
            save_results=False,
        )

    # Check results
    assert report.model_name == model_id
    assert "test_task" in report.task_results
    assert report.task_results["test_task"].metrics["accuracy"] == 0.9

    # Check that _evaluate_task was called with the correct arguments
    mock_evaluate_task.assert_called_once()
    args, kwargs = mock_evaluate_task.call_args
    assert kwargs["model_id"] == model_id
    assert kwargs["model"] == mock_model
    assert kwargs["task"].task_id == "test_task"


def test_list_available_tasks(test_harness):
    """Test listing available tasks."""
    tasks = test_harness.list_available_tasks()
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "test_task"
    assert tasks[0]["name"] == "Test Task"
    assert tasks[0]["num_examples"] == 1


def test_get_task_info(test_harness):
    """Test getting information about a specific task."""
    task_info = test_harness.get_task_info("test_task")
    assert task_info["task_id"] == "test_task"
    assert task_info["name"] == "Test Task"
    assert "input_schema" in task_info
    assert "output_schema" in task_info
    assert "metrics" in task_info
    assert len(task_info["metrics"]) > 0


@patch("json.dump")
@patch("builtins.open", new_callable=MagicMock)
def test_save_to_cache(mock_open, mock_json_dump, test_harness, tmp_path):
    """Test saving results to cache."""
    from dataclasses import asdict, dataclass

    @dataclass
    class TestData:
        test: str

    cache_key = str(tmp_path / "cache" / "test_cache.json")
    data = TestData(test="data")
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    test_harness._save_to_cache(cache_key, data)

    mock_open.assert_called_once_with(cache_key, "w")
    args, _ = mock_json_dump.call_args
    assert args[0] == asdict(data)


@patch("json.load")
@patch("builtins.open", new_callable=MagicMock)
@patch("pathlib.Path.exists")
@patch("bench.evaluation.result_aggregator.TaskResult")
def test_load_from_cache(
    mock_task_result, mock_exists, mock_open, mock_json_load, test_harness, tmp_path
):
    """Test loading results from cache."""
    cache_key = str(tmp_path / "cache" / "test_cache.json")
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_json_load.return_value = {"test": "data"}
    mock_exists.return_value = True  # Make Path.exists() return True

    result = test_harness._load_from_cache(cache_key)

    assert result == mock_task_result.return_value
    mock_exists.assert_called_once()
    mock_open.assert_called_once_with(cache_key, "r")
    mock_json_load.assert_called_once_with(mock_file)
    mock_task_result.assert_called_once_with(**{"test": "data"})


def test_generate_run_id(test_harness):
    """Test generating a run ID."""
    model_id = "test_model"
    task_ids = ["task1", "task2"]

    run_id = test_harness._generate_run_id(model_id, task_ids)

    # Check that the run ID is a string and not too long
    assert isinstance(run_id, str)
    assert len(run_id) <= 32

    # Check that the same inputs produce the same run ID
    assert run_id == test_harness._generate_run_id(model_id, task_ids)

    # Check that different inputs produce different run IDs
    different_run_id = test_harness._generate_run_id("different_model", task_ids)
    assert run_id != different_run_id
