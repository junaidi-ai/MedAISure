"""Tests for the TaskLoader component."""

import json
from pathlib import Path

import pytest

from bench.evaluation.task_loader import TaskLoader
from bench.models.medical_task import MedicalTask, TaskType

# Use the example_task_definition from conftest.py instead of defining it here


@pytest.mark.compat
def test_task_loader_init(temp_tasks_dir):
    """Test that TaskLoader initializes correctly with a tasks directory."""
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    assert loader.tasks_dir == Path(temp_tasks_dir)


def test_load_task(temp_tasks_dir, example_task_definition):
    """Test loading a task from a YAML file."""
    # Create a task file
    task_file = Path(temp_tasks_dir) / "example_task.json"
    with open(task_file, "w") as f:
        json.dump(example_task_definition, f)

    # Load the task
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    task = loader.load_task("example_task")

    # Verify the loaded task
    assert isinstance(task, MedicalTask)
    assert task.task_id == "example_task"
    assert task.task_type == TaskType.QA
    assert task.description == example_task_definition["description"]
    assert len(task.inputs) == len(example_task_definition["inputs"])
    assert len(task.expected_outputs) == len(
        example_task_definition["expected_outputs"]
    )
    assert task.metrics == example_task_definition["metrics"]


@pytest.mark.compat
def test_load_nonexistent_task(temp_tasks_dir):
    """Test loading a non-existent task raises FileNotFoundError."""
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    with pytest.raises(FileNotFoundError):
        loader.load_task("nonexistent_task")


# Input validation is handled by Pydantic's built-in validation


def test_load_tasks(temp_tasks_dir, example_task_definition):
    """Test loading multiple tasks at once."""
    # Create multiple task files
    tasks = {}
    for i in range(3):
        task_id = f"task_{i}"
        task_data = example_task_definition.copy()
        task_data["task_id"] = task_id
        task_file = Path(temp_tasks_dir) / f"{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f)
        tasks[task_id] = task_data

    # Load all tasks
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    loaded_tasks = loader.load_tasks(["task_0", "task_1"])

    # Verify loaded tasks
    assert len(loaded_tasks) == 2
    assert "task_0" in loaded_tasks
    assert "task_1" in loaded_tasks
    assert isinstance(loaded_tasks["task_0"], MedicalTask)
    assert isinstance(loaded_tasks["task_1"], MedicalTask)
    assert loaded_tasks["task_0"].task_id == "task_0"
    assert loaded_tasks["task_1"].task_id == "task_1"
