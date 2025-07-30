"""Tests for the TaskLoader component."""

from pathlib import Path

import pytest

from bench.evaluation.task_loader import MedicalTask, TaskLoader


def test_task_loader_init(temp_tasks_dir):
    """Test that TaskLoader initializes correctly with a tasks directory."""
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    assert loader.tasks_dir == Path(temp_tasks_dir)


def test_load_task(temp_tasks_dir, example_task_definition):
    """Test loading a task from a YAML file."""
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    task = loader.load_task("example_task")

    assert isinstance(task, MedicalTask)
    assert task.task_id == "example_task"
    assert task.name == example_task_definition["name"]
    assert task.description == example_task_definition["description"]
    assert len(task.dataset) == len(example_task_definition["dataset"])


def test_load_nonexistent_task(temp_tasks_dir):
    """Test loading a non-existent task raises FileNotFoundError."""
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    with pytest.raises(FileNotFoundError):
        loader.load_task("nonexistent_task")


def test_validate_input(temp_tasks_dir):
    """Test input validation against task schema."""
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    task = loader.load_task("example_task")

    # Valid input
    valid_input = {
        "premise": "The patient has a fever.",
        "hypothesis": "The patient is sick.",
    }
    assert task.validate_input(valid_input) is True

    # Invalid input (missing required field)
    invalid_input = {"premise": "The patient has a fever."}
    assert task.validate_input(invalid_input) is False


def test_load_tasks(temp_tasks_dir):
    """Test loading multiple tasks at once."""
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    tasks = loader.load_tasks(["example_task"])

    assert isinstance(tasks, dict)
    assert "example_task" in tasks
    assert isinstance(tasks["example_task"], MedicalTask)
