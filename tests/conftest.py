"""Pytest configuration and fixtures for MEDDSAI benchmark tests."""
import os
import json
import yaml
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def example_task_definition(test_data_dir):
    """Load and return the example task definition."""
    task_file = test_data_dir / "example_task.yaml"
    with open(task_file, 'r') as f:
        if task_file.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)

@pytest.fixture(scope="session")
def temp_tasks_dir(tmp_path_factory, example_task_definition):
    """Create a temporary directory with example tasks for testing."""
    temp_dir = tmp_path_factory.mktemp("tasks")
    
    # Create example task file
    task_file = temp_dir / "example_task.yaml"
    with open(task_file, 'w') as f:
        yaml.dump(example_task_definition, f)
    
    return temp_dir
