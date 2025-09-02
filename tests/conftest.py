"""Pytest configuration and fixtures for MedAISure benchmark tests."""

from pathlib import Path

import pytest
import yaml
import json

from bench.data.security import SecureDataHandler


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def example_task_definition():
    """Return a complete example task definition for testing."""
    return {
        "task_id": "example_task",
        "task_type": "qa",
        "name": "Example Medical QA Task",
        "description": "A sample medical QA task for testing purposes",
        "inputs": [
            {"text": "What are the symptoms of COVID-19?"},
            {"text": "How is diabetes diagnosed?"},
        ],
        "expected_outputs": [
            {"answer": "Common symptoms include fever, cough, and fatigue."},
            {"answer": "Through blood tests like fasting blood sugar or A1C."},
        ],
        "metrics": ["accuracy", "f1"],
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        "dataset": [
            {
                "input": {"text": "What are the symptoms of COVID-19?"},
                "output": {
                    "answer": "Common symptoms include fever, cough, and fatigue."
                },
            },
            {
                "input": {"text": "How is diabetes diagnosed?"},
                "output": {
                    "answer": "Through blood tests like fasting blood sugar or A1C."
                },
            },
        ],
    }


@pytest.fixture
def temp_tasks_dir(tmp_path, example_task_definition):
    """Create a temporary directory with example tasks for testing.

    Uses function scope to ensure each test gets a fresh directory.
    """
    temp_dir = tmp_path / "tasks"
    temp_dir.mkdir()

    # Create example task file
    task_file = temp_dir / "example_task.yaml"
    with open(task_file, "w") as f:
        yaml.dump(example_task_definition, f)

    return temp_dir


# -----------------------------
# Connector test data fixtures
# -----------------------------


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    """Create a simple JSON dataset file (list of objects)."""
    data = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
    f = tmp_path / "data.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    return f


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a simple CSV dataset file (with header)."""
    import csv

    rows = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
    f = tmp_path / "data.csv"
    with f.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(rows)
    return f


@pytest.fixture
def encrypted_json_file(tmp_path: Path) -> Path:
    """Create a JSON dataset whose string fields are encrypted using SecureDataHandler.

    This is useful to ensure connector + handler integration works end-to-end.
    """
    handler = SecureDataHandler("test-pass")
    raw = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    enc = [handler.encrypt_data(x) for x in raw]
    f = tmp_path / "enc.json"
    f.write_text(json.dumps(enc), encoding="utf-8")
    return f


@pytest.fixture
def encrypted_csv_file(tmp_path: Path) -> Path:
    """Create a CSV dataset whose string fields are encrypted using SecureDataHandler."""
    import csv

    handler = SecureDataHandler("test-pass")
    raw = [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]
    enc = [handler.encrypt_data(x) for x in raw]
    f = tmp_path / "enc.csv"
    with f.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "name"])
        writer.writeheader()
        writer.writerows(enc)
    return f
