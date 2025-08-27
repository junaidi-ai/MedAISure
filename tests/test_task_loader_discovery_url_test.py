"""Tests for TaskLoader discovery utilities and URL loading paths."""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import yaml

from bench.evaluation.task_loader import TaskLoader


def _minimal_task(name: str = "Alpha", metric_as_dict: bool = False) -> Dict[str, Any]:
    metrics = [{"name": "clinical_correctness"} if metric_as_dict else "clinical_correctness"]
    return {
        "task_id": name.lower(),
        "task_type": "qa",
        "name": name,
        "description": f"Task {name}",
        "inputs": [{"text": "Q: example?"}],
        "expected_outputs": [{"answer": "example"}],
        "metrics": metrics,
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
        "dataset": [{"input": {"text": "Q: example?"}, "output": {"answer": "example"}}],
    }


def test_discover_tasks_lists_yaml_and_json(temp_tasks_dir, example_task_definition):
    # Arrange: create additional YAML and JSON tasks
    alpha_yaml = temp_tasks_dir / "alpha.yaml"
    beta_json = temp_tasks_dir / "beta.json"

    with open(alpha_yaml, "w") as f:
        yaml.safe_dump(_minimal_task("Alpha"), f)

    with open(beta_json, "w") as f:
        json.dump(_minimal_task("Beta"), f)

    # Act
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    registry = loader.discover_tasks()

    # Assert
    assert "example_task" in registry  # from fixture
    assert "alpha" in registry
    assert "beta" in registry
    assert Path(registry["alpha"]).name == "alpha.yaml"
    assert Path(registry["beta"]).name == "beta.json"


def test_list_available_tasks_returns_metadata(temp_tasks_dir, example_task_definition):
    # Arrange
    alpha_yaml = temp_tasks_dir / "alpha.yaml"
    beta_json = temp_tasks_dir / "beta.json"

    alpha = _minimal_task("Alpha", metric_as_dict=True)  # test metric normalization
    alpha["dataset"] = [
        {"input": {"text": "Q1"}, "output": {"answer": "A1"}},
        {"input": {"text": "Q2"}, "output": {"answer": "A2"}},
    ]

    beta = _minimal_task("Beta")
    beta["dataset"] = [
        {"input": {"text": "Q1"}, "output": {"answer": "A1"}},
    ]

    with open(alpha_yaml, "w") as f:
        yaml.safe_dump(alpha, f)

    with open(beta_json, "w") as f:
        json.dump(beta, f)

    # Act
    loader = TaskLoader(tasks_dir=str(temp_tasks_dir))
    entries = loader.list_available_tasks()

    # Assert
    by_id = {e["task_id"]: e for e in entries}
    assert by_id["alpha"]["name"] == "Alpha"
    assert by_id["alpha"]["num_examples"] == 2
    assert by_id["alpha"]["metrics"] == ["clinical_correctness"]  # normalized

    assert by_id["beta"]["name"] == "Beta"
    assert by_id["beta"]["num_examples"] == 1
    assert by_id["beta"]["metrics"] == ["clinical_correctness"]


def test_load_task_from_url_yaml(monkeypatch, tmp_path):
    # Arrange
    data = _minimal_task("RemoteYAML")
    yaml_text = yaml.safe_dump(data)

    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/x-yaml"}
    mock_resp.text = yaml_text

    with patch("bench.evaluation.task_loader.requests.get", return_value=mock_resp) as _:
        loader = TaskLoader(tasks_dir=str(tmp_path))
        # Act
        task = loader.load_task("https://example.com/remote.yaml")

    # Assert
    assert task.name == "RemoteYAML"
    assert task.metrics == ["clinical_correctness"]
    assert len(task.inputs) == 1
    assert len(task.expected_outputs) == 1


def test_load_task_from_url_json(monkeypatch, tmp_path):
    # Arrange
    data = _minimal_task("RemoteJSON")
    json_text = json.dumps(data)

    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.text = json_text

    with patch("bench.evaluation.task_loader.requests.get", return_value=mock_resp) as _:
        loader = TaskLoader(tasks_dir=str(tmp_path))
        # Act
        task = loader.load_task("https://example.com/remote.json")

    # Assert
    assert task.name == "RemoteJSON"
    assert task.metrics == ["clinical_correctness"]
    assert len(task.inputs) == 1
    assert len(task.expected_outputs) == 1
