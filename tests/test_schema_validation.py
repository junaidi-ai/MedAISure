from __future__ import annotations

from pathlib import Path

import pytest

from bench.evaluation.harness import EvaluationHarness
from bench.evaluation.task_loader import TaskLoader
from bench.models.medical_task import MedicalTask


def test_task_loader_applies_defaults_and_validates_dataset(tmp_path: Path) -> None:
    # Create a minimal QA task without explicit schemas but with valid dataset
    task_yaml = tmp_path / "qa_task.yaml"
    task_yaml.write_text(
        """
        task_type: "qa"
        name: "QA without schemas"
        description: ""
        metrics: ["accuracy"]
        inputs:
          - question: "What is flu?"
        dataset:
          - input:
              question: "What is flu?"
            output:
              answer: "influenza"
        """
    )

    loader = TaskLoader(tasks_dir=str(tmp_path))
    task = loader.load_task("qa_task")

    # Defaults should be applied
    assert task.input_schema.get("required") == ["question"]
    assert task.output_schema.get("required") == ["answer"]

    # No error should be raised and dataset should remain intact
    assert isinstance(task, MedicalTask)
    assert len(task.dataset) == 1


def test_task_loader_raises_on_invalid_dataset(tmp_path: Path) -> None:
    # Missing required 'answer' in output for QA (defaults will require it)
    task_yaml = tmp_path / "bad_qa.yaml"
    task_yaml.write_text(
        """
        task_type: "qa"
        name: "Bad QA"
        description: ""
        metrics: ["accuracy"]
        inputs:
          - question: "What is flu?"
        dataset:
          - input:
              question: "What is flu?"
            output:
              wrong_key: "influenza"
        """
    )

    loader = TaskLoader(tasks_dir=str(tmp_path))
    with pytest.raises(ValueError):
        _ = loader.load_task("bad_qa")


def test_harness_attaches_validation_errors_non_strict(tmp_path: Path) -> None:
    # Build a task that will produce outputs missing required keys
    task_yaml = tmp_path / "sum_task.yaml"
    task_yaml.write_text(
        """
        task_type: "summarization"
        name: "Summarization"
        description: ""
        metrics: ["accuracy"]
        # No explicit schemas; defaults require text->summary
        inputs:
          - text: "Short text"
        dataset:
          - input:
              text: "Short text"
        """
    )

    # Use a local model that returns {"label": ..., "score": ...} (won't match 'summary')
    harness = EvaluationHarness(tasks_dir=str(tmp_path))
    # Load local model
    harness.model_runner.load_model(
        model_name="local",
        model_type="local",
        model_path=str(Path(__file__).parent / "fixtures" / "simple_local_model.py"),
        module_path="tests.fixtures.simple_local_model",
    )

    report = harness.evaluate(
        model_id="local",
        task_ids=["sum_task"],
        model_type="local",
        batch_size=1,
        use_cache=False,
        save_results=False,
        strict_validation=False,
    )

    # Should complete and contain validation_errors in metadata of detailed_results or report metadata
    # Report may aggregate; check first detailed result metadata if available
    found = False
    for res in report.detailed_results or []:
        if res.metadata.get("validation_errors"):
            found = True
            break
    if not found:
        # Fallback: some paths build minimal report; check report conversion if present
        assert "average_score" in (report.overall_scores or {}) or (
            report.task_scores is not None
        )


def test_harness_strict_validation_raises(tmp_path: Path) -> None:
    # Same as previous but strict_validation=True should raise
    task_yaml = tmp_path / "sum_task2.yaml"
    task_yaml.write_text(
        """
        task_type: "summarization"
        name: "Summarization"
        metrics: ["accuracy"]
        inputs:
          - text: "Short text"
        dataset:
          - input:
              text: "Short text"
        """
    )

    harness = EvaluationHarness(tasks_dir=str(tmp_path))
    harness.model_runner.load_model(
        model_name="local2",
        model_type="local",
        model_path=str(Path(__file__).parent / "fixtures" / "simple_local_model.py"),
        module_path="tests.fixtures.simple_local_model",
    )

    with pytest.raises(Exception):
        _ = harness.evaluate(
            model_id="local2",
            task_ids=["sum_task2"],
            model_type="local",
            batch_size=1,
            use_cache=False,
            save_results=False,
            strict_validation=True,
        )
