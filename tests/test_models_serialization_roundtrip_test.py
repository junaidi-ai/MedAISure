"""Serialization round-trip tests for MedAISure data models.

Covers:
- to_dict/from_dict
- to_json/from_json
- to_yaml/from_yaml
- file I/O helpers
- CSV helpers where applicable
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.models import BenchmarkReport, EvaluationResult, MedicalTask, TaskType


@pytest.fixture
def tmpdir_path(tmp_path: Path) -> Path:
    return tmp_path


def test_medical_task_roundtrip_all(tmpdir_path: Path):
    task = MedicalTask(
        task_id="t1",
        task_type=TaskType.QA,
        description="desc",
        inputs=[{"q": "?"}],
        expected_outputs=[{"a": "!"}],
        metrics=["accuracy"],
    )

    # dict/json/yaml
    d = task.to_dict()
    assert MedicalTask.from_dict(d).model_dump() == task.model_dump()

    j = task.to_json(indent=2)
    assert MedicalTask.from_json(j).model_dump() == task.model_dump()

    y = task.to_yaml()
    assert MedicalTask.from_yaml(y).model_dump() == task.model_dump()

    # file I/O
    jf = tmpdir_path / "task.json"
    yf = tmpdir_path / "task.yaml"
    task.save(jf)
    task.save(yf)
    assert MedicalTask.from_file(jf).model_dump() == task.model_dump()
    assert MedicalTask.from_file(yf).model_dump() == task.model_dump()


def test_evaluation_result_roundtrip_all(tmpdir_path: Path):
    res = EvaluationResult(
        model_id="m1",
        task_id="t1",
        inputs=[{"q": "?"}],
        model_outputs=[{"a": "!"}],
        metrics_results={"accuracy": 0.9},
    )

    # dict/json/yaml (exclude timestamp which is normalized)
    d = res.to_dict(exclude={"timestamp"})
    assert EvaluationResult.from_dict(d).model_dump(exclude={"timestamp"}) == d

    j = res.to_json(indent=2)
    assert EvaluationResult.from_json(j).model_dump(
        exclude={"timestamp"}
    ) == res.model_dump(exclude={"timestamp"})

    y = res.to_yaml()
    assert EvaluationResult.from_yaml(y).model_dump(
        exclude={"timestamp"}
    ) == res.model_dump(exclude={"timestamp"})

    # file I/O
    jf = tmpdir_path / "res.json"
    yf = tmpdir_path / "res.yaml"
    res.save(jf)
    res.save(yf)
    assert EvaluationResult.from_file(jf).model_dump(
        exclude={"timestamp"}
    ) == res.model_dump(exclude={"timestamp"})
    assert EvaluationResult.from_file(yf).model_dump(
        exclude={"timestamp"}
    ) == res.model_dump(exclude={"timestamp"})

    # CSV helpers
    assert "q" in res.inputs_to_csv()
    assert "a" in res.outputs_to_csv()


def test_benchmark_report_roundtrip_all(tmpdir_path: Path):
    report = BenchmarkReport(
        model_id="m1",
        detailed_results=[
            EvaluationResult(
                model_id="m1",
                task_id="t1",
                inputs=[{"q": "?"}],
                model_outputs=[{"a": "!"}],
                metrics_results={"accuracy": 1.0},
            )
        ],
    )

    d = report.to_dict()
    assert BenchmarkReport.from_dict(d).model_dump(
        exclude={"timestamp"}
    ) == report.model_dump(exclude={"timestamp"})

    j = report.to_json(indent=2)
    assert BenchmarkReport.from_json(j).model_dump(
        exclude={"timestamp"}
    ) == report.model_dump(exclude={"timestamp"})

    y = report.to_yaml()
    assert BenchmarkReport.from_yaml(y).model_dump(
        exclude={"timestamp"}
    ) == report.model_dump(exclude={"timestamp"})

    # file I/O
    jf = tmpdir_path / "rep.json"
    yf = tmpdir_path / "rep.yaml"
    report.save(jf)
    report.save(yf)
    assert BenchmarkReport.from_file(jf).model_dump(
        exclude={"timestamp"}
    ) == report.model_dump(exclude={"timestamp"})
    assert BenchmarkReport.from_file(yf).model_dump(
        exclude={"timestamp"}
    ) == report.model_dump(exclude={"timestamp"})

    # CSV helpers
    report.add_evaluation_result(
        EvaluationResult(
            model_id="m1",
            task_id="t2",
            inputs=[{"q": "2"}],
            model_outputs=[{"a": "2"}],
            metrics_results={"accuracy": 0.0},
        )
    )
    csv_overall = report.overall_scores_to_csv()
    csv_tasks = report.task_scores_to_csv()
    assert "metric" in csv_overall and "score" in csv_overall
    assert "task_id" in csv_tasks and "metric" in csv_tasks and "score" in csv_tasks
