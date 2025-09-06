import json
from pathlib import Path


from bench.leaderboard.submission import (
    build_and_validate_submission,
    validate_submission,
)
from bench.evaluation.result_aggregator import ResultAggregator
from bench.models.evaluation_result import EvaluationResult


def _make_simple_report() -> ResultAggregator:
    ra = ResultAggregator(output_dir="results-test")
    er = EvaluationResult(
        model_id="unit-model",
        task_id="task-1",
        inputs=[{"id": "ex1", "text": "hello"}, {"id": "ex2", "text": "world"}],
        model_outputs=[
            {"label": "A", "reasoning": "short rationale"},
            {"label": "B"},
        ],
        metrics_results={"accuracy": 1.0},
    )
    ra.add_evaluation_result(er, run_id="run-xyz")
    return ra


def test_build_and_validate_submission_basic():
    ra = _make_simple_report()
    report = ra.get_report("run-xyz")

    payload = build_and_validate_submission(report, include_reasoning=True)

    # Basic top-level checks
    assert isinstance(payload, dict)
    assert payload.get("schema_version") == 1
    assert payload.get("run_id") == "run-xyz"
    assert payload.get("model_id") == "unit-model"
    assert isinstance(payload.get("submissions"), list) and payload["submissions"]

    # Per-task structure
    sub = payload["submissions"][0]
    assert sub.get("task_id") == "task-1"
    items = sub.get("items")
    assert isinstance(items, list) and len(items) == 2

    # Item structure
    assert items[0]["input_id"] == "ex1"
    assert isinstance(items[0]["prediction"], dict)
    # Reasoning present for first item
    assert "reasoning" in items[0]
    # Second item has no reasoning field
    assert "reasoning" not in items[1]


def test_export_leaderboard_submission_roundtrip(tmp_path: Path):
    ra = _make_simple_report()
    out = tmp_path / "submission.json"

    p = ra.export_leaderboard_submission(
        run_id="run-xyz", output_path=out, include_reasoning=True, validate=True
    )
    assert p.exists()

    data = json.loads(p.read_text())
    # Validate again (ensures JSON Schema path OK when available)
    validate_submission(data)

    # Minimal structural assertions
    assert data["run_id"] == "run-xyz"
    assert data["model_id"] == "unit-model"
    assert data["submissions"][0]["task_id"] == "task-1"
