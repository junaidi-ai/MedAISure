from __future__ import annotations

from pathlib import Path

from bench.evaluation import EvaluationHarness
from bench.evaluation.result_aggregator import ResultAggregator


def _write_min_task_with_dataset(path: Path) -> str:
    data = {
        "task_id": "mini_qa",
        "name": "Mini QA",
        "task_type": "qa",
        "description": "Small QA task for report export tests",
        "metrics": ["clinical_correctness"],
        "inputs": [{"text": "example"}],
        "expected_outputs": [{"answer": "example"}],
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        "output_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
        "dataset": [
            {"text": "patient has fever", "answer": "fever"},
            {"text": "patient has cough", "answer": "cough"},
        ],
    }
    (path / "mini_qa.json").write_text(__import__("json").dumps(data))
    return "mini_qa"


def _local_model_module_and_path(tmp_path: Path) -> tuple[str, str]:
    module_path = "tests.fixtures.simple_local_model"
    model_path = str(tmp_path / "dummy.model")
    Path(model_path).write_text("dummy")
    return module_path, model_path


def test_markdown_and_html_exports_with_examples_and_validation_errors(
    tmp_path: Path,
) -> None:
    tasks_dir = tmp_path / "tasks"
    results_dir = tmp_path / "results"
    export_dir = tmp_path / "exports"
    tasks_dir.mkdir()
    results_dir.mkdir()
    export_dir.mkdir()

    task_id = _write_min_task_with_dataset(tasks_dir)
    module_path, model_path = _local_model_module_and_path(tmp_path)

    harness = EvaluationHarness(tasks_dir=str(tasks_dir), results_dir=str(results_dir))
    report = harness.evaluate(
        model_id="simple_local_model",
        task_ids=[task_id],
        model_type="local",
        model_path=model_path,
        module_path=module_path,
        default_label="neutral",
        rules=[("fever", "fever"), ("cough", "cough")],
        use_cache=False,
        save_results=False,
        batch_size=2,
    )

    # Inject sample validation errors to validate exporter rendering
    report.metadata.setdefault("validation_errors", []).append(
        "example validation issue"
    )

    run_id = report.metadata.get("run_id") or "test_run"
    agg = ResultAggregator(output_dir=export_dir)
    agg.reports[run_id] = report

    md_path = export_dir / f"{run_id}.md"
    html_path = export_dir / f"{run_id}.html"

    agg.export_report_markdown(
        run_id,
        md_path,
        include_examples=True,
        max_examples=2,
        include_validation_errors=True,
    )
    agg.export_report_html(
        run_id,
        html_path,
        include_examples=True,
        max_examples=2,
        include_validation_errors=True,
    )

    md_text = md_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")

    # Markdown assertions
    assert "## Examples" in md_text
    assert "prediction:" in md_text and "input:" in md_text
    assert "## Validation Errors" in md_text
    assert "example validation issue" in md_text

    # HTML assertions
    assert "Examples (truncated)" in html_text
    assert "<table>" in html_text and "prediction" in html_text
    assert "Validation Errors" in html_text
    assert "example validation issue" in html_text
