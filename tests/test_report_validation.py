from __future__ import annotations

from pathlib import Path

from bench.models.benchmark_report import BenchmarkReport
from bench.models.evaluation_result import EvaluationResult
from bench.reports.factory import ReportFactory


def sample_report() -> BenchmarkReport:
    res1 = EvaluationResult(
        model_id="demo-model",
        task_id="task.a",
        metrics_results={"accuracy": 0.8, "f1": 0.7},
        inputs=[{"text": "hello"}],
        model_outputs=[{"response": "world"}],
        metadata={"note": "ok"},
    )
    res2 = EvaluationResult(
        model_id="demo-model",
        task_id="task.b",
        metrics_results={"accuracy": 0.9},
        inputs=[{"text": "foo"}],
        model_outputs=[{"response": "bar"}],
    )
    report = BenchmarkReport(model_id="demo-model")
    report.add_evaluation_result(res1)
    report.add_evaluation_result(res2)
    return report


def test_json_markdown_html_csv_validation(tmp_path: Path) -> None:
    br = sample_report()

    # JSON
    json_gen = ReportFactory.create_generator("json")
    json_payload = json_gen.generate(br)
    json_gen.validate(json_payload)

    # Markdown
    md_gen = ReportFactory.create_generator("markdown")
    md_text = md_gen.generate(br)
    md_gen.validate(md_text)

    # HTML
    html_gen = ReportFactory.create_generator("html")
    html_text = html_gen.generate(br)
    html_gen.validate(html_text)

    # CSV
    csv_gen = ReportFactory.create_generator("csv")
    csv_map = csv_gen.generate(br)
    csv_gen.validate(csv_map)

    # Also ensure CSV save works to directory
    out_dir = tmp_path / "csv_out"
    csv_gen.save(csv_map, out_dir)
    assert (out_dir / "overall_scores.csv").exists()
    assert (out_dir / "task_scores.csv").exists()
    assert (out_dir / "detailed_metrics.csv").exists()
    assert (out_dir / "detailed_inputs.csv").exists()
    assert (out_dir / "detailed_outputs.csv").exists()
