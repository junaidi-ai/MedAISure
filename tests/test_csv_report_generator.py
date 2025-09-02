from pathlib import Path

from bench.models.benchmark_report import BenchmarkReport
from bench.models.evaluation_result import EvaluationResult
from bench.reports import CSVReportGenerator, ReportFactory


def make_sample_report() -> BenchmarkReport:
    res = EvaluationResult(
        model_id="unit-model",
        task_id="task_a",
        inputs=[{"q": "a?", "id": 1}],
        model_outputs=[{"a": "x", "score": 0.7}],
        metrics_results={"accuracy": 0.9, "f1": 0.8},
    )
    br = BenchmarkReport(
        model_id="unit-model",
        overall_scores={"accuracy": 0.91, "f1": 0.82},
        task_scores={
            "task_a": {"accuracy": 0.92, "f1": 0.83},
            "task_b": {"accuracy": 0.90, "f1": 0.81},
        },
        detailed_results=[res],
        metadata={"run_id": "unit-1234"},
    )
    return br


def test_csv_report_factory():
    gen = ReportFactory.create_generator("csv")
    assert isinstance(gen, CSVReportGenerator)


def test_csv_generate_and_save_dir(tmp_path: Path):
    br = make_sample_report()
    gen = CSVReportGenerator()
    data = gen.generate(br)
    assert isinstance(data, dict)
    # Ensure expected keys exist
    for k in [
        "overall_scores.csv",
        "task_scores.csv",
        "detailed_metrics.csv",
        "detailed_inputs.csv",
        "detailed_outputs.csv",
    ]:
        assert k in data
        assert isinstance(data[k], str)
        assert "," in data[k] or "metric" in data[k]  # crude sanity

    outdir = tmp_path / "csv_report"
    gen.save(data, outdir)
    assert outdir.exists()
    for k in data.keys():
        p = outdir / k
        assert p.exists()
        assert p.read_text().strip() != ""


def test_csv_save_single_file(tmp_path: Path):
    br = make_sample_report()
    gen = CSVReportGenerator()
    data = gen.generate(br)
    outfile = tmp_path / "report.csv"
    gen.save(data, outfile)
    assert outfile.exists()
    text = outfile.read_text()
    # Should be task_scores table by default
    assert "task_id,metric,score" in text
