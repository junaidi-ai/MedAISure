from pathlib import Path

import pytest

from bench.models.benchmark_report import BenchmarkReport
from bench.reports import (
    ReportFactory,
    JSONReportGenerator,
    MarkdownReportGenerator,
    HTMLReportGenerator,
)


@pytest.fixture()
def sample_report() -> BenchmarkReport:
    return BenchmarkReport(
        model_id="unit-model",
        overall_scores={"accuracy": 0.91, "f1": 0.82},
        task_scores={
            "task_a": {"accuracy": 0.92, "f1": 0.83},
            "task_b": {"accuracy": 0.90, "f1": 0.81},
        },
        detailed_results=[],
        metadata={"run_id": "unit-1234"},
    )


def test_report_factory_creates_generators():
    assert isinstance(ReportFactory.create_generator("json"), JSONReportGenerator)
    assert isinstance(ReportFactory.create_generator("md"), MarkdownReportGenerator)
    assert isinstance(
        ReportFactory.create_generator("markdown"), MarkdownReportGenerator
    )
    assert isinstance(ReportFactory.create_generator("html"), HTMLReportGenerator)
    with pytest.raises(Exception):
        ReportFactory.create_generator("unknown")


def test_json_report_generator_generates_and_saves(
    tmp_path: Path, sample_report: BenchmarkReport
):
    gen = JSONReportGenerator()
    data = gen.generate(sample_report)
    assert isinstance(data, dict)
    assert data.get("model_id") == "unit-model"
    out = tmp_path / "report.json"
    gen.save(data, out)
    assert out.exists()
    assert out.read_text().strip().startswith("{")


def test_markdown_report_generator_generates_and_saves(
    tmp_path: Path, sample_report: BenchmarkReport
):
    gen = MarkdownReportGenerator()
    content = gen.generate(sample_report)
    assert isinstance(content, str)
    # basic sanity: headers and model id present
    assert "#" in content and "unit-model" in content
    out = tmp_path / "report.md"
    gen.save(content, out)
    assert out.exists()
    assert "unit-model" in out.read_text()


def test_html_report_generator_generates_and_saves(
    tmp_path: Path, sample_report: BenchmarkReport
):
    gen = HTMLReportGenerator()
    content = gen.generate(sample_report)
    assert isinstance(content, str)
    # basic sanity: html tags present
    assert "<html" in content.lower() and "</html>" in content.lower()
    out = tmp_path / "report.html"
    gen.save(content, out)
    assert out.exists()
    text = out.read_text().lower()
    assert "<html" in text and "unit-model" in text
