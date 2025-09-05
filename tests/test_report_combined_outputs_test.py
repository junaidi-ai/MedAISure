from bench.models.benchmark_report import BenchmarkReport
from bench.reports.factory import ReportFactory


def _make_report_with_combined():
    r = BenchmarkReport(
        model_id="test-model",
        task_scores={
            "t1": {"diagnostics": 0.8, "safety": 0.6},
            "t2": {"diagnostics": 0.4},
        },
        overall_scores={},
        detailed_results=[],
        metadata={},
    )
    r.add_combined_score(
        {"diagnostics": 0.5, "safety": 0.5}, metric_name="combined_score"
    )
    return r


def test_json_report_contains_combined_score():
    r = _make_report_with_combined()
    gen = ReportFactory.create_generator("json")
    payload = gen.generate(r)
    assert "combined_score" in payload["overall_scores"]
    # Ensure per-task combined present as well
    assert all(
        ("combined_score" in scores) for scores in payload["task_scores"].values()
    )


def test_csv_report_contains_combined_score():
    r = _make_report_with_combined()
    gen = ReportFactory.create_generator("csv")
    mapping = gen.generate(r)
    assert "overall_scores.csv" in mapping and "task_scores.csv" in mapping
    overall_csv = mapping["overall_scores.csv"]
    task_csv = mapping["task_scores.csv"]
    assert "combined_score" in overall_csv
    assert "combined_score" in task_csv


def test_markdown_report_contains_combined_score():
    r = _make_report_with_combined()
    gen = ReportFactory.create_generator("md")
    text = gen.generate(r)
    assert "## Overall Scores" in text
    assert "combined_score" in text


def test_html_report_contains_combined_score():
    r = _make_report_with_combined()
    gen = ReportFactory.create_generator("html")
    html = gen.generate(r)
    assert "Overall Scores" in html
    assert "combined_score" in html
