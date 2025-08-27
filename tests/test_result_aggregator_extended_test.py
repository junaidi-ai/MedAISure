"""Extended tests for ResultAggregator additional features."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from bench.evaluation.result_aggregator import ResultAggregator
from bench.models.evaluation_result import EvaluationResult


@pytest.fixture
def agg(tmp_path):
    return ResultAggregator(output_dir=tmp_path)


@pytest.fixture
def run_with_results(agg: ResultAggregator):
    ts = datetime.now(timezone.utc)
    run_id = "run_a"
    # Two tasks with two metrics
    r1 = EvaluationResult(
        model_id="m",
        task_id="t1",
        inputs=[{"i": 1}],
        model_outputs=[{"o": 1}],
        metrics_results={"accuracy": 0.8, "f1": 0.7},
        metadata={},
        timestamp=ts,
    )
    r2 = EvaluationResult(
        model_id="m",
        task_id="t2",
        inputs=[{"i": 2}],
        model_outputs=[{"o": 2}],
        metrics_results={"accuracy": 0.9, "f1": 0.6},
        metadata={},
        timestamp=ts,
    )
    agg.add_evaluation_result(r1, run_id)
    agg.add_evaluation_result(r2, run_id)
    return run_id


def test_aggregate_statistics_basic(agg: ResultAggregator, run_with_results: str):
    stats = agg.aggregate_statistics(run_with_results)
    assert set(stats.keys()) == {"accuracy", "f1"}
    # mean and median across [0.8,0.9] and [0.7,0.6]
    assert stats["accuracy"]["mean"] == pytest.approx(0.85)
    assert stats["accuracy"]["median"] == pytest.approx(0.85)
    assert stats["f1"]["mean"] == pytest.approx(0.65)
    assert stats["f1"]["median"] == pytest.approx(0.65)


def test_aggregate_statistics_percentiles_and_task_filter(agg: ResultAggregator, run_with_results: str):
    # Filter to t1 only => values are just that task
    stats = agg.aggregate_statistics(run_with_results, percentiles=[0, 50, 100], tasks=["t1"])
    assert stats["accuracy"]["mean"] == pytest.approx(0.8)
    assert stats["accuracy"]["p0"] == pytest.approx(0.8)
    assert stats["accuracy"]["p50"] == pytest.approx(0.8)
    assert stats["accuracy"]["p100"] == pytest.approx(0.8)


def test_filter_and_sort_tasks(agg: ResultAggregator, run_with_results: str):
    rows = agg.filter_and_sort_tasks(run_with_results, metrics=["accuracy"], sort_by="accuracy", descending=False)
    # Should have t1 then t2 when ascending by accuracy (0.8, 0.9)
    assert [r["task_id"] for r in rows] == ["t1", "t2"]
    assert list(rows[0].keys()) == ["task_id", "accuracy"]


def test_exporters_csv_md_html(tmp_path: Path, agg: ResultAggregator, run_with_results: str):
    # CSV
    csv_path = tmp_path / "report.csv"
    agg.export_report_csv(run_with_results, csv_path)
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "row_type,task_id,accuracy,f1" in content
    assert "OVERALL" in content

    # Markdown
    md_path = tmp_path / "report.md"
    agg.export_report_markdown(run_with_results, md_path)
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "| Task ID |" in md
    assert "| OVERALL |" in md

    # HTML
    html_path = tmp_path / "report.html"
    agg.export_report_html(run_with_results, html_path)
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Benchmark Report" in html


def test_compare_runs_overall_and_per_task(tmp_path: Path, agg: ResultAggregator):
    # Create two runs with slightly different metrics
    ts = datetime.now(timezone.utc)
    run_a = "run_A"
    run_b = "run_B"

    for run_id, acc_t1, acc_t2 in [(run_a, 0.80, 0.90), (run_b, 0.85, 0.88)]:
        r1 = EvaluationResult(
            model_id="m",
            task_id="t1",
            inputs=[{"i": 1}],
            model_outputs=[{"o": 1}],
            metrics_results={"accuracy": acc_t1},
            metadata={},
            timestamp=ts,
        )
        r2 = EvaluationResult(
            model_id="m",
            task_id="t2",
            inputs=[{"i": 2}],
            model_outputs=[{"o": 2}],
            metrics_results={"accuracy": acc_t2},
            metadata={},
            timestamp=ts,
        )
        agg.add_evaluation_result(r1, run_id)
        agg.add_evaluation_result(r2, run_id)

    diff = agg.compare_runs(run_a, run_b, metrics=["accuracy"], relative=False)
    # overall_a = (0.8+0.9)/2 = 0.85, overall_b = (0.85+0.88)/2 = 0.865
    assert diff["overall"]["accuracy"] == pytest.approx(0.015)
    assert diff["per_task"]["t1"]["accuracy"] == pytest.approx(0.05)
    assert diff["per_task"]["t2"]["accuracy"] == pytest.approx(-0.02)


def test_plot_metric_distribution_data_only(agg: ResultAggregator, run_with_results: str):
    data = agg.plot_metric_distribution(run_with_results, "accuracy", output_path=None)
    assert data["metric"] == "accuracy"
    assert set(data["tasks"]) == {"t1", "t2"}
    assert len(data["values"]) == 2
