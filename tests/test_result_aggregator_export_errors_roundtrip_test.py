from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import json
import pytest

from bench.evaluation.result_aggregator import ResultAggregator
from bench.models.evaluation_result import EvaluationResult


def _make_agg_with_data(tmp_path: Path) -> tuple[ResultAggregator, str]:
    agg = ResultAggregator(output_dir=tmp_path)
    # Minimal evaluation result
    er = EvaluationResult(
        model_id="m1",
        task_id="t1",
        inputs=[{"x": 1}],
        model_outputs=[{"y": 2}],
        metrics_results={"accuracy": 0.8, "f1": 0.7},
        metadata={"model_name": "foo"},
    )
    agg.add_evaluation_result(er, run_id="run1")
    return agg, "run1"


def test_export_markdown_failure_raises_value_error(tmp_path: Path):
    agg, run_id = _make_agg_with_data(tmp_path)
    out = tmp_path / "out.md"

    with patch.object(Path, "write_text", side_effect=IOError("disk full")):
        with pytest.raises(ValueError) as ei:
            agg.export_report_markdown(run_id, out)
        assert "Failed to export Markdown" in str(ei.value)


def test_export_html_failure_raises_value_error(tmp_path: Path):
    agg, run_id = _make_agg_with_data(tmp_path)
    out = tmp_path / "out.html"

    with patch.object(Path, "write_text", side_effect=IOError("perm denied")):
        with pytest.raises(ValueError) as ei:
            agg.export_report_html(run_id, out)
        assert "Failed to export HTML" in str(ei.value)


def test_export_csv_failure_raises_value_error(tmp_path: Path):
    agg, run_id = _make_agg_with_data(tmp_path)
    out = tmp_path / "out.csv"

    # Patch built-in open used in the exporter
    with patch("builtins.open", side_effect=IOError("io")):
        with pytest.raises(ValueError) as ei:
            agg.export_report_csv(run_id, out)
        assert "Failed to export CSV" in str(ei.value)


def test_save_report_failure_raises_value_error(tmp_path: Path):
    agg, run_id = _make_agg_with_data(tmp_path)
    report = agg.get_report(run_id)
    out = tmp_path / "out.json"

    with patch.object(type(report), "save", side_effect=IOError("no space")):
        with pytest.raises(ValueError) as ei:
            agg.save_report(report, out)
        assert "Failed to save report" in str(ei.value)


def test_json_round_trip(tmp_path: Path):
    agg, run_id = _make_agg_with_data(tmp_path)

    # Save JSON via wrapper and load back
    json_path = agg.export_report_json(run_id, tmp_path / "rep.json")
    assert json_path.exists()

    # Validate file contents are valid JSON and re-loadable by BenchmarkReport
    data: Dict[str, Any]
    with open(json_path, "r") as f:
        data = json.load(f)
    assert "overall_scores" in data and "task_scores" in data

    # round-trip load
    from bench.models.benchmark_report import BenchmarkReport

    rep2 = BenchmarkReport.from_file(json_path)
    assert rep2.overall_scores == agg.get_report(run_id).overall_scores
