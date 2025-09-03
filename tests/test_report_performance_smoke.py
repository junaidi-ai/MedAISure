from __future__ import annotations

import time
from pathlib import Path

from bench.models.benchmark_report import BenchmarkReport
from bench.models.evaluation_result import EvaluationResult
from bench.reports import (
    JSONReportGenerator,
    MarkdownReportGenerator,
    HTMLReportGenerator,
    CSVReportGenerator,
)


def _make_large_report(n_tasks: int = 200, n_metrics: int = 3) -> BenchmarkReport:
    overall = {f"metric_{i}": i / (n_metrics + 1) for i in range(1, n_metrics + 1)}
    task_scores = {
        f"task_{t}": {
            f"metric_{i}": (i + t) / (n_metrics + t + 1)
            for i in range(1, n_metrics + 1)
        }
        for t in range(n_tasks)
    }
    results: list[EvaluationResult] = []
    for t in range(n_tasks):
        metrics_results = {
            f"metric_{i}": (i + t) / (n_metrics + t + 2)
            for i in range(1, n_metrics + 1)
        }
        res = EvaluationResult(
            model_id="perf-model",
            task_id=f"task_{t}",
            inputs=[{"q": f"question {t}", "idx": t}],
            model_outputs=[{"a": f"answer {t}", "score": 0.5 + (t % 10) / 20.0}],
            metrics_results=metrics_results,
        )
        results.append(res)
    return BenchmarkReport(
        model_id="perf-model",
        overall_scores=overall,
        task_scores=task_scores,
        detailed_results=results,
        metadata={"run_id": "perf-000"},
    )


def test_report_generators_performance_smoke(tmp_path: Path) -> None:
    """Smoke performance test: ensure generators run and produce outputs quickly.

    Keeps runtime small to be CI-friendly. Not a strict benchmark.
    """
    br = _make_large_report(n_tasks=150, n_metrics=4)

    # JSON
    t0 = time.time()
    jgen = JSONReportGenerator()
    jdata = jgen.generate(br)
    jgen.validate(jdata)
    assert isinstance(jdata, dict) and jdata.get("model_id") == "perf-model"
    # Use generator save to handle non-JSON-native types like datetime
    jgen.save(jdata, tmp_path / "report.json")
    # End time capture not needed; we measure total_time at the end

    # Markdown
    mgen = MarkdownReportGenerator()
    mdata = mgen.generate(br)
    mgen.validate(mdata)
    assert isinstance(mdata, str) and "Benchmark Report" in mdata
    (tmp_path / "report.md").write_text(mdata)

    # HTML
    hgen = HTMLReportGenerator()
    hdata = hgen.generate(br)
    hgen.validate(hdata)
    assert isinstance(hdata, str) and "<html" in hdata.lower()
    (tmp_path / "report.html").write_text(hdata)

    # CSV (dir)
    cgen = CSVReportGenerator()
    cdata = cgen.generate(br)
    cgen.validate(cdata)
    outdir = tmp_path / "csv"
    cgen.save(cdata, outdir)
    assert (outdir / "task_scores.csv").exists()

    total_time = time.time() - t0
    # JSON generation should be very fast; overall run should be well under a few seconds
    assert total_time < 5.0, f"Report generation too slow: {total_time:.2f}s"
