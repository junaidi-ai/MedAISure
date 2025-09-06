#!/usr/bin/env python3
"""Run a sample `evaluate --export-submission` with a dummy harness and print stdout.

This avoids heavy model loads by monkeypatching EvaluationHarness inside bench.cli_typer
with a lightweight DummyHarness that returns a small BenchmarkReport.
"""

from __future__ import annotations

import os
import sys
import io
from contextlib import redirect_stdout
from pathlib import Path

# Ensure plain output for docs
os.environ["MEDAISURE_NO_RICH"] = "1"

# Workspace paths
ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    sys.path.insert(0, str(ROOT))

    # Import inside function to ensure path is set
    from bench.models.benchmark_report import BenchmarkReport
    from bench.models.evaluation_result import EvaluationResult
    import bench.cli_typer as cli

    # Dummy harness that returns a tiny report
    class DummyHarness:
        def __init__(self, tasks_dir: str, results_dir: str, on_progress=None):
            self.tasks_dir = tasks_dir
            self.results_dir = results_dir
            self.on_progress = on_progress

        def evaluate(self, *args, **kwargs):
            er = EvaluationResult(
                model_id="dummy-model",
                task_id="t1",
                inputs=[{"id": "1"}],
                model_outputs=[{"prediction": "yes", "reasoning": "r1"}],
                metrics_results={"accuracy": 1.0},
            )
            br = BenchmarkReport(model_id="dummy-model")
            br.add_evaluation_result(er)
            br.metadata["run_id"] = "run-sample"
            return br

    # Patch the harness used by CLI
    cli.EvaluationHarness = DummyHarness  # type: ignore[attr-defined]

    # Prepare args: evaluate with export-submission
    args = [
        "evaluate",
        "dummy-model",
        "--tasks",
        "t1",
        "--tasks-dir",
        str(ROOT / "bench" / "tasks"),  # not used by DummyHarness
        "--model-type",
        "local",
        "--output-dir",
        str(ROOT / "results"),
        "--format",
        "json",
        "--save-results",
        "--export-submission",
        str(ROOT / "docs" / "snippets" / "_tmp_submission.json"),
        "--export-submission-include-reasoning",
    ]

    # Capture stdout while running the app
    f = io.StringIO()
    with redirect_stdout(f):
        cli.app(args=args)  # type: ignore[arg-type]
    out = f.getvalue()

    # Print to stdout so the caller captures it
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
