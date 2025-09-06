#!/usr/bin/env python3
"""Capture CLI output for docs and write to snippets files.

This script generates deterministic plain-text output by disabling Rich rendering
(MEDAISURE_NO_RICH=1) and writes the results to files under docs/snippets/.

Run:
    python docs/scripts/capture_cli.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNIPPETS = ROOT / "docs" / "snippets"


def run_cmd(args: list[str]) -> str:
    env = dict(os.environ)
    env["MEDAISURE_NO_RICH"] = "1"
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return proc.stdout


def main() -> int:
    SNIPPETS.mkdir(parents=True, exist_ok=True)

    # 1) list-datasets with composition
    out = run_cmd(
        ["python", "-m", "bench.cli_typer", "list-datasets", "--with-composition"]
    )
    (SNIPPETS / "list_datasets_with_composition.txt").write_text(out)

    # 2) generate-submission --help
    out = run_cmd(["python", "-m", "bench.cli_typer", "generate-submission", "--help"])
    (SNIPPETS / "generate_submission_help.txt").write_text(out)

    # 3) evaluate --help
    out = run_cmd(["python", "-m", "bench.cli_typer", "evaluate", "--help"])
    (SNIPPETS / "evaluate_help.txt").write_text(out)

    # 4) Sample evaluate --export-submission run (monkeypatched harness)
    out = run_cmd(
        ["python", str(ROOT / "docs" / "scripts" / "capture_sample_evaluate_export.py")]
    )
    (SNIPPETS / "evaluate_export_sample.txt").write_text(out)

    # 5) generate-submission error (missing run-id)
    # Use a temp results dir with no files to provoke a clear error message
    empty_results = ROOT / "docs" / "snippets" / "_empty_results"
    empty_results.mkdir(parents=True, exist_ok=True)
    try:
        out = run_cmd(
            [
                "python",
                "-m",
                "bench.cli_typer",
                "generate-submission",
                "--run-id",
                "nonexistent",
                "--results-dir",
                str(empty_results),
                "--out",
                str(SNIPPETS / "_tmp_submission.json"),
            ]
        )
    except Exception:
        # The command exits non-zero; capture the combined stdout via the exception string if needed.
        # But subprocess.run with check=True already puts stdout in the exception.
        # To ensure we have output, rebuild by calling without check.
        import subprocess

        env = dict(os.environ)
        env["MEDAISURE_NO_RICH"] = "1"
        proc = subprocess.run(
            [
                "python",
                "-m",
                "bench.cli_typer",
                "generate-submission",
                "--run-id",
                "nonexistent",
                "--results-dir",
                str(empty_results),
                "--out",
                str(SNIPPETS / "_tmp_submission.json"),
            ],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        out = proc.stdout
    (SNIPPETS / "generate_submission_error.txt").write_text(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
