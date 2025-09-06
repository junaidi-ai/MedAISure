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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
