#!/usr/bin/env python3
"""
Insert a brief "Performance tips" markdown cell into a Jupyter notebook.

Usage:
  python scripts/insert_perf_tip_cell.py path/to/notebook.ipynb

This script is idempotent: if a cell with the unique marker already exists,
no duplicate will be added.
"""

import sys
from pathlib import Path

try:
    import nbformat as nbf
except Exception:
    print("ERROR: nbformat is required. Install with: pip install nbformat")
    sys.exit(1)

MARKER = "[Performance Tips — MedAISure]"
CELL_SOURCE = (
    "## Performance tips\n\n"
    f"See the comprehensive guide: {MARKER} (docs/04c_performance_tips.md).\n\n"
    "Highlights:\n"
    "- Use smaller batch sizes on constrained hardware.\n"
    "- Enable caching (`use_cache=True`) to avoid recomputation.\n"
    "- For HuggingFace, set appropriate `device` and `torch_dtype` (e.g., 'float16'/'bfloat16').\n"
    "- Limit `max_new_tokens` and sample size when iterating.\n"
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/insert_perf_tip_cell.py path/to/notebook.ipynb")
        sys.exit(2)

    nb_path = Path(sys.argv[1])
    if not nb_path.exists():
        print(f"ERROR: Notebook not found: {nb_path}")
        sys.exit(3)

    nb = nbf.read(nb_path, as_version=4)

    # Check if marker already present
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown" and MARKER in "".join(
            cell.get("source", [])
        ):
            print("Performance tips cell already present. No changes made.")
            break
    else:
        # Append new markdown cell at the end
        md = nbf.v4.new_markdown_cell(CELL_SOURCE)
        nb.cells.append(md)
        nbf.write(nb, nb_path)
        print(f"Inserted performance tips cell into: {nb_path}")


if __name__ == "__main__":
    main()
