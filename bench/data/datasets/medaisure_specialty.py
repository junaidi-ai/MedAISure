"""Loader for MedAISure-Specialty dataset.

Provides convenience accessors around the dataset registry entry
(`medaisure-specialty`) and validation hooks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..dataset_registry import get_default_registry


def get_metadata() -> Dict[str, object]:
    reg = get_default_registry()
    meta = reg.get("medaisure-specialty")
    return meta.model_dump()


def load_examples(limit: Optional[int] = None) -> List[Dict[str, object]]:
    # Placeholder examples not materialized
    return []


def get_composition() -> Dict[str, int]:
    reg = get_default_registry()
    meta = reg.get("medaisure-specialty")
    return dict(meta.composition or {})


def list_categories() -> List[str]:
    reg = get_default_registry()
    meta = reg.get("medaisure-specialty")
    return list(meta.task_categories or [])


def validate() -> None:
    reg = get_default_registry()
    meta = reg.get("medaisure-specialty")
    data = meta.model_dump()

    size = data.get("size")
    comp = data.get("composition") or {}
    if size is not None and comp:
        total = sum(int(v) for v in comp.values())
        if int(size) != total:
            raise ValueError(
                f"composition totals {total} but size is {size} for medaisure-specialty"
            )

    cats = set(data.get("task_categories") or [])
    missing = [k for k in comp.keys() if k not in cats]
    if missing:
        raise ValueError(
            f"composition has categories not declared in task_categories: {missing}"
        )


def example_listing_rows(limit: Optional[int] = 5) -> List[Dict[str, object]]:
    cats = list_categories()
    comp = get_composition()
    rows: List[Dict[str, object]] = []
    for cat in cats:
        count = int(comp.get(cat, 0))
        n = min(2, count if count > 0 else 1)
        for i in range(1, n + 1):
            rows.append(
                {
                    "task_id": f"specialty_{cat}_{i:03d}",
                    "name": f"Specialty {cat.title()} Task {i}",
                    "description": f"Placeholder example for specialty {cat} task {i}.",
                    "metrics": [],
                    "num_examples": 0,
                    "file": None,
                    "difficulty": None,
                }
            )
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows
