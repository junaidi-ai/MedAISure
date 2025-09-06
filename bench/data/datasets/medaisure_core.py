"""Loader stub for MedAISure-Core dataset.

This module exposes convenience accessors around the dataset registry entry
(`medaisure-core`) and provides lightweight hooks for validation and example
listings. It does not materialize the full dataset yet; instead, it focuses on
surface-level metadata and interfaces to be expanded later.

Exposed helpers:
- ``get_metadata()``: Return the pydantic-backed metadata for ``medaisure-core``
- ``get_composition()``: Category -> count mapping (e.g., diagnostics: 100)
- ``list_categories()``: Ordered list of category names
- ``validate()``: Best-effort validation hook (composition, categories, size)
- ``example_listing_rows()``: Placeholder rows imitating CLI list output
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..dataset_registry import get_default_registry


def get_metadata() -> Dict[str, object]:
    """Return registry metadata for ``medaisure-core``.

    Returns the underlying ``DatasetMeta`` as a plain dict via ``model_dump``.
    """
    reg = get_default_registry()
    meta = reg.get("medaisure-core")
    return meta.model_dump()


def load_examples(limit: Optional[int] = None) -> List[Dict[str, object]]:
    """Return a small placeholder list of example records.

    Real implementation would source examples from the task definitions or a
    curated store. This stub returns an empty list to keep interfaces stable.
    """
    # Placeholder: no direct dataset materialization yet
    return []


def get_composition() -> Dict[str, int]:
    """Return the composition map for ``medaisure-core``.

    Example:
        {"diagnostics": 100, "summarization": 50, "communication": 50}
    """
    reg = get_default_registry()
    meta = reg.get("medaisure-core")
    return dict(meta.composition or {})


def list_categories() -> List[str]:
    """Return a list of declared task categories for the core dataset."""
    reg = get_default_registry()
    meta = reg.get("medaisure-core")
    return list(meta.task_categories or [])


def validate() -> None:
    """Run best-effort validation checks on the registry entry.

    Current checks:
    - If ``size`` and ``composition`` are present, totals must match
    - All composition keys should be declared in ``task_categories``
    """
    reg = get_default_registry()
    meta = reg.get("medaisure-core")
    data = meta.model_dump()

    # Check size vs composition
    size = data.get("size")
    comp = data.get("composition") or {}
    if size is not None and comp:
        total = sum(int(v) for v in comp.values())
        if int(size) != total:
            raise ValueError(
                f"composition totals {total} but size is {size} for medaisure-core"
            )

    # Check composition categories are declared
    cats = set(data.get("task_categories") or [])
    missing = [k for k in comp.keys() if k not in cats]
    if missing:
        raise ValueError(
            f"composition has categories not declared in task_categories: {missing}"
        )


def example_listing_rows(limit: Optional[int] = 5) -> List[Dict[str, object]]:
    """Return placeholder task listing rows for documentation/examples.

    The real CLI pulls task listings from YAML/JSON files. Since the full core
    set is not materialized yet, this produces a minimal set of rows that match
    the structure returned by ``TaskLoader.list_available_tasks``.
    """
    categories = list_categories()
    comp = get_composition()
    rows: List[Dict[str, object]] = []
    # Create a few illustrative placeholder IDs per category
    for cat in categories:
        count = int(comp.get(cat, 0))
        # Use up to 2 placeholders per category to keep output concise
        n = min(2, count if count > 0 else 1)
        for i in range(1, n + 1):
            rows.append(
                {
                    "task_id": f"core_{cat}_{i:03d}",
                    "name": f"Core {cat.title()} Task {i}",
                    "description": f"Placeholder example for {cat} task {i}.",
                    "metrics": [],
                    "num_examples": 0,
                    "file": None,
                    "difficulty": None,
                }
            )
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows
