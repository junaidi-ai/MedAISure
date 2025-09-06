from pathlib import Path

import pytest

from bench.data.dataset_registry import DatasetRegistry
from bench.data.datasets import medaisure_core as core
from bench.data.datasets import medaisure_hard as hard
from bench.data.datasets import medaisure_specialty as spec


def test_registry_contains_new_datasets():
    reg = DatasetRegistry()
    ids = [d.id for d in reg.list()]
    assert "medaisure-hard" in ids
    assert "medaisure-specialty" in ids


def test_registry_composition_and_sizes_sum_correctly():
    reg = DatasetRegistry()
    # core
    m_core = reg.get("medaisure-core")
    assert m_core.size == 200
    assert sum(m_core.composition.values()) == m_core.size
    # hard
    m_hard = reg.get("medaisure-hard")
    assert m_hard.size == 60
    assert sum(m_hard.composition.values()) == m_hard.size
    # specialty
    m_spec = reg.get("medaisure-specialty")
    assert m_spec.size == 80
    assert sum(m_spec.composition.values()) == m_spec.size


@pytest.mark.parametrize(
    "mod, dsid",
    [
        (core, "medaisure-core"),
        (hard, "medaisure-hard"),
        (spec, "medaisure-specialty"),
    ],
)
def test_loader_validate_and_helpers(mod, dsid):
    # validate() should not raise
    mod.validate()
    # composition keys subset of categories
    comp = mod.get_composition()
    cats = set(mod.list_categories())
    assert set(comp.keys()).issubset(cats)
    # metadata returns correct id
    meta = mod.get_metadata()
    assert meta["id"] == dsid
    # example listing rows shape
    rows = mod.example_listing_rows(limit=3)
    assert isinstance(rows, list)
    if rows:
        keys = sorted(rows[0].keys())
        assert keys == [
            "description",
            "difficulty",
            "file",
            "metrics",
            "name",
            "num_examples",
            "task_id",
        ]


def test_specialty_tasks_exist_and_loadable():
    # Ensure new specialty task YAMLs exist
    tasks_dir = Path("bench/tasks")
    for fname in [
        "diagnostic_reasoning_neurology.yaml",
        "diagnostic_reasoning_oncology.yaml",
        "diagnostic_reasoning_radiology.yaml",
        "diagnostic_reasoning_rheumatology.yaml",
        "diagnostic_reasoning_endocrinology.yaml",
        "diagnostic_reasoning_gastroenterology.yaml",
    ]:
        p = tasks_dir / fname
        assert p.exists(), f"Missing task file: {p}"
