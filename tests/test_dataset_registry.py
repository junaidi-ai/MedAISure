from pathlib import Path

from bench.data.dataset_registry import DatasetRegistry
import bench.data.dataset_registry as dr


def test_default_registry_parses_and_contains_core():
    # Use the packaged default registry file
    pkg_dir = Path(dr.__file__).resolve().parent
    reg_path = pkg_dir / "datasets" / "registry.json"
    assert reg_path.exists(), "Packaged registry.json should exist"

    reg = DatasetRegistry(registry_path=reg_path)
    items = reg.list()
    ids = [m.id for m in items]
    assert "medaisure-core" in ids

    core = reg.get("medaisure-core")
    assert core.size == 200
    assert core.composition == {
        "diagnostics": 100,
        "summarization": 50,
        "communication": 50,
    }
    # composition sums to size ensured by validator
    assert sum(core.composition.values()) == core.size


def test_registry_crud_with_temp(tmp_path: Path):
    reg_path = tmp_path / "reg.json"
    reg = DatasetRegistry(registry_path=reg_path)
    # add
    core = reg.add(
        {
            "id": "tmp-ds",
            "name": "Temp",
            "size": 3,
            "composition": {"a": 1, "b": 2},
        }
    )
    assert core.id == "tmp-ds"
    # get
    fetched = reg.get("tmp-ds")
    assert fetched.name == "Temp"
    # update
    updated = reg.update("tmp-ds", {"name": "Temp2"})
    assert updated.name == "Temp2"
    # remove
    reg.remove("tmp-ds")
    try:
        reg.get("tmp-ds")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
