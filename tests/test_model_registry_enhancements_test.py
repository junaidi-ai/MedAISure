from typing import Any, Dict, List

from bench.evaluation import ModelInterface, ModelRegistry


def test_model_registry_versioning_and_metadata(tmp_path):
    class Dummy(ModelInterface):
        def __init__(self, _id: str, meta: Dict[str, Any]):
            self._id = _id
            self._meta = meta

        def predict(self, inputs: List[Dict[str, Any]]):
            return [{"ok": True} for _ in inputs]

        @property
        def model_id(self) -> str:
            return self._id

        @property
        def metadata(self) -> Dict[str, Any]:
            return self._meta

    reg = ModelRegistry()

    m_v1 = Dummy("m", {"ver": 1})
    m_v2 = Dummy("m", {"ver": 2})

    reg.register_model(m_v1, version="v1")
    reg.register_model(m_v2, version="v2")

    # Latest should be v2
    latest = reg.get_model("m")
    assert latest is m_v2

    # Version-specific retrieval
    assert reg.get_model("m", version="v1") is m_v1
    assert reg.get_model("m", version="v2") is m_v2

    # Listing
    ids = reg.list_models()
    assert "m" in ids
    with_versions = reg.list_models(include_versions=True)
    assert set(with_versions["m"]) == {"v1", "v2"}

    # Metadata query
    assert reg.get_metadata("m").get("ver") == 2
    assert reg.get_metadata("m", version="v1").get("ver") == 1

    # Default config
    reg.set_default_config("m", {"batch_size": 8})
    assert reg.get_default_config("m")["batch_size"] == 8

    # Persistence (topology and configs only)
    save_path = tmp_path / "registry.json"
    reg.save_registry(str(save_path))

    reg2 = ModelRegistry()
    reg2.load_registry(str(save_path))

    # After load, topology exists but models must be re-registered
    assert "m" in reg2.list_models(include_versions=True)
    assert reg2.get_default_config("m")["batch_size"] == 8
    # No model object yet
    assert reg2.get_model("m") is None
