from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from bench.evaluation import (
    ModelRunner,
    ModelRegistry,
    LocalModel,
    HuggingFaceModel,
    APIModel,
)


def _inputs_two() -> List[Dict[str, Any]]:
    return [{"text": "patient has fever"}, {"text": "patient has cough"}]


def test_registry_versioning_to_runner_interface_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two LocalModel variants under the same logical id via ModelRegistry versions
    def predict_v1(batch: List[Dict[str, Any]]):
        return [{"label": "fever", "score": 1.0} for _ in batch]

    def predict_v2(batch: List[Dict[str, Any]]):
        return [{"label": "cough", "score": 1.0} for _ in batch]

    m_v1 = LocalModel(predict_fn=predict_v1, model_id="local-cls")
    m_v2 = LocalModel(predict_fn=predict_v2, model_id="local-cls")

    reg = ModelRegistry()
    reg.register_model(m_v1, version="v1")
    reg.register_model(m_v2, version="v2")

    mr = ModelRunner()

    # Retrieve v1 and register into runner as interface
    selected = reg.get_model("local-cls", version="v1")
    assert selected is not None
    mr.register_interface_model(selected)
    out_v1 = mr.run_model("local-cls", inputs=_inputs_two(), batch_size=2)
    assert len(out_v1) == 2
    assert {o.get("label") for o in out_v1} == {"fever"}

    # Switch to v2 by unloading and registering the other version
    mr.unload_model("local-cls")
    selected = reg.get_model("local-cls", version="v2")
    assert selected is not None
    mr.register_interface_model(selected)
    out_v2 = mr.run_model("local-cls", inputs=_inputs_two(), batch_size=2)
    assert len(out_v2) == 2
    assert {o.get("label") for o in out_v2} == {"cough"}


def test_consistent_output_across_interface_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LocalModel
    local = LocalModel(
        predict_fn=lambda b: [
            {
                "label": ("fever" if "fever" in i.get("text", "") else "cough"),
                "score": 1.0,
            }
            for i in b
        ],
        model_id="if-local",
    )

    # Mocked HuggingFaceModel
    # Build a fake transformers with a pipeline that returns label dicts
    import types
    import sys

    def fake_pipeline(task, model=None, tokenizer=None, device=-1, **kwargs):
        def _call(texts):
            outs = []
            for t in texts:
                outs.append(
                    {"label": "fever" if "fever" in t else "cough", "score": 0.9}
                )
            return outs

        return _call

    fake_tf = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        AutoModelForSequenceClassification=types.SimpleNamespace(
            from_pretrained=lambda *a, **k: object()
        ),
        pipeline=fake_pipeline,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)  # type: ignore[arg-type]

    hf = HuggingFaceModel(
        model_name="dummy-cls", hf_task="text-classification", model_id="if-hf"
    )

    # Mocked APIModel
    api = APIModel(
        api_url="https://api.example/predict", model_id="if-api", api_key="k"
    )

    class Resp:
        def __init__(self, status_code: int, data: Any) -> None:
            self.status_code = status_code
            self._data = data
            self.text = json.dumps(data)

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("bad status")

        def json(self) -> Any:
            return self._data

    def fake_post(url, json=None, headers=None, params=None, timeout=None, auth=None):  # type: ignore[override]
        # Return list of outputs matching input length
        data = [
            {
                "label": "fever" if "fever" in (x.get("text", "")) else "cough",
                "score": 0.8,
            }
            for x in json
        ]
        return Resp(200, data)

    import requests

    monkeypatch.setattr(requests, "post", fake_post)

    # Registry and runner
    reg = ModelRegistry()
    for m in (local, hf, api):
        reg.register_model(m)

    mr = ModelRunner()
    # Register all three into runner via interface path
    for mid in reg.list_models():
        m = reg.get_model(mid)
        assert m is not None
        mr.register_interface_model(m)

    # Run each model and verify consistent formats
    inputs = _inputs_two()
    out_local = mr.run_model("if-local", inputs, batch_size=2)
    out_hf = mr.run_model("if-hf", inputs, batch_size=2)
    out_api = mr.run_model("if-api", inputs, batch_size=2)

    for outs in (out_local, out_hf, out_api):
        assert len(outs) == 2
        # Each output is a dict with 'label' and either has score or prediction, but we expect score here
        assert all(isinstance(o, dict) for o in outs)
        assert all("label" in o for o in outs)

    # Labels should align with input semantics across all types
    expected = ["fever", "cough"]
    assert [o.get("label") for o in out_local] == expected
    assert [o.get("label") for o in out_hf] == expected
    assert [o.get("label") for o in out_api] == expected


def test_interface_error_propagation_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    # predict_fn raises on second item; runner should catch and return empty dicts for the batch
    def bad_predict(batch: List[Dict[str, Any]]):
        outs = []
        for idx, _ in enumerate(batch):
            if idx == 1:
                raise RuntimeError("boom")
            outs.append({"label": "ok", "score": 1.0})
        return outs

    local = LocalModel(predict_fn=bad_predict, model_id="if-bad")
    mr = ModelRunner()
    mr.register_interface_model(local)

    inputs = _inputs_two()
    outs = mr.run_model("if-bad", inputs, batch_size=2)
    # All outputs become empty dicts for the failed batch per runner error handling
    assert len(outs) == 2
    assert outs == [{}, {}]


def test_model_switching_via_registry_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same logical model id with two behaviors; verify selecting via registry version yields different predictions
    m1 = LocalModel(
        predict_fn=lambda b: [{"label": "A"} for _ in b], model_id="sw-model"
    )
    m2 = LocalModel(
        predict_fn=lambda b: [{"label": "B"} for _ in b], model_id="sw-model"
    )
    reg = ModelRegistry()
    reg.register_model(m1, version="1.0")
    reg.register_model(m2, version="2.0")

    mr = ModelRunner()
    mr.register_interface_model(reg.get_model("sw-model", version="1.0"))  # type: ignore[arg-type]
    out1 = mr.run_model("sw-model", inputs=_inputs_two(), batch_size=2)
    assert {o.get("label") for o in out1} == {"A"}

    mr.unload_model("sw-model")
    mr.register_interface_model(reg.get_model("sw-model", version="2.0"))  # type: ignore[arg-type]
    out2 = mr.run_model("sw-model", inputs=_inputs_two(), batch_size=2)
    assert {o.get("label") for o in out2} == {"B"}


def _select_with_fallback(
    reg: ModelRegistry, model_id: str, primary: str, fallback: str
):
    m = reg.get_model(model_id, version=primary)
    if m is None:
        m = reg.get_model(model_id, version=fallback)
    return m


def test_registry_version_fallback_selects_available_version() -> None:
    # Only v1 exists; request v2 then fall back to v1
    m_v1 = LocalModel(
        predict_fn=lambda b: [{"label": "V1"} for _ in b], model_id="fb-model"
    )
    reg = ModelRegistry()
    reg.register_model(m_v1, version="v1")

    selected = _select_with_fallback(reg, "fb-model", primary="v2", fallback="v1")
    assert selected is not None and selected.model_id == "fb-model"

    mr = ModelRunner()
    mr.register_interface_model(selected)
    outs = mr.run_model("fb-model", inputs=_inputs_two(), batch_size=2)
    assert {o.get("label") for o in outs} == {"V1"}


def test_registry_version_fallback_prefers_primary_if_available() -> None:
    # Both v1 and v2 exist; ensure primary takes precedence
    m_v1 = LocalModel(
        predict_fn=lambda b: [{"label": "V1"} for _ in b], model_id="fb2-model"
    )
    m_v2 = LocalModel(
        predict_fn=lambda b: [{"label": "V2"} for _ in b], model_id="fb2-model"
    )
    reg = ModelRegistry()
    reg.register_model(m_v1, version="v1")
    reg.register_model(m_v2, version="v2")

    selected = _select_with_fallback(reg, "fb2-model", primary="v2", fallback="v1")
    assert selected is not None and selected.model_id == "fb2-model"

    mr = ModelRunner()
    mr.register_interface_model(selected)
    outs = mr.run_model("fb2-model", inputs=_inputs_two(), batch_size=2)
    assert {o.get("label") for o in outs} == {"V2"}
