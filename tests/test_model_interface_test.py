import types
import sys
from typing import Any, Dict, List


from bench.evaluation import (
    APIModel,
    HuggingFaceModel,
    LocalModel,
    ModelInterface,
    ModelRegistry,
    ModelRunner,
)


def test_local_model_init_with_predict_fn_and_prediction_shape():
    def predict_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"label": "OK", "score": 1.0, "i": i} for i, _ in enumerate(batch)]

    m = LocalModel(predict_fn=predict_fn, model_id="local-ok")
    inputs = [{"text": "a"}, {"text": "b"}]
    out = m.predict(inputs)

    assert isinstance(out, list)
    assert len(out) == len(inputs)
    assert all(isinstance(x, dict) for x in out)
    assert out[0]["label"] == "OK"
    assert out[1]["i"] == 1
    assert m.model_id == "local-ok"


def test_local_model_loader_predict_method_used_and_error_handling():
    class Dummy:
        def predict(self, batch: List[Dict[str, Any]]):
            return [{"ok": True} for _ in batch]

    def loader(_path: str):
        return Dummy()

    m = LocalModel(
        model_path="/tmp/whatever.bin", loader=loader, model_id="local-dummy"
    )
    out = m.predict([{"x": 1}, {"x": 2}])
    assert len(out) == 2
    assert out[0] == {"ok": True}

    # Force error path: loader returns non-callable/no predict object and no predict_fn
    m_bad = LocalModel(model_path="/tmp/whatever.bin", loader=lambda p: object())
    out_bad = m_bad.predict([{"x": 1}, {"x": 2}, {"x": 3}])
    assert len(out_bad) == 3
    assert all(isinstance(x, dict) for x in out_bad)


def test_api_model_success_and_error(monkeypatch):
    calls = {"n": 0}

    class RespOK:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [{"label": "Y", "score": 0.9}]

    def fake_post(url, json, headers=None, timeout=0, **kwargs):
        calls["n"] += 1
        return RespOK()

    monkeypatch.setattr("requests.post", fake_post)

    m = APIModel(api_url="https://api.example/predict", api_key="k", model_id="api-x")
    out = m.predict([{"text": "abc"}])
    assert len(out) == 1 and out[0]["label"] == "Y"
    assert calls["n"] == 1

    # Error path
    def fake_post_err(*args, **kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr("requests.post", fake_post_err)
    out_err = m.predict([{"text": "abc"}, {"text": "def"}])
    assert len(out_err) == 2
    assert all(isinstance(x, dict) for x in out_err)


def test_hf_model_pipeline_mocked(monkeypatch):
    # Mock transformers.pipeline to avoid downloads and heavy deps
    class Pipe:
        def __init__(self, task, **kwargs):
            self.task = task
            self.model = types.SimpleNamespace(
                config=types.SimpleNamespace(id2label={0: "NEG", 1: "POS"})
            )

        def __call__(self, texts):
            if self.task == "text-classification":
                return [{"label": "POS", "score": 0.8} for _ in texts]
            if self.task == "summarization":
                return [{"summary_text": f"sum:{t[:5]}"} for t in texts]
            if self.task == "text-generation":
                return [{"generated_text": f"gen:{t[:5]}"} for t in texts]
            return [str(t) for t in texts]

    def fake_pipeline(task, *args, **kwargs):
        return Pipe(task)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return object()

    def fake_import(name):
        if name == "transformers":
            mod = types.SimpleNamespace(
                AutoTokenizer=FakeAutoTokenizer,
                pipeline=fake_pipeline,
            )
            return mod
        return __import__(name)

    monkeypatch.setitem(sys.modules, "transformers", fake_import("transformers"))

    # Now create HF model and predict for each task
    for task, key in (
        ("text-classification", "label"),
        ("summarization", "summary"),
        ("text-generation", "text"),
    ):
        m = HuggingFaceModel("tiny-model", hf_task=task, model_id=f"hf-{task}")
        out = m.predict([{"text": "hello world"}])
        assert isinstance(out, list) and len(out) == 1
        assert key in out[0]


def test_model_registry_register_and_get():
    reg = ModelRegistry()

    class Echo(ModelInterface):
        def __init__(self, _id: str):
            self._id = _id

        def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [{"echo": i} for i in inputs]

        @property
        def model_id(self) -> str:
            return self._id

    e = Echo("echo-1")
    reg.register_model(e)
    assert reg.get_model("echo-1") is e
    assert "echo-1" in reg.list_models()


def test_model_runner_register_interface_and_run():
    class Constant(ModelInterface):
        def __init__(self, val: str, _id: str = "const"):
            self.val = val
            self._id = _id

        def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [{"value": self.val} for _ in inputs]

        @property
        def model_id(self) -> str:
            return self._id

    runner = ModelRunner()
    c = Constant("X", _id="c1")
    runner.load_model("c1", model_type="interface", model=c)
    out = runner.run_model("c1", inputs=[{"a": 1}, {"a": 2}], batch_size=2)
    assert len(out) == 2 and out[0]["value"] == "X"
