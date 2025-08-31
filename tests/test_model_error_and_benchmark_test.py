import asyncio
from typing import Any, Dict, List

import types
import sys

import pytest

from bench.evaluation import APIModel, LocalModel


def test_api_model_retries_success(monkeypatch):
    calls = {"n": 0}

    class Resp:
        def __init__(self, code, payload=None):
            self.status_code = code
            self._payload = payload or [{"ok": True}]

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    def fake_post(url, json, headers=None, params=None, timeout=0, auth=None, **kwargs):
        calls["n"] += 1
        # First attempt -> 500, second -> 200
        if calls["n"] == 1:
            return Resp(500)
        return Resp(200)

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=fake_post))

    m = APIModel(api_url="https://api.example/predict", max_retries=2, backoff_factor=0)
    out = m.predict([{"x": 1}])
    assert out and out[0].get("ok") is True
    assert calls["n"] == 2  # retried once after 500


def test_api_model_all_failures_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise TimeoutError("timeout")

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=fake_post))

    m = APIModel(api_url="https://api.example/predict", max_retries=2, backoff_factor=0)
    out = m.predict([{"x": 1}, {"x": 2}])
    assert out == [{}, {}]


def test_api_model_async_retry_on_status(monkeypatch):
    # Simulate httpx AsyncClient where first call fails with response.status_code in retry set
    class FakeResp:
        def __init__(self, ok=True):
            self._ok = ok
            self.status_code = 200 if ok else 503

        def raise_for_status(self):
            if not self._ok:
                e = Exception("error")
                # attach response-like object with status_code
                e.response = types.SimpleNamespace(status_code=self.status_code)
                raise e

        def json(self):
            return [{"ok": True}]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                r = FakeResp(ok=False)
                # cause raise_for_status to raise
                try:
                    r.raise_for_status()
                except Exception as e:
                    raise e
            return FakeResp(ok=True)

    fake_httpx = types.SimpleNamespace(AsyncClient=FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    m = APIModel(api_url="https://api.example/predict", max_retries=2, backoff_factor=0)
    out = asyncio.get_event_loop().run_until_complete(m.async_predict([{"x": 1}]))
    assert out and out[0].get("ok") is True


def test_metadata_presence_api_and_local(tmp_path):
    # APIModel metadata is not explicitly defined, but interface allows empty dict.
    # We validate that LocalModel has framework/local and model_id.
    def fn(batch: List[Dict[str, Any]]):
        return [{"n": i} for i, _ in enumerate(batch)]

    lm = LocalModel(predict_fn=fn, model_id="lm-1")
    md = lm.metadata
    assert md.get("framework") == "local" and md.get("model_id") == "lm-1"

    am = APIModel(api_url="https://api.example/predict")
    # APIModel has no custom metadata method; ensure property exists and is a dict
    assert isinstance(am.metadata, dict)


@pytest.mark.benchmark(group="models")
def test_benchmark_localmodel_predict_throughput(benchmark):
    def fn(batch: List[Dict[str, Any]]):
        # trivial compute
        return [{"s": sum(len(str(v)) for v in d.values())} for d in batch]

    m = LocalModel(predict_fn=fn)
    inputs = [{"a": "x" * 16} for _ in range(256)]

    def run():
        out = m.predict(inputs)
        assert len(out) == len(inputs)

    benchmark(run)
