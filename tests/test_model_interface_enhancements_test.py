import asyncio
from typing import Any, Dict, List

import types
import sys

from bench.evaluation import APIModel, LocalModel


def test_api_model_auth_modes_sync(monkeypatch):
    captured = {}

    class RespOK:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [{"ok": True}]

    def fake_post(url, json, headers=None, params=None, timeout=0, auth=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers or {}
        captured["params"] = params or {}
        captured["auth"] = auth
        return RespOK()

    monkeypatch.setattr("requests.post", fake_post)

    # bearer (default)
    m = APIModel(api_url="https://api.example/predict", api_key="K")
    out = m.predict([{"x": 1}])
    assert out and out[0].get("ok") is True
    assert captured["headers"]["Authorization"].startswith("Bearer ")

    # query param auth
    m2 = APIModel(
        api_url="https://api.example/predict",
        api_key="KQ",
        auth_mode="query",
        query_key="api_key",
    )
    out2 = m2.predict([{"x": 2}])
    assert out2 and out2[0].get("ok") is True
    assert captured["params"].get("api_key") == "KQ"

    # basic auth
    m3 = APIModel(
        api_url="https://api.example/predict",
        auth_mode="basic",
        basic_auth=("u", "p"),
    )
    out3 = m3.predict([{"x": 3}])
    assert out3 and out3[0].get("ok") is True
    assert captured["auth"] == ("u", "p")


def test_api_model_async_predict(monkeypatch):
    # Minimal httpx.AsyncClient mock
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"v": 1}]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return FakeResp()

    fake_httpx = types.SimpleNamespace(AsyncClient=FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    m = APIModel(api_url="https://api.example/predict")
    out = asyncio.get_event_loop().run_until_complete(
        m.async_predict([{"text": "hello"}])
    )
    assert isinstance(out, list) and out[0]["v"] == 1


def test_local_model_edge_cases():
    # Non-dict inputs should be coerced
    def fn(batch: List[Dict[str, Any]]):
        return [{"len": len(batch), "ok": True} for _ in batch]

    m = LocalModel(predict_fn=fn)
    out = m.predict(["a", {"x": 1}, 3])
    assert len(out) == 3 and all("ok" in x for x in out)

    # Empty inputs
    assert m.predict([]) == []
