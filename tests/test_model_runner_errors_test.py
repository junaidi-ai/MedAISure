"""Additional tests for ModelRunner error/retry, async, and batching edge cases."""

from __future__ import annotations

import asyncio
from typing import List

import pytest
import requests

from bench.evaluation.model_runner import ModelRunner


@pytest.fixture
def model_runner() -> ModelRunner:
    return ModelRunner()


class FlakyAPI:
    """Simulate an API endpoint that fails first, then succeeds."""

    def __init__(self, fail_times: int, payload: List[dict]):
        self.calls = 0
        self.fail_times = fail_times
        self.payload = payload

    def __call__(self, *args, **kwargs):  # requests.post replacement
        class Resp:
            def __init__(self, status_code: int, json_data=None):
                self.status_code = status_code
                self._json = json_data

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

            def json(self):
                if isinstance(self._json, Exception):
                    raise self._json
                return self._json

        self.calls += 1
        if self.calls <= self.fail_times:
            return Resp(500, json_data=None)
        return Resp(200, json_data=self.payload)


def test_api_model_retries_and_backoff(monkeypatch: pytest.MonkeyPatch):
    """API model should retry up to max_retries and then succeed."""
    mr = ModelRunner()

    # Register API model with retries
    mr.load_model(
        "api_retry",
        model_type="api",
        api_key="k",
        endpoint="https://api.example.com/predict",
        max_retries=2,
        backoff_factor=0.0,  # no sleep to speed up test
    )

    flaky = FlakyAPI(fail_times=1, payload=[{"label": "ok", "score": 1.0}])

    def fake_post(**kwargs):
        return flaky()

    # requests is imported inside _call_api_model, patch the global requests.post
    monkeypatch.setattr("requests.post", lambda *a, **kw: fake_post())

    # Run, first call fails, second succeeds
    out = mr.run_model("api_retry", inputs=[{"text": "x"}], batch_size=1)
    assert out and out[0]["label"] == "ok" and out[0]["score"] == 1.0


def test_api_model_exhausts_retries_returns_empty(monkeypatch: pytest.MonkeyPatch):
    """If all retries fail, should return empty list for that batch."""
    mr = ModelRunner()

    mr.load_model(
        "api_fail",
        model_type="api",
        api_key="k",
        endpoint="https://api.example.com/predict",
        max_retries=1,
        backoff_factor=0.0,
    )

    flaky = FlakyAPI(fail_times=10, payload=[])

    def fake_post(**kwargs):
        return flaky()

    # requests is imported inside _call_api_model, patch the global requests.post
    monkeypatch.setattr("requests.post", lambda *a, **kw: fake_post())

    out = mr.run_model("api_fail", inputs=[{"text": "x"}], batch_size=1)
    assert out == []


def test_api_model_invalid_json_returns_formatted(monkeypatch: pytest.MonkeyPatch):
    """If response.json raises, we should catch inside run batch and return empty for that batch via retry exhaustion."""
    mr = ModelRunner()

    mr.load_model(
        "api_bad_json",
        model_type="api",
        api_key="k",
        endpoint="https://api.example.com/predict",
        max_retries=0,
    )

    class BadJSON:
        def __call__(self, *args, **kwargs):
            class Resp:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    raise ValueError("invalid json")

            return Resp()

    bad = BadJSON()
    # requests is imported inside _call_api_model, patch the global requests.post
    monkeypatch.setattr("requests.post", lambda *a, **kw: bad())

    out = mr.run_model("api_bad_json", inputs=[{"text": "x"}], batch_size=1)
    # JSON parsing error in _call_api_model is not caught there and bubbles up
    # to run_model, which logs and inserts an empty dict for that batch entry.
    assert out == [{}]


def test_run_model_async_uses_executor(monkeypatch: pytest.MonkeyPatch):
    mr = ModelRunner()

    # Register a simple local model callable that echoes inputs
    class Echo:
        def __call__(self, batch, **kwargs):
            return [{"label": "entailment", "score": 0.9} for _ in batch]

    mr._models["m"] = Echo()
    mr._model_configs["m"] = {"type": "local"}

    # Ensure async wrapper returns same as sync path
    inputs = [{"text": "a"}, {"text": "b"}]
    # Run the async function without pytest-asyncio by using asyncio.run
    res = asyncio.run(mr.run_model_async("m", inputs, batch_size=2))
    assert len(res) == 2 and all(r.get("label") for r in res)


def test_batching_edges_and_input_validation():
    mr = ModelRunner()

    # Register local model that handles uneven batches
    class Counter:
        def __call__(self, batch, **kwargs):
            return [{"label": f"n{idx}", "score": 1.0} for idx, _ in enumerate(batch)]

    mr._models["lm"] = Counter()
    mr._model_configs["lm"] = {"type": "local"}

    # Uneven batch size
    inputs = [{"x": i} for i in range(5)]
    out = mr.run_model("lm", inputs, batch_size=2)
    assert len(out) == 5

    # Empty inputs -> should return empty list
    assert mr.run_model("lm", [], batch_size=3) == []

    # Invalid inputs -> raise
    with pytest.raises(ValueError):
        mr.run_model("lm", inputs="not-a-list")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        mr.run_model("lm", inputs=["not-dicts"])  # type: ignore[list-item]
