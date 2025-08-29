"""Additional tests for EvaluationHarness: cache behavior, callbacks, cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from bench.evaluation.harness import EvaluationHarness
from bench.models.evaluation_result import EvaluationResult
from bench.models.medical_task import MedicalTask, TaskType


@pytest.fixture
def harness(tmp_path: Path) -> EvaluationHarness:
    (tmp_path / "tasks").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "cache").mkdir()
    h = EvaluationHarness(
        tasks_dir=str(tmp_path / "tasks"),
        results_dir=str(tmp_path / "results"),
        cache_dir=str(tmp_path / "cache"),
    )

    # Stub model_runner minimal interface used by evaluate/close
    class MR:
        def __init__(self):
            self._models = {}
            self._model_configs = {}
            self.unloaded: List[str] = []

        def load_model(
            self, model_name: str, model_type: str = "local", **kwargs: Any
        ) -> None:
            self._models[model_name] = MagicMock()
            self._model_configs[model_name] = {"type": model_type}

        def unload_model(self, model_name: str) -> None:
            self.unloaded.append(model_name)
            self._models.pop(model_name, None)
            self._model_configs.pop(model_name, None)

    h.model_runner = MR()  # type: ignore[assignment]
    return h


def _mk_task(task_id: str) -> MedicalTask:
    return MedicalTask(
        task_id=task_id,
        name=task_id,
        task_type=TaskType.QA,
        description="",
        input_schema={},
        output_schema={},
        metrics=["accuracy"],
        dataset=[{"question": "q?", "answer": "a"}],
    )


def _mk_result(model_id: str, task_id: str) -> EvaluationResult:
    return EvaluationResult(
        model_id=model_id,
        task_id=task_id,
        inputs=[{"question": "q?"}],
        model_outputs=[{"answer": "a"}],
        metrics_results={"accuracy": 1.0},
        metadata={"ok": True},
    )


def test_cache_miss_then_hit(
    harness: EvaluationHarness, monkeypatch: pytest.MonkeyPatch
):
    # Prepare three tasks
    task_ids = ["t1", "t2"]

    # Monkeypatch task_loader to return our tasks
    harness.task_loader.load_task = MagicMock(
        side_effect=[_mk_task("t1"), _mk_task("t2")]
    )  # type: ignore[attr-defined]

    # First run: cache miss -> _load_from_cache returns None, _evaluate_task called twice
    monkeypatch.setattr(harness, "_load_from_cache", lambda key: None)

    eval_calls: List[str] = []

    def fake_eval(model: Any, model_id: str, task: MedicalTask, batch_size: int = 8):
        eval_calls.append(task.task_id)
        return _mk_result(model_id, task.task_id)

    monkeypatch.setattr(harness, "_evaluate_task", fake_eval)

    # Spy on _save_to_cache to capture keys written
    saved_keys: List[str] = []

    def spy_save(cache_key: str | None, data: Any) -> None:
        if cache_key:
            saved_keys.append(cache_key)
        # write through to actual to validate file structure
        EvaluationHarness._save_to_cache(harness, cache_key, data)

    monkeypatch.setattr(harness, "_save_to_cache", spy_save)

    harness.evaluate(
        model_id="m",
        task_ids=task_ids,
        model_type="local",
        batch_size=4,
        use_cache=True,
        save_results=True,
    )

    assert eval_calls == ["t1", "t2"]
    assert len(saved_keys) == 2
    for key in saved_keys:
        assert Path(key).exists()
        data = json.loads(Path(key).read_text())
        assert data["task_id"] in task_ids

    # Second run: simulate cache hit for both tasks; ensure _evaluate_task NOT called
    cached_results = {tid: _mk_result("m", tid) for tid in task_ids}

    def fake_load_from_cache(cache_key: str | None):
        # pick by filename suffix
        if not cache_key:
            return None
        for tid in task_ids:
            if cache_key.endswith(f"_{tid}.json"):
                return cached_results[tid]
        return None

    harness.task_loader.load_task = MagicMock(
        side_effect=[_mk_task("t1"), _mk_task("t2")]
    )  # reload tasks
    monkeypatch.setattr(harness, "_load_from_cache", fake_load_from_cache)

    def fail_eval(*args, **kwargs):  # should not be called
        raise AssertionError("_evaluate_task should not be called on cache hit")

    monkeypatch.setattr(harness, "_evaluate_task", fail_eval)

    report2 = harness.evaluate(
        model_id="m",
        task_ids=task_ids,
        model_type="local",
        use_cache=True,
        save_results=False,
    )

    assert report2 is not None
    # verify that results reflect both tasks
    assert set(report2.task_scores.keys()) == set(task_ids)


def test_event_callbacks_and_error_isolation(
    harness: EvaluationHarness, monkeypatch: pytest.MonkeyPatch
):
    events: Dict[str, List[Any]] = {"start": [], "end": [], "progress": [], "error": []}

    def on_start(tid: str):
        events["start"].append(tid)

    def on_end(tid: str, res: Any):
        events["end"].append((tid, isinstance(res, EvaluationResult)))

    def on_progress(done: int, total: int, tid: str | None):
        events["progress"].append((done, total, tid))

    def on_error(tid: str, exc: Exception):
        events["error"].append((tid, type(exc).__name__))

    harness._on_task_start = on_start
    harness._on_task_end = on_end
    harness._on_progress = on_progress
    harness._on_error = on_error

    tids = ["ok1", "boom", "ok2"]
    harness.task_loader.load_task = MagicMock(side_effect=[_mk_task(t) for t in tids])  # type: ignore[attr-defined]

    def eval_maybe_raise(
        model: Any, model_id: str, task: MedicalTask, batch_size: int = 8
    ):
        if task.task_id == "boom":
            raise RuntimeError("failure in eval")
        return _mk_result(model_id, task.task_id)

    monkeypatch.setattr(harness, "_evaluate_task", eval_maybe_raise)

    report = harness.evaluate(
        model_id="m",
        task_ids=tids,
        model_type="local",
        use_cache=False,
        save_results=False,
    )

    # Starts for all
    assert events["start"] == tids
    # One error recorded
    assert events["error"] and events["error"][0][0] == "boom"
    # Ends recorded for non-error tasks
    ended_tasks = [tid for (tid, ok) in events["end"] if ok]
    assert set(ended_tasks) == {"ok1", "ok2"}
    # Progress should have been reported multiple times
    assert events["progress"]
    assert report is not None


def test_cleanup_unloads_model_on_close(harness: EvaluationHarness):
    # Model will be loaded during evaluate
    harness.task_loader.load_task = MagicMock(side_effect=[_mk_task("t1")])  # type: ignore[attr-defined]

    monkeypatch = pytest.MonkeyPatch()

    def fake_eval(model: Any, model_id: str, task: MedicalTask, batch_size: int = 8):
        return _mk_result(model_id, task.task_id)

    monkeypatch.setattr(harness, "_evaluate_task", fake_eval)

    harness.evaluate(
        model_id="m_close",
        task_ids=["t1"],
        model_type="local",
        use_cache=False,
        save_results=False,
    )

    # evaluate() always calls close() in finally, so the active model should already be cleared
    assert harness._active_model_id is None
    # Ensure unload_model was called for the model
    assert (
        hasattr(harness.model_runner, "unloaded")
        and "m_close" in harness.model_runner.unloaded
    )  # type: ignore[attr-defined]
