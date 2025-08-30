from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import pytest

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional
    joblib = None  # type: ignore[assignment]

from bench.evaluation.model_interface import LocalModel


class _CallableEcho:
    def __call__(self, batch: List[Dict[str, Any]], **kwargs: Any):
        return [{"ok": True, "i": i} for i, _ in enumerate(batch)]


def test_localmodel_custom_loader_and_metadata(tmp_path: Path) -> None:
    # Write a dummy pickle file
    model_obj = _CallableEcho()
    pkl_path = tmp_path / "model.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(model_obj, f)

    m = LocalModel(model_path=str(pkl_path))

    # Prediction path (callable object)
    out = m.predict([{"x": 1}, {"x": 2}])
    assert isinstance(out, list) and len(out) == 2
    assert out[0].get("ok") is True and out[1].get("i") == 1

    # Metadata present
    meta = m.metadata
    assert meta.get("file_path") == str(pkl_path)
    assert meta.get("ext") in {".pkl", ".pickle"}
    assert meta.get("object_class") == type(model_obj).__name__
    assert meta.get("framework") in {"pickle", "local"}


@pytest.mark.skipif(joblib is None, reason="joblib not available")
def test_localmodel_joblib_loading(tmp_path: Path) -> None:
    # Use a simple callable to persist via joblib
    model_obj = _CallableEcho()
    jb_path = tmp_path / "model.joblib"
    joblib.dump(model_obj, str(jb_path))

    m = LocalModel(model_path=str(jb_path))
    out = m.predict([{"x": 1}])
    assert isinstance(out, list) and out and out[0].get("ok") is True
    assert m.metadata.get("ext") == ".joblib"


def test_localmodel_file_not_found_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pkl"
    with pytest.raises(FileNotFoundError):
        _ = LocalModel(model_path=str(missing))


def test_localmodel_custom_loader_error_propagates(tmp_path: Path) -> None:
    bad = tmp_path / "x.pkl"
    bad.write_text("not a real model")

    def loader_raises(path: str):
        raise RuntimeError("boom")

    with pytest.raises(ValueError):
        _ = LocalModel(model_path=str(bad), loader=loader_raises)


def test_localmodel_predict_fn_path() -> None:
    def predict_fn(batch: List[Dict[str, Any]]):
        return [{"y": i} for i, _ in enumerate(batch)]

    m = LocalModel(predict_fn=predict_fn, model_id="x")
    out = m.predict([{"a": 1}, {"a": 2}])
    assert len(out) == 2 and out[1]["y"] == 1
