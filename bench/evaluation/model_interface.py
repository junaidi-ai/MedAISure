"""Unified model interface and implementations for MedAISure.

This module defines a standardized `ModelInterface` that supports various model
backends (local Python, HuggingFace, HTTP API), plus a simple `ModelRegistry`.

The goal is to provide a consistent `predict(List[Dict]) -> List[Dict]` method
and standardized metadata across model types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
import os
import pickle
from pathlib import Path

try:  # optional dependency via scikit-learn
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional
    joblib = None  # type: ignore[assignment]

try:  # optional heavy dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore[assignment]


class ModelInterface(ABC):
    """Abstract base class for benchmark models.

    Implementations must provide a `predict` method that accepts a list of input
    dictionaries and returns a list of output dictionaries with a consistent
    schema for the given task type. Implementations should be deterministic
    given the same inputs and configuration.

    Error handling guidelines:
    - Do not raise for per-item prediction errors; log internally and return an
      empty dict for that item to preserve output ordering.
    - Raise early for configuration/initialization errors.
    """

    @abstractmethod
    def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run prediction on the given inputs.

        Args:
            inputs: List of input dicts.
        Returns:
            List of output dicts (same length as inputs when possible).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for the model."""
        raise NotImplementedError

    @property
    def metadata(self) -> Dict[str, Any]:
        """Optional model metadata (version, architecture, params, etc.)."""
        return {}


class LocalModel(ModelInterface):
    """Local Python model wrapper.

    Two construction paths are supported:
    - Provide a callable `predict_fn` that implements the batch interface.
    - Provide a `model_path` and an optional `loader` callable to load a model
      object. If the loaded object is callable, it will be used directly; if it
      has a `.predict(inputs)` method, that will be used.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        predict_fn: Optional[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
        ] = None,
        loader: Optional[Callable[[str], Any]] = None,
        model_id: Optional[str] = None,
    ) -> None:
        if predict_fn is None and model_path is None:
            raise ValueError("Either predict_fn or model_path must be provided")

        self._predict_fn = predict_fn
        self._model_obj: Optional[Any] = None
        self._model_file: Optional[Path] = None
        self._model_id = model_id or (
            os.path.basename(model_path) if model_path else "local-model"
        )

        # Initialize metadata container early
        self._meta: Dict[str, Any] = {}

        if model_path is not None:
            self._model_file = Path(model_path)

            # Use provided loader if any, else try auto loaders by extension
            if loader is not None:
                try:
                    self._model_obj = loader(model_path)
                except Exception as e:
                    raise ValueError(
                        f"Custom loader failed for {model_path}: {e}"
                    ) from e
            else:
                # For auto-loading, require the file to exist
                if not self._model_file.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                self._model_obj = self._auto_load_model(self._model_file)

            # Populate metadata after loading (best-effort; may be empty if file doesn't exist)
            self._populate_metadata()

    def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Basic input validation/coercion
        if not isinstance(inputs, list):  # pragma: no cover - defensive
            inputs = [inputs]  # type: ignore[list-item]
        if not all(isinstance(x, dict) for x in inputs):  # pragma: no cover - defensive
            inputs = [x if isinstance(x, dict) else {"input": x} for x in inputs]  # type: ignore[union-attr]

        try:
            if self._predict_fn is not None:
                out = self._predict_fn(inputs)
                return out if isinstance(out, list) else [out]

            if self._model_obj is None:
                return [{} for _ in inputs]

            # Prefer callable model
            if callable(self._model_obj):
                out = self._model_obj(inputs)
                return out if isinstance(out, list) else [out]

            # Fallback to .predict
            if hasattr(self._model_obj, "predict"):
                try:
                    out = self._model_obj.predict(inputs)  # type: ignore[attr-defined]
                except TypeError:
                    # Some sklearn-like models expect features, not dicts; best-effort pass-through
                    out = self._model_obj.predict(inputs)  # type: ignore[attr-defined]
                return out if isinstance(out, list) else [out]

            return [{} for _ in inputs]
        except Exception:
            # Best-effort error containment per item
            return [{} for _ in inputs]

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def metadata(self) -> Dict[str, Any]:
        # Merge base metadata with extracted
        base = {"model_id": self._model_id, "framework": "local"}
        return {**base, **(self._meta or {})}

    # ---- Internal helpers ----
    def _auto_load_model(self, path: Path) -> Any:
        """Automatically load a local model object from common file formats.

        Supported:
        - .pkl/.pickle -> pickle.load
        - .joblib -> joblib.load (if available)
        - .pt/.pth -> torch.load (if available)
        Otherwise falls back to pickle.load.
        """
        suffix = path.suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            with path.open("rb") as f:
                return pickle.load(f)
        if suffix == ".joblib":
            if joblib is None:
                raise ImportError("joblib is required to load .joblib models")
            return joblib.load(str(path))
        if suffix in {".pt", ".pth"}:
            if torch is None:
                raise ImportError("torch is required to load .pt/.pth models")
            return torch.load(str(path), map_location="cpu")
        # Default fallback
        with path.open("rb") as f:
            return pickle.load(f)

    def _populate_metadata(self) -> None:
        try:
            if not self._model_file:
                return
            stat = self._model_file.stat()
            obj = self._model_obj
            cls = type(obj).__name__ if obj is not None else None
            # Try to infer framework from object/module
            framework = None
            mod_name = getattr(type(obj), "__module__", "") if obj is not None else ""
            if "sklearn" in mod_name:
                framework = "scikit-learn"
            elif torch is not None and obj is not None and hasattr(obj, "state_dict"):
                framework = "torch"
            elif joblib is not None and self._model_file.suffix.lower() == ".joblib":
                framework = "joblib"
            elif self._model_file.suffix.lower() in {".pkl", ".pickle"}:
                framework = "pickle"

            self._meta = {
                "file_path": str(self._model_file),
                "file_size": stat.st_size,
                "file_mtime": stat.st_mtime,
                "ext": self._model_file.suffix.lower(),
                "object_class": cls,
                "object_module": mod_name,
                "framework": framework or "local",
            }
        except Exception:  # pragma: no cover - metadata best-effort only
            self._meta = self._meta or {}


class HuggingFaceModel(ModelInterface):
    """HuggingFace model using transformers Auto* and pipelines.

    This wrapper keeps dependencies optional and performs lazy imports.
    Supports common tasks: text-classification, summarization, text-generation.
    """

    def __init__(
        self,
        model_name: str,
        *,
        hf_task: str = "text-classification",
        model_id: Optional[str] = None,
        device: int = -1,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model_name = model_name
        self._hf_task = (
            "summarization" if hf_task == "text2text-generation" else hf_task
        )
        self._model_id = model_id or model_name
        self._device = device
        self._model_kwargs = model_kwargs or {}
        self._tokenizer_kwargs = tokenizer_kwargs or {}
        self._pipeline_kwargs = pipeline_kwargs or {}

        # Lazy init; created on first predict
        self._pipe = None
        self._tokenizer = None
        self._auto_model_cls = None

    def _ensure_pipeline(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import AutoTokenizer, pipeline  # type: ignore

            # Resolve AutoModel by task
            AutoModel = None
            if self._hf_task == "text-classification":
                try:
                    from transformers import (
                        AutoModelForSequenceClassification as AutoModel,
                    )  # type: ignore
                except Exception:
                    AutoModel = None
            elif self._hf_task == "summarization":
                try:
                    from transformers import AutoModelForSeq2SeqLM as AutoModel  # type: ignore
                except Exception:
                    AutoModel = None
            elif self._hf_task == "text-generation":
                try:
                    from transformers import AutoModelForCausalLM as AutoModel  # type: ignore
                except Exception:
                    AutoModel = None

            if AutoModel is not None:
                model = AutoModel.from_pretrained(
                    self._model_name, **self._model_kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name, **self._tokenizer_kwargs
                )
                self._pipe = pipeline(
                    self._hf_task,
                    model=model,
                    tokenizer=tokenizer,
                    device=self._device,
                    **self._pipeline_kwargs,
                )
                self._tokenizer = tokenizer
            else:
                self._pipe = pipeline(
                    self._hf_task,
                    model=self._model_name,
                    device=self._device,
                    **self._pipeline_kwargs,
                )
        except ImportError as e:
            raise ImportError(
                "transformers is required for HuggingFaceModel. Install with: pip install transformers torch"
            ) from e

    def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._ensure_pipeline()
        assert self._pipe is not None

        # Select input text field based on task
        def _extract_text(d: Dict[str, Any]) -> str:
            if self._hf_task == "summarization":
                for key in ("document", "text", "note"):
                    v = d.get(key)
                    if isinstance(v, str) and v.strip():
                        return v
                return str(d)
            return d.get("text", "")

        texts = [_extract_text(item) for item in inputs]
        raw = self._pipe(texts)
        if not isinstance(raw, list):
            raw = [raw]

        results: List[Dict[str, Any]] = []
        for item, result in zip(inputs, raw):
            if isinstance(result, list) and result:
                result = result[0]

            if self._hf_task == "text-classification":
                results.append(
                    {
                        "input": item,
                        "label": str(result.get("label", "")),
                        "score": float(result.get("score", 0.0)),
                    }
                )
            elif self._hf_task == "summarization":
                summary = (
                    result.get("summary_text", "")
                    if isinstance(result, dict)
                    else str(result)
                )
                results.append(
                    {"input": item, "summary": summary, "prediction": summary}
                )
            elif self._hf_task == "text-generation":
                gen = ""
                if isinstance(result, dict):
                    gen = result.get("generated_text", "")
                elif (
                    isinstance(result, list) and result and isinstance(result[0], dict)
                ):
                    gen = result[0].get("generated_text", "")
                else:
                    gen = str(result)
                results.append({"input": item, "text": gen, "prediction": gen})
            else:
                results.append(
                    {"input": item, "text": str(result), "prediction": str(result)}
                )
        return results

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def metadata(self) -> Dict[str, Any]:
        params = None
        try:
            if hasattr(self._pipe, "model"):
                params = getattr(self._pipe.model, "num_parameters", lambda: None)()
        except Exception:
            params = None
        return {
            "model_name": self._model_id,
            "framework": "huggingface",
            "task": self._hf_task,
            "parameters": params,
        }


class APIModel(ModelInterface):
    """Simple HTTP API model.

    Expects the remote API to accept either a single input dict or a list of
    input dicts and return a compatible list of outputs.
    """

    def __init__(
        self,
        api_url: str,
        *,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self._model_id = model_id or f"api-{api_url.rstrip('/').split('/')[-1]}"
        self._headers = headers or {}
        self._timeout = float(timeout)

    def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            import requests  # local import to keep optional

            headers = {"Content-Type": "application/json", **self._headers}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            resp = requests.post(
                self.api_url, json=inputs, headers=headers, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                data = [data]
            return data
        except Exception:
            return [{} for _ in inputs]

    @property
    def model_id(self) -> str:
        return self._model_id


class ModelRegistry:
    """In-memory registry for `ModelInterface` implementations."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelInterface] = {}

    def register_model(self, model: ModelInterface) -> None:
        self._models[model.model_id] = model

    def get_model(self, model_id: str) -> Optional[ModelInterface]:
        return self._models.get(model_id)

    def list_models(self) -> List[str]:
        return list(self._models.keys())
