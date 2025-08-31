"""Unified model interface and implementations for MedAISure.

This module defines a standardized `ModelInterface` that supports various model
backends (local Python, HuggingFace, HTTP API), plus a simple `ModelRegistry`.

The goal is to provide a consistent `predict(List[Dict]) -> List[Dict]` method
and standardized metadata across model types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import pickle
from pathlib import Path
import time
import threading
import asyncio

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
        # Auth configuration
        auth_mode: str = "bearer",  # bearer|header|query|basic|none
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
        query_key: str = "api_key",
        basic_auth: Optional[Tuple[str, str]] = None,  # (username, password)
        # Retry configuration
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        retry_statuses: Tuple[int, ...] = (429, 500, 502, 503, 504),
        # Simple rate limiting (requests per second)
        rate_limit_rps: Optional[float] = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self._model_id = model_id or f"api-{api_url.rstrip('/').split('/')[-1]}"
        self._headers = headers or {}
        self._timeout = float(timeout)
        self._auth_mode = auth_mode
        self._auth_header = auth_header
        self._auth_prefix = auth_prefix
        self._query_key = query_key
        self._basic_auth = basic_auth
        self._max_retries = int(max_retries)
        self._backoff_factor = float(backoff_factor)
        self._retry_statuses = set(retry_statuses)
        self._rate_limit_rps = rate_limit_rps

        # Pre-create a requests session with retry-enabled adapters
        self._session = None  # lazy init

        # Rate limiter state (simple time-based limiter per POST)
        self._rl_lock = threading.Lock()
        self._last_request_ts: float = 0.0

    def predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            import requests  # local import to keep optional

            headers = {"Content-Type": "application/json", **self._headers}
            params: Dict[str, Any] = {}
            auth = None

            # Apply authentication
            headers, params, auth = self._apply_auth(headers, params)

            # Rate limit (one POST per predict call)
            self._apply_rate_limit()

            attempts = max(1, self._max_retries)
            backoff = self._backoff_factor

            for i in range(attempts):
                try:
                    resp = requests.post(
                        self.api_url,
                        json=inputs,
                        headers=headers,
                        params=params,
                        timeout=self._timeout,
                        auth=auth,
                    )
                    # Retry on specific HTTP statuses
                    if (
                        getattr(resp, "status_code", None) in self._retry_statuses
                        and i < attempts - 1
                    ):
                        time.sleep(backoff * (2**i))
                        continue
                    data = self._parse_response(resp)
                    return self._normalize_outputs(inputs, data)
                except Exception:
                    if i < attempts - 1:
                        time.sleep(backoff * (2**i))
                        continue
                    break
        except Exception:
            pass
        # Best-effort error containment per item
        return [{} for _ in inputs]

    @property
    def model_id(self) -> str:
        return self._model_id

    # ---- Async support ----
    async def async_predict(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            import httpx  # type: ignore

            headers = {"Content-Type": "application/json", **self._headers}
            params: Dict[str, Any] = {}
            auth = None

            headers, params, auth = self._apply_auth(headers, params)

            # httpx auth can be tuple for basic; otherwise we pass headers/params
            attempts = max(1, self._max_retries)
            backoff = self._backoff_factor

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                for i in range(attempts):
                    # Simple async rate limit between attempts as well
                    await self._async_rate_limit()
                    try:
                        resp = await client.post(
                            self.api_url,
                            json=inputs,
                            headers=headers,
                            params=params,
                            auth=auth,
                        )
                        data = self._parse_httpx_response(resp)
                        return self._normalize_outputs(inputs, data)
                    except Exception as e:
                        # Retry on configured statuses or connectivity
                        status = getattr(e, "response", None)
                        code = getattr(status, "status_code", None)
                        if code in self._retry_statuses and i < attempts - 1:
                            await asyncio.sleep(backoff * (2**i))
                            continue
                        # Otherwise, fall through to failure
                        break
        except Exception:
            pass
        return [{} for _ in inputs]

    # ---- Internal helpers ----
    def _ensure_session(self):
        if self._session is not None:
            return self._session
        import requests  # local import to keep optional
        from requests.adapters import HTTPAdapter  # type: ignore

        try:
            from urllib3.util.retry import Retry  # type: ignore
        except Exception:  # pragma: no cover - fallback if urllib3 Retry API changes
            Retry = None  # type: ignore

        session = requests.Session()
        if Retry is not None:
            retry = Retry(
                total=self._max_retries,
                connect=self._max_retries,
                read=self._max_retries,
                status=self._max_retries,
                backoff_factor=self._backoff_factor,
                status_forcelist=tuple(self._retry_statuses),
                allowed_methods=(
                    frozenset(["GET", "POST", "PUT", "DELETE", "PATCH"])  # type: ignore[arg-type]
                ),
                raise_on_status=False,
                respect_retry_after_header=True,
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        self._session = session
        return session

    def _apply_auth(
        self, headers: Dict[str, str], params: Dict[str, Any]
    ) -> Tuple[Dict[str, str], Dict[str, Any], Optional[Any]]:
        auth_obj = None
        mode = (self._auth_mode or "bearer").lower()
        if mode == "bearer" and self.api_key:
            headers[self._auth_header] = f"{self._auth_prefix}{self.api_key}"
        elif mode == "header" and self.api_key:
            headers[self._auth_header] = f"{self.api_key}"
        elif mode == "query" and self.api_key:
            params[self._query_key] = self.api_key
        elif mode == "basic" and self._basic_auth:
            # requests/httpx accept (username, password)
            auth_obj = self._basic_auth
        return headers, params, auth_obj

    def _apply_rate_limit(self) -> None:
        if not self._rate_limit_rps or self._rate_limit_rps <= 0:
            return
        with self._rl_lock:
            now = time.time()
            min_interval = 1.0 / self._rate_limit_rps
            wait = (self._last_request_ts + min_interval) - now
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.time()

    async def _async_rate_limit(self) -> None:
        if not self._rate_limit_rps or self._rate_limit_rps <= 0:
            return
        # Use the same timestamp but avoid blocking loop with sync sleep
        now = time.time()
        min_interval = 1.0 / self._rate_limit_rps
        # Optimistic check outside lock
        wait = (self._last_request_ts + min_interval) - now
        if wait > 0:
            await asyncio.sleep(wait)
        # Update timestamp inside lock to avoid races with sync path
        with self._rl_lock:
            self._last_request_ts = time.time()

    def _parse_response(self, resp):
        try:
            # Raise for HTTP errors; let Retry handle via adapter, but still handle here
            if hasattr(resp, "raise_for_status"):
                resp.raise_for_status()
            data = resp.json()
        except Exception:
            try:
                # Fallback to text
                data = resp.text  # type: ignore[attr-defined]
            except Exception:
                data = None
        return data

    def _parse_httpx_response(self, resp):
        try:
            resp.raise_for_status()
            return resp.json()
        except Exception:
            try:
                return resp.text
            except Exception:
                return None

    def _normalize_outputs(
        self, inputs: List[Dict[str, Any]], data: Any
    ) -> List[Dict[str, Any]]:
        # Ensure list
        if isinstance(data, list):
            norm_list = []
            for item in data:
                if isinstance(item, dict):
                    norm_list.append(item)
                else:
                    norm_list.append({"prediction": item})
            # If API returned fewer/more items, best-effort alignment by trunc/pad
            if len(norm_list) < len(inputs):
                norm_list.extend([{} for _ in range(len(inputs) - len(norm_list))])
            elif len(norm_list) > len(inputs):
                norm_list = norm_list[: len(inputs)]
            return norm_list
        # Single object/primitive
        if isinstance(data, dict):
            return [data for _ in inputs]
        if data is None:
            return [{} for _ in inputs]
        return [{"prediction": data} for _ in inputs]


class ModelRegistry:
    """In-memory registry for `ModelInterface` implementations."""

    def __init__(self) -> None:
        # Backward-compat flat map to the "default" or latest version
        self._models: Dict[str, ModelInterface] = {}
        # Versioned storage: {model_id: {version: model}}
        self._versions: Dict[str, Dict[str, ModelInterface]] = {}
        # Track last registered version per model for default resolution
        self._latest_version: Dict[str, str] = {}
        # Optional default per-model configuration (arbitrary dict)
        self._default_config: Dict[str, Dict[str, Any]] = {}

    def register_model(
        self,
        model: ModelInterface,
        *,
        version: Optional[str] = None,
        validate: bool = True,
    ) -> None:
        """Register a model (optionally versioned).

        Args:
            model: Model to register. Must implement ModelInterface.
            version: Optional version string (e.g., "v1", "1.0.0"). If omitted,
                     the model is treated as the default/current version.
            validate: If True, perform lightweight interface validation.
        """
        if validate:
            self._validate_model(model)

        mid = model.model_id
        if not mid:
            raise ValueError("Model must have a non-empty model_id")

        # Maintain flat map for backward-compatible retrieval
        self._models[mid] = model

        # Maintain version map
        if version:
            self._versions.setdefault(mid, {})[version] = model
            self._latest_version[mid] = version
        else:
            # Use a sentinel version name for default when not provided
            v = "default"
            self._versions.setdefault(mid, {})[v] = model
            self._latest_version[mid] = v

    def get_model(
        self, model_id: str, *, version: Optional[str] = None
    ) -> Optional[ModelInterface]:
        """Retrieve a model by id (and optional version)."""
        if version is None:
            # Backward-compatible behavior: return the flat entry if present
            m = self._models.get(model_id)
            if m is not None:
                return m
            # Fallback to latest version map
            latest = self._latest_version.get(model_id)
            if latest is None:
                return None
            return self._versions.get(model_id, {}).get(latest)
        # Version-specific
        return self._versions.get(model_id, {}).get(version)

    def list_models(self, *, include_versions: bool = False) -> Any:
        """List registered models.

        Args:
            include_versions: If True, returns {id: [versions...]}. Otherwise
                              returns a list of model ids.
        """
        if not include_versions:
            return list(self._models.keys())
        return {mid: sorted(list(vers.keys())) for mid, vers in self._versions.items()}

    # ---- Validation ----
    def _validate_model(self, model: ModelInterface) -> None:
        # Ensure required interface parts exist
        if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
            raise TypeError("Model must define a callable predict(inputs) method")
        if not hasattr(model, "model_id"):
            raise TypeError("Model must define a model_id property")
        # metadata is optional by interface; don't enforce content

    # ---- Metadata helpers ----
    def get_metadata(
        self, model_id: str, *, version: Optional[str] = None
    ) -> Dict[str, Any]:
        m = self.get_model(model_id, version=version)
        return m.metadata if m is not None else {}

    # ---- Default configuration ----
    def set_default_config(self, model_id: str, config: Dict[str, Any]) -> None:
        self._default_config[model_id] = dict(config)

    def get_default_config(self, model_id: str) -> Dict[str, Any]:
        return dict(self._default_config.get(model_id, {}))

    # ---- Persistence (lightweight; no model object serialization) ----
    def save_registry(self, path: str) -> None:
        """Persist registry topology and configs to a JSON file.

        Note: Model objects are NOT serialized. This captures
        - model ids
        - versions present
        - latest version mapping
        - default configs
        """
        import json

        state = {
            "models": sorted(list(self._versions.keys())),
            "versions": {
                mid: sorted(list(vers.keys())) for mid, vers in self._versions.items()
            },
            "latest": dict(self._latest_version),
            "default_config": self._default_config,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)

    def load_registry(self, path: str) -> None:
        """Load registry topology and configs from a JSON file.

        Note: This does not reconstruct model objects; it only restores
        ids/versions/defaults/configs. Models must be re-registered.
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Reset structures but keep current models (ephemeral)
        self._versions = {mid: {} for mid in state.get("models", [])}
        for mid, version_list in state.get("versions", {}).items():
            self._versions.setdefault(mid, {})
            for v in version_list:
                # placeholder None would be type-incompatible; keep mapping empty until re-register
                pass
        self._latest_version = dict(state.get("latest", {}))
        self._default_config = dict(state.get("default_config", {}))
