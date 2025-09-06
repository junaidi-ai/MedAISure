"""Model runner for MedAISure benchmark."""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from .model_interface import ModelInterface

# Third-party imports

logger = logging.getLogger(__name__)

# Type variable for the model type
M = TypeVar("M")  # Model type
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


class ModelRunner(Generic[M, T, R]):
    """Handles loading and running different types of models.

    Supports three model sources: HuggingFace pipelines, local Python modules,
    and simple HTTP API-backed models. Caches loaded models and their configs.

    Type variables:
        M: Type variable for the model class
        T: Type variable for input data
        R: Type variable for result data

    Example – quick start:
        >>> from bench.evaluation.model_runner import ModelRunner
        >>> mr = ModelRunner()
        >>> # Load a HuggingFace pipeline (requires transformers, may download weights)
        >>> # _ = mr.load_model("hf-cls", model_type="huggingface", model_path="prajjwal1/bert-tiny", hf_task="text-classification")  # doctest: +SKIP
        >>> # Run on small inputs
        >>> # preds = mr.run_model("hf-cls", inputs=[{"text": "Patient is stable."}])  # doctest: +SKIP
        >>> isinstance(mr, ModelRunner)
        True
    """

    def __init__(self) -> None:
        """Initialize the model runner with empty model and tokenizer caches."""
        self._models: Dict[str, Any] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._tokenizers: Dict[str, Any] = {}

    def register_interface_model(self, model: "ModelInterface") -> None:
        """Register a model implementing the ModelInterface.

        This allows external callers to construct rich model wrappers and plug
        them into the runner for use with run_model().

        Args:
            model: An instance implementing ModelInterface.
        """
        # Store object and minimal config
        self._models[model.model_id] = model
        self._model_configs[model.model_id] = {
            "type": "interface",
        }
        logger.info(f"Registered interface model: {model.model_id}")

    def load_model(
        self,
        model_name: str,
        model_type: str = "local",
        model_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Any:
        """Load a model for inference.

        Args:
            model_name: Unique name for the model.
            model_type: Type of model (huggingface, local, api).
            model_path: Path to the model or module.
            **kwargs: Additional arguments for model loading.

        Returns:
            The loaded model object.

        Raises:
            ValueError: If the model type is unsupported or required
                arguments are missing.

        Examples:
            HuggingFace (pipeline):
                >>> mr = ModelRunner()
                >>> # Tiny model to keep downloads small; requires transformers
                >>> # pipe = mr.load_model(
                ... #     "hf-sum",
                ... #     model_type="huggingface",
                ... #     model_path="sshleifer/tiny-t5",
                ... #     hf_task="summarization",
                ... # )  # doctest: +SKIP

            Local module:
                >>> # Suppose mypkg.model has a load_model(path) -> callable
                >>> mr = ModelRunner()
                >>> # model = mr.load_model(
                ... #     "local-demo",
                ... #     model_type="local",
                ... #     model_path="/tmp/model.bin",
                ... #     module_path="bench.examples.mypkg.model",
                ... # )  # doctest: +SKIP

            API model registration:
                >>> mr = ModelRunner()
                >>> cfg = mr.load_model(
                ...     "api-demo", model_type="api", endpoint="https://api.example/v1/predict", api_key="sk-xxx"
                ... )
                >>> isinstance(cfg, dict)
                True
        """
        if model_name in self._models:
            logger.warning(f"Model {model_name} is already loaded. Unload it first.")
            return self._models[model_name]

        model_path_str = str(model_path) if model_path is not None else None
        model = None

        if model_type == "huggingface":
            model = self._load_huggingface_model(model_name, model_path_str, **kwargs)
        elif model_type == "local":
            model = self._load_local_model(model_name, model_path_str, **kwargs)
        elif model_type == "api":
            model = self._load_api_model(model_name, **kwargs)
        elif model_type == "interface":
            # Expect a concrete ModelInterface instance passed as 'model'
            mi = kwargs.get("model")
            if mi is None:
                raise ValueError(
                    "For model_type='interface', pass a 'model' instance implementing ModelInterface"
                )
            # Delay import for typing only to avoid hard dependency here
            self.register_interface_model(mi)
            model = mi
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model

    def _load_huggingface_model(
        self, model_name: str, model_path: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Load a HuggingFace model.

        Args:
            model_name: Name to register the model under.
            model_path: Path to the model or model name from HuggingFace Hub.
                      If None, model_name will be used as the model identifier.
            **kwargs: Additional arguments for model loading.

        Returns:
            The loaded HuggingFace pipeline.

        Raises:
            ImportError: If transformers is not installed.
            ValueError: If model loading fails.
        """
        model_type = "huggingface"
        try:
            from transformers import (
                AutoTokenizer,
                pipeline,
            )

            # Use model_name as model identifier if model_path is not provided
            model_identifier = model_path if model_path else model_name
            model_kwargs = kwargs.get("model_kwargs", {})
            tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
            generation_kwargs = kwargs.get("generation_kwargs", {})
            num_labels = kwargs.get("num_labels")
            hf_task = kwargs.get("hf_task", kwargs.get("task", "text-classification"))
            # Normalize common aliases
            if hf_task == "text2text-generation":
                hf_task = "summarization"

            logger.info(f"Loading {model_type} model: {model_identifier}")

            # Select appropriate AutoModel depending on task
            AutoModel = None
            tokenizer = None
            loaded_model = None
            # Optional advanced settings
            trust_remote_code = bool(kwargs.get("trust_remote_code", False))
            device = kwargs.get("device", -1)
            device_map = kwargs.get("device_map")
            torch_dtype = kwargs.get("torch_dtype")  # e.g., "auto", "float16"
            low_cpu_mem_usage = kwargs.get("low_cpu_mem_usage")
            revision = kwargs.get("revision")
            if hf_task == "text-classification":
                from transformers import (
                    AutoModelForSequenceClassification as _AutoModel,
                )  # type: ignore

                AutoModel = _AutoModel
            elif hf_task in {"summarization"}:
                try:
                    from transformers import AutoModelForSeq2SeqLM as _AutoModel  # type: ignore

                    AutoModel = _AutoModel
                except Exception:
                    AutoModel = None
            elif hf_task in {"text-generation"}:
                try:
                    from transformers import AutoModelForCausalLM as _AutoModel  # type: ignore

                    AutoModel = _AutoModel
                except Exception:
                    AutoModel = None

            # Load model/tokenizer if available; otherwise rely on pipeline to resolve by id
            if AutoModel is not None:
                # Prepare from_pretrained kwargs
                fp_kwargs = {
                    **({"num_labels": num_labels} if num_labels is not None else {}),
                    **model_kwargs,
                }
                if trust_remote_code:
                    fp_kwargs["trust_remote_code"] = True
                if device_map is not None:
                    fp_kwargs["device_map"] = device_map
                if torch_dtype is not None:
                    fp_kwargs["torch_dtype"] = torch_dtype
                if low_cpu_mem_usage is not None:
                    fp_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
                if revision is not None:
                    fp_kwargs["revision"] = revision

                loaded_model = AutoModel.from_pretrained(
                    model_identifier,
                    **fp_kwargs,
                )
                try:
                    # Disable gradients and set eval mode to reduce overhead
                    import torch  # type: ignore

                    torch.set_grad_enabled(False)
                except Exception:
                    pass
                try:
                    loaded_model.eval()  # type: ignore[attr-defined]
                except Exception:
                    pass

                tokenizer = AutoTokenizer.from_pretrained(
                    model_identifier, **tokenizer_kwargs
                )
                pipe = pipeline(
                    hf_task,
                    model=loaded_model,
                    tokenizer=tokenizer,
                    device=device,
                    **kwargs.get("pipeline_kwargs", {}),
                )
            else:
                pipe = pipeline(
                    hf_task,
                    model=model_identifier,
                    device=device,
                    **kwargs.get("pipeline_kwargs", {}),
                )

            # Store the pipeline and its components
            self._models[model_name] = pipe
            self._tokenizers[model_name] = tokenizer

            # Build metadata/config snapshot
            meta: Dict[str, Any] = {
                "type": model_type,
                "path": model_identifier,
                "hf_task": hf_task,
                "generation_kwargs": generation_kwargs,
                "device": device,
                "device_map": device_map,
                "torch_dtype": str(torch_dtype) if torch_dtype is not None else None,
                "trust_remote_code": trust_remote_code,
                **kwargs,
            }
            try:
                model_ref = getattr(pipe, "model", None)
                if model_ref is not None and hasattr(model_ref, "config"):
                    cfg = model_ref.config
                    meta.update(
                        {
                            "model_type_name": getattr(cfg, "model_type", None),
                            "vocab_size": getattr(cfg, "vocab_size", None),
                            "max_position_embeddings": getattr(
                                cfg, "max_position_embeddings", None
                            ),
                            "num_hidden_layers": getattr(
                                cfg, "num_hidden_layers", None
                            ),
                            "hidden_size": getattr(cfg, "hidden_size", None),
                            "id2label": getattr(cfg, "id2label", None),
                            "label2id": getattr(cfg, "label2id", None),
                        }
                    )
                # Number of parameters
                if loaded_model is None:
                    loaded_model = getattr(pipe, "model", None)
                if loaded_model is not None:
                    try:
                        total_params = sum(p.numel() for p in loaded_model.parameters())  # type: ignore[attr-defined]
                        meta["num_parameters"] = int(total_params)
                        meta["dtype"] = str(next(loaded_model.parameters()).dtype)  # type: ignore[attr-defined]
                        # Device best-effort
                        try:
                            meta["model_device"] = str(
                                next(loaded_model.parameters()).device
                            )  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                # Best-effort metadata extraction
                pass

            self._model_configs[model_name] = meta

            logger.info(
                f"Successfully loaded {model_type} model: {model_name} "
                f"from {model_identifier}"
            )
            return pipe

        except ImportError as e:
            raise ImportError(
                "transformers library is required for HuggingFace models. "
                "Install with: pip install transformers torch"
            ) from e
        except Exception as e:
            logger.error(
                f"Error loading model {model_identifier} with error: {str(e)}",
                exc_info=True,
            )
            raise ValueError(
                f"Failed to load HuggingFace model '{model_identifier}': {str(e)}"
            ) from e

    def unload_model(self, model_name: str) -> None:
        """Unload a model and clean up resources.

        Args:
            model_name: Name of the model to unload.

        Example:
            >>> mr = ModelRunner()
            >>> _ = mr.load_model("api-x", model_type="api", endpoint="https://api.example", api_key="k")
            >>> mr.unload_model("api-x")
            >>> # No exception means success
        """
        if model_name in self._models:
            # Best-effort cleanup of torch resources
            try:
                model = self._models.get(model_name)
                model_ref = getattr(model, "model", None)
                if model_ref is not None and hasattr(model_ref, "cpu"):
                    model_ref.cpu()
                # Empty CUDA cache if available
                try:
                    import torch  # type: ignore

                    if torch.cuda.is_available():  # pragma: no cover - env-dependent
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass
            del self._models[model_name]
        if model_name in self._model_configs:
            del self._model_configs[model_name]
        if model_name in self._tokenizers:
            del self._tokenizers[model_name]
        logger.info(f"Unloaded model: {model_name}")

    def _load_local_model(
        self, model_name: str, model_path: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Load a local model from a Python module.

        Args:
            model_name: Name to register the model under.
            model_path: Path to the model or directory.
            **kwargs: Additional arguments for model loading.
                - module_path: Required. Python import path to the module.
                - load_func: Optional. Name of the load function (default: 'load_model')

        Returns:
            The loaded model object.

        Raises:
            ImportError: If the module cannot be imported.
            ValueError: If model_path or module_path is not provided or invalid.

        Example:
            >>> mr = ModelRunner()
            >>> # model = mr._load_local_model(
            ... #     "demo", model_path="/tmp/m.bin", module_path="bench.examples.mypkg.model"
            ... # )  # doctest: +SKIP
        """
        if not model_path:
            raise ValueError("model_path is required for local models")

        module_path = kwargs.pop("module_path", None)
        if not module_path:
            raise ValueError("module_path is required for local models")

        load_func_name = kwargs.pop("load_func", "load_model")

        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Get the load function
            load_func = getattr(module, load_func_name, None)
            if load_func is None or not callable(load_func):
                raise ValueError(
                    f"Module {module_path} has no callable '{load_func_name}' function"
                )

            # Load the model
            model = load_func(model_path, **kwargs)

            # Store the model and its config
            self._models[model_name] = model
            self._model_configs[model_name] = {
                "type": "local",
                "path": model_path,
                "module": module_path,
                "load_func": load_func_name,
                **kwargs,
            }
            logger.info(f"Loaded local model {model_name} from {module_path}")

            return model

        except ImportError as e:
            raise ImportError(
                f"Failed to import model module {module_path}: {e}"
            ) from e

    def _load_api_model(self, model_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Register an API-based model.

        Args:
            model_name: Name to register the model under.
            **kwargs: Must include 'api_key' and 'endpoint'. Optional:
                - timeout (float): request timeout in seconds (default: 30.0)
                - max_retries (int): number of retries on request failure (default: 0)
                - backoff_factor (float): base backoff seconds
                between retries (default: 0.5)
                - headers (dict): additional headers

        Returns:
            The model configuration dictionary.

        Raises:
            ValueError: If required arguments are missing.

        Example:
            >>> mr = ModelRunner()
            >>> cfg = mr._load_api_model("api-a", endpoint="https://api.example", api_key="k")
            >>> set(cfg.keys()) >= {"type", "endpoint", "api_key"}
            True
        """
        if "api_key" not in kwargs:
            raise ValueError("api_key is required for API models")
        if "endpoint" not in kwargs:
            raise ValueError("endpoint is required for API models")

        # Store the API configuration
        model_config = {
            "type": "api",
            "endpoint": kwargs["endpoint"],
            "api_key": kwargs["api_key"],
            "headers": kwargs.get("headers", {}),
            "timeout": float(kwargs.get("timeout", 30.0)),
            "max_retries": int(kwargs.get("max_retries", 0)),
            "backoff_factor": float(kwargs.get("backoff_factor", 0.5)),
        }

        # Store the model config
        self._model_configs[model_name] = model_config

        # Create a simple callable that will make the API request
        def api_call(inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self._call_api_model(model_config, inputs)

        self._models[model_name] = api_call
        logger.info(f"Registered API model: {model_name} ({kwargs['endpoint']})")

        return model_config

    def _create_api_callable(
        self, model_name: str, **kwargs: Any
    ) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Create a callable that handles API requests.

        Args:
            model_name: Name of the model.
            **kwargs: Configuration for the API call.

        Returns:
            A function that takes a list of inputs and returns model predictions.
        """

        import requests  # type: ignore[import-untyped]

        endpoint: str = kwargs["endpoint"]
        headers: Dict[str, str] = kwargs.get("headers", {})
        request_format: Callable[[List[Dict[str, Any]]], Any] = kwargs["request_format"]
        response_parser: Callable[[Any], List[Dict[str, Any]]] = kwargs[
            "response_parser"
        ]

        def api_call(inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Make an API call with the given inputs."""
            try:
                # Format the request
                payload = request_format(inputs)

                # Make the request
                response = requests.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()

                # Parse the response
                response_data = response.json()
                return response_parser(response_data)

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed for model {model_name}: {e}")
                # Return empty list on error to maintain consistent return type
                return []

        return api_call

    def run_model(
        self,
        model_id: str,
        inputs: List[Dict[str, Any]],
        batch_size: int = 8,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch of inputs.

        Args:
            model_id: ID of the loaded model.
            inputs: List of input dictionaries.
            batch_size: Number of inputs to process in a batch.
            **kwargs: Additional arguments for model inference.

        Returns:
            List of model outputs with 'label' and 'score' keys.

        Raises:
            ValueError: If model_id is not found or inputs are invalid.

        Examples:
            API model:
                >>> mr = ModelRunner()
                >>> _ = mr.load_model("api-x", model_type="api", endpoint="https://api.example", api_key="k")
                >>> out = mr.run_model("api-x", inputs=[{"text": "abc"}], batch_size=1)
                >>> isinstance(out, list)
                True

            HuggingFace (summarization):
                >>> mr = ModelRunner()
                >>> # _ = mr.load_model("hf-sum", model_type="huggingface", model_path="sshleifer/tiny-t5", hf_task="summarization")  # doctest: +SKIP
                >>> # out = mr.run_model("hf-sum", inputs=[{"document": "Short note."}], batch_size=1)  # doctest: +SKIP
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not loaded. Call load_model() first.")

        if not isinstance(inputs, list) or not all(
            isinstance(item, dict) for item in inputs
        ):
            raise ValueError("Inputs must be a list of dictionaries.")

        # Get model type to handle different model types appropriately
        model_type = self._model_configs[model_id].get("type", "unknown")
        model = self._models[model_id]
        results: List[Dict[str, Any]] = []

        # Get label map from config if available
        label_map = self._model_configs[model_id].get("label_map", {})

        # Process inputs in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]

            try:
                batch_start = time.time()
                if model_type == "huggingface":
                    # For HuggingFace pipeline
                    hf_task = self._model_configs.get(model_id, {}).get(
                        "hf_task", "text-classification"
                    )
                    gen_kwargs = self._model_configs.get(model_id, {}).get(
                        "generation_kwargs", {}
                    )
                    # Select input text field based on task type
                    if hf_task == "summarization":
                        # Prefer common clinical note fields and communication messages
                        def _extract_text(d: Dict[str, Any]) -> str:
                            for key in ("document", "text", "note", "patient_message"):
                                v = d.get(key)
                                if isinstance(v, str) and v.strip():
                                    return v
                            # Combine fields if present (e.g., patient_message + context)
                            pm = str(d.get("patient_message", "")).strip()
                            ctx = str(d.get("context", "")).strip()
                            if pm and ctx:
                                return f"{pm}\n\nContext: {ctx}"
                            if pm:
                                return pm
                            # Fallback to stringified input
                            return str(d)

                        texts = [_extract_text(item) for item in batch]
                    else:
                        # For text-generation or other tasks, prefer 'text' but fall back to common fields
                        def _extract_text_generic(d: Dict[str, Any]) -> str:
                            txt = str(d.get("text", "")).strip()
                            if txt:
                                return txt
                            for key in ("patient_message", "document", "note"):
                                v = d.get(key)
                                if isinstance(v, str) and v.strip():
                                    return v
                            pm = str(d.get("patient_message", "")).strip()
                            ctx = str(d.get("context", "")).strip()
                            if pm and ctx:
                                return f"{pm}\n\nContext: {ctx}"
                            if pm:
                                return pm
                            return str(d)

                        texts = [_extract_text_generic(item) for item in batch]

                    # Pass generation kwargs for generative tasks
                    call_kwargs: Dict[str, Any] = {}
                    if hf_task in {"summarization", "text-generation"} and isinstance(
                        gen_kwargs, dict
                    ):
                        call_kwargs = gen_kwargs
                    batch_results = model(texts, **call_kwargs)
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]

                    for item, result in zip(batch, batch_results):
                        # HF may return list of candidates per input
                        if isinstance(result, list) and result:
                            result = result[0]

                        if hf_task == "text-classification":
                            raw_label = str(result.get("label", ""))
                            score = float(result.get("score", 0.0))
                            predicted_label = label_map.get(raw_label, raw_label)
                            if (
                                predicted_label.startswith("LABEL_")
                                and "_" in predicted_label
                            ):
                                try:
                                    idx = int(predicted_label.split("_")[1])
                                    if (
                                        not label_map
                                        and hasattr(model, "model")
                                        and hasattr(model.model, "config")
                                    ):
                                        if hasattr(model.model.config, "id2label"):
                                            label_map = model.model.config.id2label
                                            predicted_label = label_map.get(
                                                idx, predicted_label
                                            )
                                except (ValueError, IndexError):
                                    pass
                            results.append(
                                {
                                    "input": item,
                                    "label": predicted_label,
                                    "score": score,
                                    "raw_label": raw_label,
                                }
                            )
                        elif hf_task == "summarization":
                            summary = (
                                result.get("summary_text", "")
                                if isinstance(result, dict)
                                else str(result)
                            )
                            results.append(
                                {
                                    "input": item,
                                    "summary": summary,
                                    "prediction": summary,
                                }
                            )
                        elif hf_task == "text-generation":
                            gen = ""
                            if isinstance(result, dict):
                                gen = result.get("generated_text", "")
                            elif (
                                isinstance(result, list)
                                and result
                                and isinstance(result[0], dict)
                            ):
                                gen = result[0].get("generated_text", "")
                            else:
                                gen = str(result)
                            results.append(
                                {
                                    "input": item,
                                    "text": gen,
                                    "prediction": gen,
                                }
                            )
                        else:
                            # Fallback: store as raw text
                            results.append(
                                {
                                    "input": item,
                                    "text": str(result),
                                    "prediction": str(result),
                                }
                            )

                elif model_type == "local":
                    # For local models that implement __call__
                    batch_results = model(batch, **kwargs)
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]
                    results.extend(batch_results)

                elif model_type == "api":
                    # For API models that implement __call__
                    batch_results = model(batch)
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]
                    results.extend(batch_results)

                elif model_type == "interface":
                    # For ModelInterface implementations
                    # We only rely on the standard predict() batch method
                    batch_results = model.predict(batch)
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]
                    results.extend(batch_results)

                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

            except Exception as e:
                logger.error(
                    f"Error running model {model_id} on batch: {e}", exc_info=True
                )
                # Add empty dicts for failed predictions to maintain order and type
                results.extend([{} for _ in batch])
            finally:
                try:
                    latency = time.time() - batch_start
                    logger.debug(
                        "Model batch completed",
                        extra={
                            "model_id": model_id,
                            "model_type": model_type,
                            "batch_size": len(batch),
                            "latency_sec": round(latency, 6),
                        },
                    )
                except Exception:
                    # Best-effort logging only
                    pass

        return results

    def _call_api_model(
        self, model_config: Dict[str, Any], batch: List[Dict[str, Any]], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Helper method to call API-based models.

        Args:
            model_config: Configuration dictionary for the API model.
            batch: List of input dictionaries to process.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            List of model predictions as dictionaries with 'label' and 'score' keys.
        """
        import requests

        # Get the endpoint and headers from config
        endpoint = model_config["endpoint"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_config['api_key']}",
            **model_config.get("headers", {}),
        }

        timeout = float(model_config.get("timeout", 30.0))
        max_retries = int(model_config.get("max_retries", 0))
        backoff = float(model_config.get("backoff_factor", 0.5))

        attempt = 0
        while True:
            attempt += 1
            start = time.time()
            try:
                # Make the API request
                response = requests.post(
                    url=endpoint, headers=headers, json=batch, timeout=timeout, **kwargs
                )
                # Check for errors
                response.raise_for_status()
                latency = time.time() - start
                logger.debug(
                    "API call succeeded",
                    extra={
                        "endpoint": endpoint,
                        "batch_size": len(batch),
                        "status_code": response.status_code,
                        "latency_sec": round(latency, 6),
                        "attempt": attempt,
                    },
                )
                # Parse and return the response
                result = response.json()
                break
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "API request failed for model %s: %s. Attempt %s/%s",
                    model_config,
                    e,
                    attempt,
                    max_retries + 1,
                )
                if attempt > max_retries:
                    # Return empty list on error to maintain consistent return type
                    return []
                # Exponential backoff
                sleep_s = backoff * (2 ** (attempt - 1))
                time.sleep(sleep_s)

        # Ensure we return a list of results with the expected structure
        if not isinstance(result, list):
            result = [result]

        # Convert each item to the expected dictionary format
        formatted_results = []
        for item in result:
            if isinstance(item, dict):
                # Ensure the dict has the expected keys
                if "label" not in item or "score" not in item:
                    formatted_results.append(
                        {"label": str(item), "score": float(item.get("score", 1.0))}
                    )
                else:
                    formatted_results.append(item)
            else:
                # Convert non-dict results to the expected format
                formatted_results.append({"label": str(item), "score": 1.0})

        return formatted_results

    async def run_model_async(
        self,
        model_id: str,
        inputs: List[Dict[str, Any]],
        batch_size: int = 8,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Asynchronous wrapper around run_model using a thread executor.

        This avoids adding async HTTP dependencies while providing an async API.

        Example:
            >>> import asyncio
            >>> mr = ModelRunner()
            >>> _ = mr.load_model("api-y", model_type="api", endpoint="https://api.example", api_key="k")
            >>> async def _go():
            ...     return await mr.run_model_async("api-y", inputs=[{"text": "abc"}], batch_size=1)
            >>> # asyncio.run(_go())  # doctest: +SKIP
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.run_model(model_id, inputs, batch_size=batch_size, **kwargs),
        )
