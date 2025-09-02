from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
)


logger = logging.getLogger(__name__)


class Step(Protocol):
    """Protocol for a preprocessing step; must be callable and serializable."""

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]: ...  # noqa: E701

    @property
    def name(self) -> str: ...  # noqa: E701

    def to_dict(self) -> Dict[str, Any]: ...  # noqa: E701


S = TypeVar("S", bound="SerializableStep")


@dataclass
class SerializableStep:
    """Base class for serializable steps.

    Subclasses should override `__call__` and may add fields. The default
    `to_dict`/`from_dict` handle basic dataclass fields.
    """

    _type: str = field(init=False, default="SerializableStep")

    @property
    def name(self) -> str:
        return self._type

    def __call__(
        self, item: Dict[str, Any]
    ) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["_type"] = self._type
        return d

    @classmethod
    def from_dict(
        cls: Type[S], data: Mapping[str, Any]
    ) -> S:  # pragma: no cover - usually use registry
        obj = cls()  # type: ignore[call-arg]
        for k, v in data.items():
            if k == "_type":
                continue
            setattr(obj, k, v)
        return obj


class StepRegistry:
    """Registry to (de)serialize pipeline steps by type name."""

    _registry: Dict[str, Type[SerializableStep]] = {}

    @classmethod
    def register(cls, step_cls: Type[SerializableStep]) -> Type[SerializableStep]:
        cls._registry[step_cls.__name__] = step_cls
        return step_cls

    @classmethod
    def create(cls, data: Mapping[str, Any]) -> SerializableStep:
        t = data.get("_type") or data.get("type") or data.get("name")
        if not t or t not in cls._registry:
            raise ValueError(f"Unknown step type for deserialization: {t}")
        step_cls = cls._registry[t]
        return step_cls.from_dict(data)  # type: ignore[return-value]


# --- Built-in Steps ---


@StepRegistry.register
@dataclass
class MedicalTextNormalizeStep(SerializableStep):
    """Basic medical text normalization.

    - lowercases text
    - collapses whitespace
    - strips leading/trailing spaces
    - optional replacements mapping
    - applies to specific keys (if provided) else all string values
    """

    keys: Optional[List[str]] = None
    replacements: Optional[Mapping[str, str]] = None

    _type: str = field(init=False, default="MedicalTextNormalizeStep")

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(item)

        def norm(s: str) -> str:
            s2 = re.sub(r"\s+", " ", s.strip().lower())
            if self.replacements:
                for k, v in self.replacements.items():
                    s2 = s2.replace(k.lower(), v.lower())
            return s2

        if self.keys:
            for k in self.keys:
                if k in out and isinstance(out[k], str):
                    out[k] = norm(out[k])
        else:
            for k, v in list(out.items()):
                if isinstance(v, str):
                    out[k] = norm(v)
        return out


@StepRegistry.register
@dataclass
class EntityMaskStep(SerializableStep):
    """Lightweight entity recognition preprocessing via regex masking.

    Masks dates, MRN-like IDs, emails, and phone numbers commonly seen in
    medical text. Apply to provided keys or all string values.
    """

    keys: Optional[List[str]] = None
    mask_token: str = "<ENT>"

    _type: str = field(init=False, default="EntityMaskStep")

    # Simple regex patterns
    _patterns: ClassVar[List[Tuple[str, re.Pattern]]] = [
        (
            "date",
            re.compile(
                r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
            ),
        ),
        ("mrn", re.compile(r"\b\d{6,10}\b")),
        ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
        ("phone", re.compile(r"\+?\d[\d\-\s]{7,}\d")),
    ]

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(item)

        def mask_text(s: str) -> str:
            m = s
            for _, pat in self._patterns:
                m = pat.sub(self.mask_token, m)
            return m

        if self.keys:
            for k in self.keys:
                if k in out and isinstance(out[k], str):
                    out[k] = mask_text(out[k])
        else:
            for k, v in list(out.items()):
                if isinstance(v, str):
                    out[k] = mask_text(v)
        return out


@StepRegistry.register
@dataclass
class MissingDataHandlerStep(SerializableStep):
    """Handle missing data via simple strategies.

    - fill_defaults: mapping of key -> default value if missing or None/''
    - drop_if_missing_any: if any of these keys are missing/empty, drop record (raise)
    - drop_keys: remove keys entirely if present but empty
    """

    fill_defaults: Optional[Mapping[str, Any]] = None
    drop_if_missing_any: Optional[List[str]] = None
    drop_keys: Optional[List[str]] = None

    _type: str = field(init=False, default="MissingDataHandlerStep")

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(item)

        def is_empty(v: Any) -> bool:
            return v is None or (isinstance(v, str) and v.strip() == "")

        # Fill defaults
        if self.fill_defaults:
            for k, default in self.fill_defaults.items():
                if k not in out or is_empty(out.get(k)):
                    out[k] = default

        # Drop keys if empty
        if self.drop_keys:
            for k in self.drop_keys:
                if k in out and is_empty(out[k]):
                    out.pop(k, None)

        # Validate required keys
        if self.drop_if_missing_any:
            for k in self.drop_if_missing_any:
                if k not in out or is_empty(out.get(k)):
                    # Signal to pipeline to drop this item by using a sentinel
                    out["__DROP_ITEM__"] = True
                    return out

        return out


@StepRegistry.register
@dataclass
class AugmentationStep(SerializableStep):
    """Simple text augmentation.

    - token_dropout_prob: randomly drops tokens (space-delimited)
    - synonym_map: mapping of token -> synonym replacement
    Applies to provided keys or all string values.
    """

    keys: Optional[List[str]] = None
    token_dropout_prob: float = 0.0
    synonym_map: Optional[Mapping[str, str]] = None

    _type: str = field(init=False, default="AugmentationStep")

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        import random

        out = dict(item)

        def augment(s: str) -> str:
            toks = s.split()
            kept: List[str] = []
            for t in toks:
                # token dropout
                if (
                    self.token_dropout_prob > 0
                    and random.random() < self.token_dropout_prob
                ):
                    continue
                # synonym replacement
                if self.synonym_map and t in self.synonym_map:
                    kept.append(self.synonym_map[t])
                else:
                    kept.append(t)
            return " ".join(kept)

        if self.keys:
            for k in self.keys:
                if k in out and isinstance(out[k], str):
                    out[k] = augment(out[k])
        else:
            for k, v in list(out.items()):
                if isinstance(v, str):
                    out[k] = augment(v)
        return out


class DataPreprocessor:
    """
    Composable preprocessing pipeline with optional parallelism, metrics,
    and serialization support.

    Steps can be either:
    - SerializableStep instances (preferred)
    - Raw callables (backwards compatible; not serializable)
    """

    def __init__(self) -> None:
        self.steps: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []
        self._serializable_steps: List[SerializableStep] = []
        self.metrics: Dict[str, Any] = {
            "runs": 0,
            "items_processed": 0,
            "per_step": {},  # step_name -> {count, time_ms}
        }

    # --- Pipeline management ---
    def add_step(
        self, step: Callable[[Dict[str, Any]], Dict[str, Any]] | SerializableStep
    ) -> None:
        self.steps.append(step)
        if isinstance(step, SerializableStep):
            self._serializable_steps.append(step)

    def add_steps(
        self,
        steps: Iterable[Callable[[Dict[str, Any]], Dict[str, Any]] | SerializableStep],
    ) -> None:
        for s in steps:
            self.add_step(s)

    # --- Processing ---
    def process(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(data_item)
        for step in self.steps:
            t0 = time.perf_counter()
            try:
                result = step(result)
            except Exception as e:  # pragma: no cover - defensive
                logger.exception(
                    "Preprocess step failed: %s", getattr(step, "name", str(step))
                )
                raise e
            finally:
                dt = (time.perf_counter() - t0) * 1000.0
                name = getattr(step, "name", getattr(step, "__name__", str(step)))
                m = self.metrics.setdefault("per_step", {}).setdefault(
                    name, {"count": 0, "time_ms": 0.0}
                )
                m["count"] += 1
                m["time_ms"] += dt

            # Drop sentinel handling
            if result.get("__DROP_ITEM__"):
                return {"__DROPPED__": True}
        return result

    def process_batch(
        self,
        data_items: List[Dict[str, Any]],
        *,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        drop_dropped: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process a batch, optionally in parallel.

        If `parallel` is True, uses ThreadPoolExecutor. If any items are
        marked dropped, they are filtered if `drop_dropped=True`.
        """
        self.metrics["runs"] = int(self.metrics.get("runs", 0)) + 1
        self.metrics["items_processed"] = int(
            self.metrics.get("items_processed", 0)
        ) + len(data_items)

        if not parallel:
            results = [self.process(item) for item in data_items]
        else:
            results = [None] * len(data_items)  # type: ignore[list-item]
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_to_idx = {
                    ex.submit(self.process, item): i
                    for i, item in enumerate(data_items)
                }
                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    results[idx] = fut.result()
            results = [r for r in results if r is not None]  # type: ignore[assignment]

        if drop_dropped:
            results = [r for r in results if not r.get("__DROPPED__")]  # type: ignore[assignment]
        return results  # type: ignore[return-value]

    # --- Serialization ---
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self._serializable_steps],
            "metrics": self.metrics,
            "version": 1,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataPreprocessor":
        dp = cls()
        for sd in data.get("steps", []):
            step = StepRegistry.create(sd)
            dp.add_step(step)
        return dp

    @classmethod
    def from_json(cls, s: str) -> "DataPreprocessor":
        return cls.from_dict(json.loads(s))
