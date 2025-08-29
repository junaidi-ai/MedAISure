"""Registry for class-based MedAISure metrics.

Provides a container to register Metric instances and compute a set of metrics
by name over expected/model outputs.

Enhancements:
- Validation for registrations and lookups
- Optional parallel execution for metric calculations
- Simple in-memory cache keyed by content hash of inputs
- Aggregation helpers for combining metric dicts
- Serialization/deserialization helpers for metric results
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import concurrent.futures as _futures
import hashlib
import json
import statistics

from .base import Metric


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}
        # cache key: (metric_name, content_hash) -> float
        self._cache: Dict[Tuple[str, str], float] = {}

    def register_metric(self, metric: Metric) -> None:
        """Register a Metric instance by its unique name.

        Raises:
            ValueError: if metric is not a Metric or name duplicates existing.
        """
        if not isinstance(metric, Metric):
            raise ValueError("Only Metric instances can be registered")
        name = metric.name
        if not isinstance(name, str) or not name:
            raise ValueError("Metric must have a non-empty string name")
        if name in self._metrics:
            raise ValueError(f"Metric with name '{name}' is already registered")
        self._metrics[name] = metric

    def get_metric(self, name: str) -> Optional[Metric]:
        return self._metrics.get(name)

    def calculate_metrics(
        self,
        metric_names: List[str],
        expected_outputs: List[Dict],
        model_outputs: List[Dict],
        *,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """Calculate multiple metrics by name.

        Args:
            metric_names: list of metric names to compute.
            expected_outputs: reference outputs.
            model_outputs: model outputs.
            parallel: if True, compute metrics in a ThreadPool.
            max_workers: limit for parallel threads; ignored if parallel=False.
            use_cache: if True, use and populate the in-memory cache.

        Raises:
            KeyError: if a requested metric is not registered.
        """
        # Validate names first
        for name in metric_names:
            if name not in self._metrics:
                raise KeyError(f"Metric '{name}' is not registered")

        # Prepare a stable hash of inputs
        content_key = self._hash_io(expected_outputs, model_outputs)

        # Return cached where available
        results: Dict[str, float] = {}
        remaining: List[str] = []
        if use_cache:
            for name in metric_names:
                key = (name, content_key)
                if key in self._cache:
                    results[name] = self._cache[key]
                else:
                    remaining.append(name)
        else:
            remaining = list(metric_names)

        def _compute(name: str) -> Tuple[str, float]:
            metric = self._metrics[name]
            score = float(metric.calculate(expected_outputs, model_outputs))
            return name, score

        computed: List[Tuple[str, float]]
        if remaining:
            if parallel:
                with _futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {ex.submit(_compute, n): n for n in remaining}
                    computed = [f.result() for f in futures]
            else:
                computed = [_compute(n) for n in remaining]
        else:
            computed = []

        for name, score in computed:
            results[name] = score
            if use_cache:
                self._cache[(name, content_key)] = score

        return results

    # --------------------------
    # Utilities
    # --------------------------
    def _hash_io(self, expected_outputs: List[Dict], model_outputs: List[Dict]) -> str:
        """Create a stable content hash for caching based on inputs/outputs."""
        try:
            payload = json.dumps(
                {"expected": expected_outputs, "model": model_outputs},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            # Fallback: use repr if not JSON-serializable
            payload = repr((expected_outputs, model_outputs))
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_info(self) -> Dict[str, Any]:
        return {"entries": len(self._cache)}

    # --------------------------
    # Aggregation Helpers
    # --------------------------
    @staticmethod
    def aggregate_mean(results: Iterable[Dict[str, float]]) -> Dict[str, float]:
        """Mean aggregate across multiple metric dicts (by key)."""
        buckets: Dict[str, List[float]] = {}
        for d in results:
            for k, v in d.items():
                buckets.setdefault(k, []).append(float(v))
        return {
            k: statistics.fmean(vs) if vs else float("nan") for k, vs in buckets.items()
        }

    # --------------------------
    # Serialization Helpers
    # --------------------------
    @staticmethod
    def serialize_results(results: Dict[str, float]) -> str:
        return json.dumps(results, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def deserialize_results(payload: str) -> Dict[str, float]:
        data = json.loads(payload)
        return {k: float(v) for k, v in data.items()}
