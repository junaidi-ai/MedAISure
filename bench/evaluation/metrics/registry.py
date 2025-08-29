"""Registry for class-based MedAISure metrics.

Provides a simple container to register Metric instances and compute a set of
metrics by name over expected/model outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import Metric


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}

    def register_metric(self, metric: Metric) -> None:
        """Register a Metric instance by its unique name."""
        self._metrics[metric.name] = metric

    def get_metric(self, name: str) -> Optional[Metric]:
        return self._metrics.get(name)

    def calculate_metrics(
        self,
        metric_names: List[str],
        expected_outputs: List[Dict],
        model_outputs: List[Dict],
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name in metric_names:
            metric = self.get_metric(name)
            if metric is not None:
                results[name] = float(metric.calculate(expected_outputs, model_outputs))
        return results
