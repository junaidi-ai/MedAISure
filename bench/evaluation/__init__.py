"""Core evaluation framework for MedAISure benchmark.

This package uses lazy imports to avoid importing heavy dependencies
at initialization time. Submodules are only imported when their
attributes are accessed.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "TaskLoader",
    "ModelRunner",
    "MetricCalculator",
    "ResultAggregator",
    "EvaluationHarness",
    "Metric",
    "MetricRegistry",
]


_ATTR_TO_MODULE = {
    "TaskLoader": "bench.evaluation.task_loader",
    "ModelRunner": "bench.evaluation.model_runner",
    "MetricCalculator": "bench.evaluation.metric_calculator",
    "ResultAggregator": "bench.evaluation.result_aggregator",
    "EvaluationHarness": "bench.evaluation.harness",
    "Metric": "bench.evaluation.metrics.base",
    "MetricRegistry": "bench.evaluation.metrics.registry",
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name in _ATTR_TO_MODULE:
        mod = import_module(_ATTR_TO_MODULE[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
