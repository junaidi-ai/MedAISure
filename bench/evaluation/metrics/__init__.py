"""MedAISure metrics package.

Exports the base Metric interface, common clinical metric implementations, and a
helper to obtain a default registry with all built-in metrics registered.
"""

from __future__ import annotations

from .base import Metric
from .clinical import (
    ClinicalAccuracyMetric,
    ClinicalRelevanceMetric,
    DiagnosticAccuracyMetric,
    ReasoningQualityMetric,
)
from .registry import MetricRegistry

__all__ = [
    "Metric",
    "MetricRegistry",
    "ClinicalAccuracyMetric",
    "ReasoningQualityMetric",
    "DiagnosticAccuracyMetric",
    "ClinicalRelevanceMetric",
    "get_default_registry",
]


def get_default_registry() -> MetricRegistry:
    """Return a registry pre-populated with built-in metrics."""
    reg = MetricRegistry()
    reg.register_metric(ClinicalAccuracyMetric())
    reg.register_metric(ReasoningQualityMetric())
    reg.register_metric(DiagnosticAccuracyMetric())
    reg.register_metric(ClinicalRelevanceMetric())
    return reg
