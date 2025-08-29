"""Base interfaces for the MedAISure metrics system.

Defines an abstract Metric interface that all concrete metrics must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class Metric(ABC):
    """Abstract base class for all metrics.

    Implementations must provide a unique ``name`` and a ``calculate`` method
    that returns a numeric score in the range [0, 1] when applicable. The
    contract is intentionally lightweight to support simple, dependency-light
    metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name used for registry lookup."""
        raise NotImplementedError

    @abstractmethod
    def calculate(
        self, expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> float:
        """Compute the metric score.

        Args:
            expected_outputs: List of ground-truth dictionaries.
            model_outputs: List of model output dictionaries.

        Returns:
            A float score. Implementations should return ``float('nan')`` when
            a score cannot be computed.
        """
        raise NotImplementedError

    @staticmethod
    def validate_inputs(
        expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> None:
        """Validate basic preconditions for metric calculation.

        Checks:
        - Both inputs are lists
        - Both lists are non-empty
        - Lists have equal length
        - All elements are dictionaries

        Raises:
            TypeError: If inputs are not lists or contain non-dict elements
            ValueError: If lists are empty or lengths mismatch
        """
        if not isinstance(expected_outputs, list) or not isinstance(
            model_outputs, list
        ):
            raise TypeError("expected_outputs and model_outputs must be lists")
        if len(expected_outputs) == 0 or len(model_outputs) == 0:
            raise ValueError("expected_outputs and model_outputs must be non-empty")
        if len(expected_outputs) != len(model_outputs):
            raise ValueError(
                "expected_outputs and model_outputs must have the same length"
            )
        if any(not isinstance(x, dict) for x in expected_outputs):
            raise TypeError("All items in expected_outputs must be dicts")
        if any(not isinstance(x, dict) for x in model_outputs):
            raise TypeError("All items in model_outputs must be dicts")
