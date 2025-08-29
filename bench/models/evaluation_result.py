from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class EvaluationResult(BaseModel):
    """Represents the result of evaluating a model on a specific task."""

    model_id: str = Field(
        ..., description="Identifier for the evaluated model (e.g., name or hash)."
    )
    task_id: str = Field(
        ..., description="Identifier of the benchmark task the results correspond to."
    )

    inputs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of raw input records used for evaluation (order matters).",
    )
    model_outputs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of model outputs aligned 1:1 with inputs.",
    )

    metrics_results: Dict[str, float] = Field(
        default_factory=dict,
        description="Flat mapping of metric name to numeric score for this task.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional run metadata (e.g., model_version, seed, notes).",
    )

    # Explicit timestamp field; tests expect a datetime instance
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this result was created.",
    )

    @field_validator("inputs")
    @classmethod
    def _validate_inputs(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if v is None:
            return []
        return v

    @field_validator("model_id", "task_id")
    @classmethod
    def _validate_non_empty_ids(cls, v: str, info: ValidationInfo) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return v

    @field_validator("model_outputs")
    @classmethod
    def _validate_model_outputs(
        cls, v: List[Dict[str, Any]], info: ValidationInfo
    ) -> List[Dict[str, Any]]:
        if v is None:
            return []
        data = info.data or {}
        inputs = data.get("inputs", [])
        if inputs and len(v) != len(inputs):
            raise ValueError("model_outputs length must match inputs length")
        return v

    @field_validator("metrics_results")
    @classmethod
    def _validate_metrics_results(cls, v: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, val in (v or {}).items():
            if not isinstance(val, (int, float)):
                raise ValueError("metrics_results values must be numeric")
            out[k] = float(val)
        return out

    # --- Convenience serialization helpers ---
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-Python dict representation of the model."""
        return self.model_dump()

    def to_json(self, indent: int | None = None) -> str:
        """Return a JSON string representation of the model."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create an EvaluationResult from a plain dict with validation."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, data: str) -> "EvaluationResult":
        """Create an EvaluationResult from a JSON string with validation."""
        return cls.model_validate_json(data)

    # --- Simple summary/statistics helpers ---
    def metric_summary(self) -> Dict[str, Dict[str, float]]:
        """Return simple summary stats for each metric.

        For each metric in `metrics_results`, compute:
        - count: number of contributing values (always 1 for a single task result)
        - mean, min, max: identical to the metric value for this single result

        These helpers are intentionally simple; higher-level aggregation should
        be performed by `BenchmarkReport`.
        """
        summary: Dict[str, Dict[str, float]] = {}
        for name, value in (self.metrics_results or {}).items():
            val = float(value)
            summary[name] = {
                "count": 1.0,
                "mean": val,
                "min": val,
                "max": val,
            }
        return summary

    def item_count(self) -> int:
        """Return the number of evaluated items (aligned pairs of input/output)."""
        # Lengths are validated to match by `_validate_model_outputs`.
        return max(len(self.inputs or []), len(self.model_outputs or []))

    def has_metric(self, name: str) -> bool:
        """Return True if a metric with the given name exists in results."""
        return name in (self.metrics_results or {})
