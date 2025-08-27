from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class EvaluationResult(BaseModel):
    """Represents the result of evaluating a model on a specific task."""

    model_id: str
    task_id: str

    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    model_outputs: List[Dict[str, Any]] = Field(default_factory=list)

    metrics_results: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Explicit timestamp field; tests expect a datetime instance
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("inputs")
    @classmethod
    def _validate_inputs(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if v is None:
            return []
        return v

    @field_validator("model_outputs")
    @classmethod
    def _validate_model_outputs(cls, v: List[Dict[str, Any]], values: Dict[str, Any]) -> List[Dict[str, Any]]:
        if v is None:
            return []
        inputs = values.get("inputs", [])
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
