from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    DIAGNOSTIC_REASONING = "diagnostic_reasoning"
    QA = "qa"
    SUMMARIZATION = "summarization"
    COMMUNICATION = "communication"


class MedicalTask(BaseModel):
    """Represents a medical evaluation task definition."""

    task_id: str
    name: Optional[str] = None
    task_type: TaskType
    description: str = ""

    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    expected_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)

    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)

    # Inline dataset samples for lightweight tasks/examples
    dataset: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator("task_type", mode="before")
    @classmethod
    def _validate_task_type(cls, v: Any) -> TaskType:
        if isinstance(v, TaskType):
            return v
        if isinstance(v, str):
            try:
                return TaskType(v)
            except Exception:
                raise ValueError("invalid task_type")
        raise ValueError("invalid task_type")

    @field_validator("task_id")
    @classmethod
    def _non_empty_task_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("task_id must be a non-empty string")
        return v

    @field_validator("inputs")
    @classmethod
    def _validate_inputs(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if v is None or len(v) == 0:
            raise ValueError("inputs must not be empty")
        return v

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, v: List[str]) -> List[str]:
        if any((m is None) or (isinstance(m, str) and m.strip() == "") for m in v):
            raise ValueError("metrics must not contain empty values")
        return v

    # --- Convenience serialization helpers ---
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-Python dict representation of the model."""
        return self.model_dump()

    def to_json(self, indent: Optional[int] = None) -> str:
        """Return a JSON string representation of the model."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicalTask":
        """Create a MedicalTask from a plain dict with validation."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, data: str) -> "MedicalTask":
        """Create a MedicalTask from a JSON string with validation."""
        return cls.model_validate_json(data)
