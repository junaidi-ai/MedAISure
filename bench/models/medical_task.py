from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ValidationInfo


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
        # ensure uniqueness and normalized spacing
        normalized = [m.strip() for m in v]
        if len(set(normalized)) != len(normalized):
            raise ValueError("metrics must be unique")
        return normalized

    @field_validator("expected_outputs")
    @classmethod
    def _validate_expected_outputs(
        cls, v: List[Dict[str, Any]], info: ValidationInfo
    ) -> List[Dict[str, Any]]:
        if v is None:
            return []
        inputs = (info.data or {}).get("inputs", [])
        # If provided, require 1:1 alignment with inputs for simple tasks
        if inputs and v and len(v) != len(inputs):
            raise ValueError(
                "expected_outputs length must match inputs length when both are provided"
            )
        return v

    @field_validator("input_schema")
    @classmethod
    def _validate_input_schema(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        # Keep schema lightweight: ensure structure is a dict and optional 'required' is a list of strings
        v = v or {}
        if not isinstance(v, dict):
            raise ValueError("input_schema must be a dictionary")
        req = v.get("required")
        if req is not None:
            if not isinstance(req, list) or not all(isinstance(k, str) for k in req):
                raise ValueError("input_schema.required must be a list of strings")
        return v

    @field_validator("output_schema")
    @classmethod
    def _validate_output_schema(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        v = v or {}
        if not isinstance(v, dict):
            raise ValueError("output_schema must be a dictionary")
        req = v.get("required")
        if req is not None:
            if not isinstance(req, list) or not all(isinstance(k, str) for k in req):
                raise ValueError("output_schema.required must be a list of strings")
        return v

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(
        cls, v: List[Dict[str, Any]], info: ValidationInfo
    ) -> List[Dict[str, Any]]:
        # Optional dataset; if provided, check minimal key presence per schemas
        v = v or []
        inp_req = []
        out_req = []
        data = info.data or {}
        try:
            inp_req = list(
                (data.get("input_schema", {}) or {}).get("required", []) or []
            )
            out_req = list(
                (data.get("output_schema", {}) or {}).get("required", []) or []
            )
        except Exception:
            # If schemas malformed, other validators will raise
            inp_req = []
            out_req = []
        for i, row in enumerate(v):
            if not isinstance(row, dict):
                raise ValueError(f"dataset[{i}] must be a dict")
            # If dataset rows carry nested 'input'/'output', validate keys
            if inp_req and isinstance(row.get("input"), dict):
                missing = [k for k in inp_req if k not in row["input"]]
                if missing:
                    raise ValueError(
                        f"dataset[{i}].input missing required keys: {missing}"
                    )
            if out_req and isinstance(row.get("output"), dict):
                missing = [k for k in out_req if k not in row["output"]]
                if missing:
                    raise ValueError(
                        f"dataset[{i}].output missing required keys: {missing}"
                    )
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
