"""Core task data models used by MedAISure.

This module defines the `TaskType` enum and the `MedicalTask` Pydantic model,
which represent the canonical schema for describing benchmark tasks.

Quick start:
    >>> from bench.models.medical_task import MedicalTask, TaskType
    >>> task = MedicalTask(
    ...     task_id="qa-demo",
    ...     task_type=TaskType.QA,
    ...     description="Answer short medical questions",
    ...     inputs=[{"question": "What is BP?"}],
    ...     expected_outputs=[{"answer": "blood pressure"}],
    ...     metrics=["accuracy"],
    ...     input_schema={"required": ["question"]},
    ...     output_schema={"required": ["answer"]},
    ... )
    >>> task.to_dict()["task_type"]
    'qa'
    >>> print(task.convert("yaml").splitlines()[0])
    schema_version: 1
"""

from __future__ import annotations

from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import csv
import json
import yaml

from pydantic import BaseModel, Field, field_validator, ValidationInfo


class TaskType(str, Enum):
    """Enumeration of supported benchmark task types.

    - DIAGNOSTIC_REASONING: Case-based reasoning → diagnosis
    - QA: Question answering
    - SUMMARIZATION: Clinical/medical text summarization
    - COMMUNICATION: Patient/provider communication generation
    """

    DIAGNOSTIC_REASONING = "diagnostic_reasoning"
    QA = "qa"
    SUMMARIZATION = "summarization"
    COMMUNICATION = "communication"


class MedicalTask(BaseModel):
    """Represents a medical evaluation task definition.

    Fields:
        schema_version: Integer version for forward compatibility (default 1).
        task_id: Unique identifier for the task (non-empty string).
        name: Optional human-friendly name (defaults to task_id when omitted).
        task_type: One of `TaskType` values.
        description: Optional free-text description.
        inputs: List of input example dicts for the task (non-empty).
        expected_outputs: List of expected output example dicts, 1:1 with inputs when both provided.
        metrics: Non-empty, unique metric names.
        input_schema/output_schema: Minimal schemas with optional `required: [..]` keys.
        dataset: Optional inline records used for lightweight examples or micro-benchmarks.

    Examples:
        Create and serialize a task
        >>> t = MedicalTask(
        ...     task_id="sum-1", task_type=TaskType.SUMMARIZATION,
        ...     inputs=[{"document": "Patient note"}],
        ...     expected_outputs=[{"summary": "Short note"}],
        ...     metrics=["rouge_l"],
        ...     input_schema={"required": ["document"]},
        ...     output_schema={"required": ["summary"]},
        ... )
        >>> isinstance(json.loads(t.to_json()), dict)
        True

        Save and load from YAML
        >>> path = Path("/tmp/medaisure_demo.yaml")
        >>> t.save(path, format="yaml")
        >>> MedicalTask.from_file(path).task_id
        'sum-1'
    """

    # Simple schema versioning for forward compatibility
    schema_version: int = 1

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
    def to_dict(
        self,
        *,
        include: Optional[set | dict] = None,
        exclude: Optional[set | dict] = None,
    ) -> Dict[str, Any]:
        """Return a plain-Python dict representation of the model.

        Supports partial serialization via include/exclude.
        """
        return self.model_dump(include=include, exclude=exclude)

    def to_json(
        self,
        indent: Optional[int] = None,
        *,
        include: Optional[set | dict] = None,
        exclude: Optional[set | dict] = None,
    ) -> str:
        """Return a JSON string representation of the model.

        Args:
            indent: Optional indentation for pretty printing.
            include/exclude: Optional fields to include/exclude.
        """
        return self.model_dump_json(indent=indent, include=include, exclude=exclude)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicalTask":
        """Create a `MedicalTask` from a plain dict with validation."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, data: str) -> "MedicalTask":
        """Create a `MedicalTask` from a JSON string with validation."""
        return cls.model_validate_json(data)

    # --- YAML helpers ---
    def to_yaml(self) -> str:
        """Return a YAML string representation of the model."""
        return yaml.safe_dump(json.loads(self.model_dump_json()), sort_keys=False)

    @classmethod
    def from_yaml(cls, data: str) -> "MedicalTask":
        """Create a `MedicalTask` from YAML string with validation.

        Ensures a default `schema_version` when absent.
        """
        payload = yaml.safe_load(data) or {}
        if "schema_version" not in payload:
            payload["schema_version"] = 1
        return cls.model_validate(payload)

    # --- CSV helpers ---
    def dataset_to_csv(self) -> str:
        """Export `dataset` to CSV string (best-effort flattening)."""
        rows = self.dataset or []
        if not rows:
            return ""
        # Compute headers union
        headers: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in headers:
                    headers.append(k)
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in headers})
        return buf.getvalue()

    # --- File I/O ---
    def save(self, file_path: str | Path, format: Optional[str] = None) -> None:
        """Persist the task to disk in JSON or YAML format.

        Args:
            file_path: Target file path; extension is used when `format` is not provided.
            format: One of {"json", "yaml", "yml"}.
        Raises:
            ValueError: If format/extension is not supported.
        """
        path = Path(file_path)
        fmt = (format or path.suffix.lstrip(".")).lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            path.write_text(self.to_json(indent=2))
        elif fmt in {"yaml", "yml"}:
            path.write_text(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format for MedicalTask.save: {fmt}")

    @classmethod
    def from_file(cls, file_path: str | Path) -> "MedicalTask":
        """Load and validate a task from a JSON or YAML file."""
        path = Path(file_path)
        text = path.read_text()
        suf = path.suffix.lower()
        if suf == ".json":
            return cls.from_json(text)
        if suf in {".yaml", ".yml"}:
            return cls.from_yaml(text)
        raise ValueError(f"Unsupported file type for MedicalTask.from_file: {suf}")

    # --- Conversion helper ---
    def convert(self, to: str) -> str:
        """Convert the task to a different textual format ("json" or "yaml")."""
        to = to.lower()
        if to == "json":
            return self.to_json(indent=2)
        if to in {"yaml", "yml"}:
            return self.to_yaml()
        raise ValueError(f"Unsupported conversion target: {to}")
