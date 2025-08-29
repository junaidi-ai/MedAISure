from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING, Optional

import csv
import json
import yaml

from pydantic import BaseModel, Field, ValidationInfo, field_validator

if TYPE_CHECKING:  # Avoid runtime import cycles; satisfies linters and type checkers
    from .medical_task import MedicalTask


class EvaluationResult(BaseModel):
    """Represents the result of evaluating a model on a specific task."""

    # Simple schema versioning for forward compatibility
    schema_version: int = 1

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
        if not isinstance(v, list):
            raise ValueError("inputs must be a list of dicts")
        for i, item in enumerate(v):
            if not isinstance(item, dict):
                raise ValueError(f"inputs[{i}] must be a dict")
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
        if not isinstance(v, list):
            raise ValueError("model_outputs must be a list of dicts")
        for i, item in enumerate(v):
            if not isinstance(item, dict):
                raise ValueError(f"model_outputs[{i}] must be a dict")
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

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: datetime) -> datetime:
        # Ensure timezone-aware; coerce naive to UTC
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    # --- Relationship checks (opt-in) ---
    def validate_against_task(self, task: "MedicalTask") -> None:
        """Validate relationships and basic schema alignment with a `MedicalTask`.

        - Ensure `task_id` matches
        - If task.metrics provided, ensure all reported metrics exist in task.metrics
        - If task.input_schema.required present, ensure each input dict has required keys
        - If task.output_schema.required present, ensure each model_output dict has required keys
        """
        if self.task_id != task.task_id:
            raise ValueError(
                f"task_id mismatch: result={self.task_id} task={task.task_id}"
            )

        if task.metrics:
            unknown = [
                m for m in (self.metrics_results or {}).keys() if m not in task.metrics
            ]
            if unknown:
                raise ValueError(
                    f"metrics_results contains metrics not defined by task: {unknown}"
                )

        req_in = (task.input_schema or {}).get("required") or []
        if req_in:
            for i, rec in enumerate(self.inputs or []):
                missing = [k for k in req_in if k not in rec]
                if missing:
                    raise ValueError(
                        f"inputs[{i}] missing required keys per task.input_schema: {missing}"
                    )

        req_out = (task.output_schema or {}).get("required") or []
        if req_out:
            for i, rec in enumerate(self.model_outputs or []):
                missing = [k for k in req_out if k not in rec]
                if missing:
                    raise ValueError(
                        f"model_outputs[{i}] missing required keys per task.output_schema: {missing}"
                    )

    # --- Convenience serialization helpers ---
    def to_dict(
        self,
        *,
        include: Optional[set | dict] = None,
        exclude: Optional[set | dict] = None,
    ) -> Dict[str, Any]:
        """Return a plain-Python dict representation of the model.

        Supports partial serialization via `include`/`exclude` like pydantic's model_dump.
        """
        return self.model_dump(include=include, exclude=exclude)

    def to_json(
        self,
        indent: int | None = None,
        *,
        include: Optional[set | dict] = None,
        exclude: Optional[set | dict] = None,
    ) -> str:
        """Return a JSON string representation of the model with optional include/exclude."""
        return self.model_dump_json(indent=indent, include=include, exclude=exclude)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create an EvaluationResult from a plain dict with validation."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, data: str) -> "EvaluationResult":
        """Create an EvaluationResult from a JSON string with validation."""
        return cls.model_validate_json(data)

    # --- YAML helpers ---
    def to_yaml(self) -> str:
        """Return a YAML string representation of the model."""
        return yaml.safe_dump(json.loads(self.model_dump_json()), sort_keys=False)

    @classmethod
    def from_yaml(cls, data: str) -> "EvaluationResult":
        """Create an EvaluationResult from a YAML string with validation."""
        payload = yaml.safe_load(data) or {}
        # Basic schema version normalization
        if "schema_version" not in payload:
            payload["schema_version"] = 1
        return cls.model_validate(payload)

    # --- CSV helpers for tabular fields ---
    def _list_of_dicts_to_csv(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        # Collect all headers seen across rows for robustness
        headers: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in headers:
                    headers.append(k)
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: self._coerce_csv_value(r.get(k)) for k in headers})
        return buf.getvalue()

    @staticmethod
    def _coerce_csv_value(v: Any) -> Any:
        # Keep primitives, stringify others (e.g., datetime) via ISO or str
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    def inputs_to_csv(self) -> str:
        """Export `inputs` list of dicts to CSV string."""
        return self._list_of_dicts_to_csv(self.inputs or [])

    def outputs_to_csv(self) -> str:
        """Export `model_outputs` list of dicts to CSV string."""
        return self._list_of_dicts_to_csv(self.model_outputs or [])

    @classmethod
    def from_inputs_csv(
        cls, model_id: str, task_id: str, csv_text: str
    ) -> "EvaluationResult":
        """Create instance with `inputs` populated from CSV; other fields empty."""
        reader = csv.DictReader(StringIO(csv_text))
        inputs = [dict(row) for row in reader]
        return cls(model_id=model_id, task_id=task_id, inputs=inputs)

    @classmethod
    def from_outputs_csv(
        cls, model_id: str, task_id: str, csv_text: str
    ) -> "EvaluationResult":
        """Create instance with `model_outputs` populated from CSV; other fields empty."""
        reader = csv.DictReader(StringIO(csv_text))
        outs = [dict(row) for row in reader]
        return cls(model_id=model_id, task_id=task_id, model_outputs=outs)

    # --- File I/O with format selection ---
    def save(self, file_path: str | Path, format: Optional[str] = None) -> None:
        """Save to a file; format inferred from extension if not provided.

        Supported formats: json, yaml.
        """
        path = Path(file_path)
        fmt = (format or path.suffix.lstrip(".")).lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            path.write_text(self.to_json(indent=2))
        elif fmt in {"yaml", "yml"}:
            path.write_text(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format for EvaluationResult.save: {fmt}")

    @classmethod
    def from_file(cls, file_path: str | Path) -> "EvaluationResult":
        path = Path(file_path)
        suffix = path.suffix.lower()
        text = path.read_text()
        if suffix == ".json":
            return cls.from_json(text)
        if suffix in {".yaml", ".yml"}:
            return cls.from_yaml(text)
        raise ValueError(
            f"Unsupported file type for EvaluationResult.from_file: {suffix}"
        )

    # --- Conversion helpers ---
    def convert(self, to: str) -> str:
        """Convert the whole object to a target format string (json|yaml)."""
        to = to.lower()
        if to == "json":
            return self.to_json(indent=2)
        if to in {"yaml", "yml"}:
            return self.to_yaml()
        raise ValueError(f"Unsupported conversion target: {to}")

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
