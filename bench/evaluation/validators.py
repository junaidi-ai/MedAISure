"""Validation helpers and minimal default schemas for tasks.

Key utilities:
- `DEFAULT_SCHEMAS`: Minimal required key sets per `TaskType`.
- `ensure_task_schemas(...)`: Fill in missing schemas using defaults.
- `validate_task_dataset(task)`: Validate inline `task.dataset` rows against schemas.

Examples:
    >>> from bench.models.task_types import ClinicalSummarizationTask
    >>> t = ClinicalSummarizationTask("sum-demo")
    >>> t.dataset = [
    ...     {"input": {"document": "Patient has HTN."}, "output": {"summary": "HTN noted."}}
    ... ]
    >>> validate_task_dataset(t)  # no exception

Flat-row dataset:
    >>> from bench.models.task_types import MedicalQATask
    >>> qa = MedicalQATask("qa-demo")
    >>> qa.dataset = [
    ...     {"question": "What is BP?"},
    ... ]
    >>> validate_task_dataset(qa)  # validates against input_schema only
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..models.medical_task import TaskType


# Default minimal schemas per TaskType (used if task doesn't specify schemas)
# Note: SUMMARIZATION inputs default to "document" to align with
# `ClinicalSummarizationTask` expectations.
DEFAULT_SCHEMAS: Dict[TaskType, Tuple[Dict[str, Any], Dict[str, Any]]] = {
    TaskType.QA: ({"required": ["question"]}, {"required": ["answer"]}),
    TaskType.SUMMARIZATION: ({"required": ["document"]}, {"required": ["summary"]}),
    TaskType.DIAGNOSTIC_REASONING: (
        {"required": ["symptoms"]},
        {"required": ["diagnosis"]},
    ),
    TaskType.COMMUNICATION: ({"required": ["prompt"]}, {"required": ["response"]}),
}


def _required_keys(schema: Dict[str, Any]) -> List[str]:
    req = schema.get("required") or []
    return [str(k) for k in req if isinstance(k, (str, int))]


def validate_record_against_schema(
    record: Dict[str, Any],
    schema: Dict[str, Any],
    *,
    label: str,
    index: int | None = None,
) -> None:
    """Validate a single flat dict against a minimal schema.

    Currently enforces presence of required keys only.

    Args:
        record: Flat dict to validate.
        schema: Minimal schema with optional `required: [..]`.
        label: Human-friendly label for error messages.
        index: Optional row index for error messages.
    Raises:
        ValueError: If required keys are missing.
    """
    if not schema:
        return
    required = _required_keys(schema)
    if not required:
        return
    missing = [k for k in required if k not in record]
    if missing:
        where = f"{label}[{index}]" if index is not None else label
        raise ValueError(f"{where} missing required keys: {missing}")


def ensure_task_schemas(
    task_type: TaskType,
    input_schema: Dict[str, Any] | None,
    output_schema: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return input/output schemas ensuring defaults by TaskType when empty.

    If a schema is an empty dict or None, substitute with task-type defaults.
    """
    in_s = input_schema or {}
    out_s = output_schema or {}
    if not in_s or not out_s:
        din, dout = DEFAULT_SCHEMAS.get(task_type, ({}, {}))
        if not in_s:
            in_s = din
        if not out_s:
            out_s = dout
    return in_s, out_s


def validate_task_dataset(task: Any) -> None:
    """Validate a task's inline dataset records against its schemas.

    Supports rows shaped as either flat records or {"input": {...}, "output": {...}}.

    Raises:
        ValueError: When rows are not dicts or required keys are missing.
    """
    in_schema, out_schema = ensure_task_schemas(
        task.task_type, task.input_schema, task.output_schema
    )

    for i, row in enumerate(task.dataset or []):
        if not isinstance(row, dict):
            raise ValueError(f"dataset[{i}] must be a dict")
        # Nested form
        if "input" in row or "output" in row:
            if "input" in row and isinstance(row["input"], dict):
                validate_record_against_schema(
                    row["input"], in_schema, label="dataset.input", index=i
                )
            if "output" in row and isinstance(row["output"], dict):
                validate_record_against_schema(
                    row["output"], out_schema, label="dataset.output", index=i
                )
        else:
            # Flat form: validate against input schema only (typical for inputs-only datasets)
            validate_record_against_schema(row, in_schema, label="dataset", index=i)
