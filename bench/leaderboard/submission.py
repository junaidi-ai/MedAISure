"""Leaderboard submission schema, builder, and validator utilities.

The submission format captures predictions (and optional reasoning traces) per task.
It is intentionally minimal and self-contained to avoid heavy dependencies.

Schema (v1):
{
  "schema_version": 1,
  "run_id": "<string>",
  "model_id": "<string>",
  "created_at": "<ISO 8601 datetime>",
  "submissions": [
    {
      "task_id": "<string>",
      "items": [
        {
          "input_id": "<string>",   # index as string when no natural id present
          "prediction": { ... },     # required dict
          "reasoning": "..."        # optional string or omitted
        }
      ]
    }
  ]
}

Validation here focuses on structure and required keys. It does not enforce a full
JSON Schema draft. For stricter needs, this module can be extended or integrated
with jsonschema in the future without breaking the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..models.benchmark_report import BenchmarkReport

# JSON Schema for strict validation
SUBMISSION_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "run_id", "model_id", "created_at", "submissions"],
    "properties": {
        "schema_version": {"type": "integer", "minimum": 1},
        "run_id": {"type": "string", "minLength": 1},
        "model_id": {"type": "string", "minLength": 1},
        "created_at": {"type": "string"},
        "submissions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["task_id", "items"],
                "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["input_id", "prediction"],
                            "properties": {
                                "input_id": {"type": "string", "minLength": 1},
                                "prediction": {"type": "object", "minProperties": 1},
                                "reasoning": {"type": "string", "minLength": 1},
                            },
                            "additionalProperties": True,
                        },
                    },
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class SubmissionSchema:
    version: int = 1

    @staticmethod
    def required_top_keys() -> List[str]:
        return ["schema_version", "run_id", "model_id", "created_at", "submissions"]

    @staticmethod
    def required_submission_keys() -> List[str]:
        return ["task_id", "items"]

    @staticmethod
    def required_item_keys() -> List[str]:
        return ["input_id", "prediction"]


def _iso_now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_submission_from_report(
    report: BenchmarkReport,
    *,
    include_reasoning: bool = True,
) -> Dict[str, Any]:
    """Transform a BenchmarkReport into a leaderboard submission payload.

    - Each task in report.detailed_results is grouped by task_id
    - Each aligned pair of input/output becomes an item with an input_id and prediction
    - If outputs include fields like "reasoning" or "trace", include them when enabled
    """
    # Group results by task
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for res in report.detailed_results or []:
        # Prepare aligned pairs from inputs/model_outputs
        n = max(len(res.inputs or []), len(res.model_outputs or []))
        for i in range(n):
            inp = (res.inputs or [{}])[i] if i < len(res.inputs or []) else {}
            out = (
                (res.model_outputs or [{}])[i]
                if i < len(res.model_outputs or [])
                else {}
            )
            # Determine input_id: prefer explicit id key if present, else index
            input_id = str(inp.get("id", i)) if isinstance(inp, dict) else str(i)
            item: Dict[str, Any] = {
                "input_id": input_id,
                "prediction": out if isinstance(out, dict) else {"output": out},
            }
            if include_reasoning and isinstance(out, dict):
                # Heuristics for common reasoning trace fields
                for k in ("reasoning", "chain_of_thought", "trace", "rationale"):
                    if k in out and out[k] is not None:
                        # Store as simple string when possible
                        item["reasoning"] = (
                            out[k] if isinstance(out[k], str) else str(out[k])
                        )
                        break
            by_task.setdefault(res.task_id, []).append(item)

    submissions = [
        {"task_id": task_id, "items": items} for task_id, items in by_task.items()
    ]

    payload: Dict[str, Any] = {
        "schema_version": SubmissionSchema.version,
        "run_id": str((report.metadata or {}).get("run_id", "unknown")),
        "model_id": report.model_id,
        "created_at": _iso_now_utc(),
        "submissions": submissions,
    }
    return payload


def validate_submission(payload: Dict[str, Any]) -> None:
    """Validate the submission payload against a JSON Schema (strict),
    with a minimal structural fallback if jsonschema is unavailable.

    Raises ValueError with a descriptive message on first failure.
    """
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(instance=payload, schema=SUBMISSION_JSON_SCHEMA)
        return
    except ImportError:
        # Fallback: minimal structural checks (previous behavior)
        pass

    if not isinstance(payload, dict):
        raise ValueError("submission must be a JSON object")
    for k in SubmissionSchema.required_top_keys():
        if k not in payload:
            raise ValueError(f"missing top-level field: {k}")
    if not isinstance(payload["schema_version"], int) or payload["schema_version"] < 1:
        raise ValueError("schema_version must be an integer >= 1")
    if not isinstance(payload["run_id"], str) or not payload["run_id"].strip():
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(payload["model_id"], str) or not payload["model_id"].strip():
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(payload.get("submissions"), list):
        raise ValueError("submissions must be an array")
    if len(payload.get("submissions") or []) == 0:
        raise ValueError("submissions must be a non-empty array")
    for i, sub in enumerate(payload["submissions"]):
        if not isinstance(sub, dict):
            raise ValueError(f"submissions[{i}] must be an object")
        for k in SubmissionSchema.required_submission_keys():
            if k not in sub:
                raise ValueError(f"submissions[{i}] missing field: {k}")
        if not isinstance(sub["task_id"], str) or not sub["task_id"].strip():
            raise ValueError(f"submissions[{i}].task_id must be a non-empty string")
        items = sub["items"]
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError(f"submissions[{i}].items must be a non-empty array")
        for j, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"submissions[{i}].items[{j}] must be an object")
            for k in SubmissionSchema.required_item_keys():
                if k not in item:
                    raise ValueError(f"submissions[{i}].items[{j}] missing field: {k}")
            if not isinstance(item["input_id"], str) or not item["input_id"].strip():
                raise ValueError(
                    f"submissions[{i}].items[{j}].input_id must be a non-empty string"
                )
            pred = item["prediction"]
            if not isinstance(pred, dict) or len(pred) == 0:
                raise ValueError(
                    f"submissions[{i}].items[{j}].prediction must be a non-empty object"
                )
            if "reasoning" in item and not (
                isinstance(item["reasoning"], str) and item["reasoning"].strip()
            ):
                raise ValueError(
                    f"submissions[{i}].items[{j}].reasoning must be a non-empty string when present"
                )


def build_and_validate_submission(
    report: BenchmarkReport,
    *,
    include_reasoning: bool = True,
) -> Dict[str, Any]:
    payload = build_submission_from_report(report, include_reasoning=include_reasoning)
    validate_submission(payload)
    return payload
