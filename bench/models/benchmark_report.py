from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .evaluation_result import EvaluationResult


class BenchmarkReport(BaseModel):
    """Aggregated benchmark report for a model across tasks."""

    model_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    overall_scores: Dict[str, float] = Field(default_factory=dict)
    task_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    detailed_results: List[EvaluationResult] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("overall_scores")
    @classmethod
    def _validate_overall_scores(cls, v: Dict[str, Any]) -> Dict[str, float]:
        return {k: float(val) for k, val in (v or {}).items()}

    @field_validator("task_scores")
    @classmethod
    def _validate_task_scores(cls, v: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for task_id, metrics in (v or {}).items():
            out[task_id] = {k: float(val) for k, val in (metrics or {}).items()}
        return out

    def add_evaluation_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result and update aggregates."""
        self.detailed_results.append(result)

        # Default simple score: if metrics_results present, use first numeric value
        score_value: Optional[float] = None
        for v in result.metrics_results.values():
            if isinstance(v, (int, float)):
                score_value = float(v)
                break

        if score_value is None:
            return

        # Update per-task score as simple average over results for that task
        task_id = result.task_id
        existing = self.task_scores.get(task_id, {})
        if existing:
            # Average of existing metrics if same keys, otherwise store/overwrite 'average_score'
            if "average_score" in existing:
                prev = existing["average_score"]
                # naive two-point average; in practice, we'd track counts
                existing["average_score"] = (prev + score_value) / 2.0
            else:
                existing["average_score"] = score_value
        else:
            existing = {"average_score": score_value}
        self.task_scores[task_id] = existing

        # Update overall average over all tasks' average_score
        task_avgs = [m.get("average_score", 0.0) for m in self.task_scores.values()]
        if task_avgs:
            self.overall_scores["accuracy"] = sum(task_avgs) / len(task_avgs)

    # Persistence helpers used by the harness and docs
    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump()
        # Ensure datetime is serialized
        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)

    # Backwards-compatible alias used in docs
    def to_file(self, file_path: Union[str, Path]) -> None:
        self.save(file_path)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "BenchmarkReport":
        path = Path(file_path)
        with path.open("r") as f:
            data = json.load(f)
        return cls.model_validate(data)
