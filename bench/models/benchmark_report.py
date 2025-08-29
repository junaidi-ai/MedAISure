from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from .evaluation_result import EvaluationResult


class BenchmarkReport(BaseModel):
    """Aggregated benchmark report for a model across tasks."""

    model_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    overall_scores: Dict[str, float] = Field(default_factory=dict)
    task_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    detailed_results: List[EvaluationResult] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Internal counters to support incremental averaging; excluded from serialization
    _task_metric_counts: Dict[str, Dict[str, int]] = PrivateAttr(default_factory=dict)

    @field_validator("overall_scores")
    @classmethod
    def _validate_overall_scores(cls, v: Dict[str, Any]) -> Dict[str, float]:
        return {k: float(val) for k, val in (v or {}).items()}

    @field_validator("task_scores")
    @classmethod
    def _validate_task_scores(
        cls, v: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for task_id, metrics in (v or {}).items():
            out[task_id] = {k: float(val) for k, val in (metrics or {}).items()}
        return out

    def add_evaluation_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result and update aggregates.

        Aggregation strategy:
        - For each metric in result.metrics_results, keep an incremental average per task.
        - overall_scores is the average per metric across all tasks where that metric exists.
        """
        self.detailed_results.append(result)

        task_id = result.task_id
        if task_id not in self.task_scores:
            self.task_scores[task_id] = {}
        if task_id not in self._task_metric_counts:
            self._task_metric_counts[task_id] = {}

        # Update per-task metrics using incremental average
        for metric, value in (result.metrics_results or {}).items():
            try:
                val = float(value)
            except Exception:
                # Skip non-numeric
                continue
            prev_avg = self.task_scores[task_id].get(metric)
            count = self._task_metric_counts[task_id].get(metric, 0)
            if prev_avg is None or count == 0:
                new_avg = val
                new_count = 1
            else:
                new_count = count + 1
                new_avg = (prev_avg * count + val) / new_count
            self.task_scores[task_id][metric] = new_avg
            self._task_metric_counts[task_id][metric] = new_count

        # Recompute overall scores as average per metric across tasks
        overall: Dict[str, float] = {}
        metric_totals: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        for _task, metrics in self.task_scores.items():
            for metric, avg in metrics.items():
                metric_totals[metric] = metric_totals.get(metric, 0.0) + float(avg)
                metric_counts[metric] = metric_counts.get(metric, 0) + 1
        for metric, total in metric_totals.items():
            overall[metric] = total / metric_counts[metric]
        self.overall_scores = overall

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

    # --- Convenience serialization helpers ---
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-Python dict representation of the report."""
        return self.model_dump()

    def to_json(self, indent: int | None = None) -> str:
        """Return a JSON string representation of the report."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkReport":
        """Create a BenchmarkReport from a plain dict with validation."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, data: str) -> "BenchmarkReport":
        """Create a BenchmarkReport from a JSON string with validation."""
        return cls.model_validate_json(data)
