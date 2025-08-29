from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from .evaluation_result import EvaluationResult
from .medical_task import MedicalTask


class BenchmarkReport(BaseModel):
    """Aggregated benchmark report for a model across tasks."""

    model_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the report was created.",
    )

    overall_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Average metric scores across all tasks (metric -> mean score).",
    )
    task_scores: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-task average scores (task_id -> {metric -> mean score}).",
    )
    detailed_results: List[EvaluationResult] = Field(
        default_factory=list,
        description="Raw evaluation results used to compute aggregates.",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this report (e.g., run_id, notes).",
    )

    # Internal counters to support incremental averaging; excluded from serialization
    _task_metric_counts: Dict[str, Dict[str, int]] = PrivateAttr(default_factory=dict)

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("model_id must be a non-empty string")
        return v

    @field_validator("detailed_results")
    @classmethod
    def _validate_detailed_results(
        cls, v: List[EvaluationResult]
    ) -> List[EvaluationResult]:
        v = v or []
        if not isinstance(v, list):
            raise ValueError("detailed_results must be a list")
        # Pydantic will coerce, but ensure each item is an EvaluationResult instance
        for i, item in enumerate(v):
            if not isinstance(item, EvaluationResult):
                raise ValueError(f"detailed_results[{i}] must be an EvaluationResult")
        return v

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

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

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "BenchmarkReport":
        # Ensure all detailed_results share the same model_id
        for i, res in enumerate(self.detailed_results or []):
            if res.model_id != self.model_id:
                raise ValueError(
                    f"detailed_results[{i}].model_id ({res.model_id}) does not match report.model_id ({self.model_id})"
                )
        return self

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

    # --- Relationship validation helpers ---
    def validate_against_tasks(self, tasks: Dict[str, MedicalTask]) -> None:
        """Validate that results align with provided task definitions.

        - Each EvaluationResult.task_id must exist in tasks
        - Each EvaluationResult must pass `validate_against_task` against its task
        - Warn via exception if task_scores contains metrics for unknown tasks
        """
        # Check detailed_results against tasks
        for i, res in enumerate(self.detailed_results or []):
            task = tasks.get(res.task_id)
            if task is None:
                raise ValueError(
                    f"Unknown task_id in detailed_results[{i}]: {res.task_id}"
                )
            res.validate_against_task(task)

        # Ensure task_scores only references known tasks and are numeric
        for task_id, metrics in (self.task_scores or {}).items():
            if task_id not in tasks:
                raise ValueError(f"task_scores references unknown task_id: {task_id}")
            for name, val in (metrics or {}).items():
                try:
                    float(val)
                except Exception:
                    raise ValueError(f"task_scores[{task_id}][{name}] must be numeric")

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

    # --- Summary/statistics helpers ---
    def summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute simple summary stats (count, mean, min, max) per metric.

        The computation is based on `task_scores` (per-task averages). This provides
        a quick overview without re-walking raw `detailed_results`.
        """
        stats: Dict[str, Dict[str, float]] = {}
        # Collect all metric values across tasks
        metric_values: Dict[str, List[float]] = {}
        for _task, metrics in (self.task_scores or {}).items():
            for name, val in (metrics or {}).items():
                metric_values.setdefault(name, []).append(float(val))

        for name, values in metric_values.items():
            if not values:
                continue
            vmin = min(values)
            vmax = max(values)
            mean = sum(values) / len(values)
            stats[name] = {
                "count": float(len(values)),
                "mean": mean,
                "min": vmin,
                "max": vmax,
            }
        return stats

    def metric_history(self, metric: str) -> List[float]:
        """Return the sequence of per-task average values for a given metric."""
        vals: List[float] = []
        for _task, metrics in (self.task_scores or {}).items():
            if metric in metrics:
                vals.append(float(metrics[metric]))
        return vals

    # --- Visualization helpers (optional) ---
    def plot_overall_scores(self) -> None:
        """Plot a simple bar chart of overall metric scores.

        Requires matplotlib; if not available, raises an informative ImportError.
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            raise ImportError(
                "matplotlib is required for plotting. Install it or use to_json()/to_dict() to export data."
            ) from e

        metrics = list((self.overall_scores or {}).keys())
        values = [self.overall_scores[m] for m in metrics]
        plt.figure(figsize=(6, 3))
        plt.bar(metrics, values)
        plt.title("Overall Scores")
        plt.ylabel("Score")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_task_scores(self, metric: str) -> None:
        """Plot a simple bar chart of per-task scores for a selected metric.

        Requires matplotlib; if not available, raises an informative ImportError.
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            raise ImportError(
                "matplotlib is required for plotting. Install it or use to_json()/to_dict() to export data."
            ) from e

        tasks = []
        values = []
        for task_id, metrics in (self.task_scores or {}).items():
            if metric in metrics:
                tasks.append(task_id)
                values.append(metrics[metric])

        plt.figure(figsize=(6, 3))
        plt.bar(tasks, values)
        plt.title(f"Per-task scores: {metric}")
        plt.ylabel("Score")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()
