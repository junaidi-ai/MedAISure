"""Result aggregation for MEDDSAI benchmark."""

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..models.benchmark_report import BenchmarkReport
from ..models.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates and manages evaluation results across multiple tasks.

    This class handles the collection, aggregation, and persistence of evaluation
    results, including computing summary statistics across tasks.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize the ResultAggregator.

        Args:
            output_dir: Directory to save output reports. If None, uses a default
                location.
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage of results
        self.reports: Dict[str, BenchmarkReport] = {}

    def add_evaluation_result(
        self,
        evaluation_result: EvaluationResult,
        run_id: Optional[str] = None,
    ) -> None:
        """Add an evaluation result to the aggregator.

        Args:
            evaluation_result: The evaluation result to add
            run_id: Optional run ID to associate with this result. If not provided,
                   a new run will be created.
        """
        if run_id is None:
            # Generate a deterministic run ID if not provided
            run_id = self.generate_run_id(
                model_name=evaluation_result.metadata.get("model_name", "unknown"),
                task_ids=[evaluation_result.task_id],
                timestamp=evaluation_result.timestamp.isoformat(),
            )

        if run_id not in self.reports:
            # Create a new report for this run
            self.reports[run_id] = BenchmarkReport(
                model_id=evaluation_result.model_id,
                timestamp=evaluation_result.timestamp,
                overall_scores={},
                task_scores={},
                detailed_results=[],
                metadata={
                    "run_id": run_id,
                    "start_time": evaluation_result.timestamp.isoformat(),
                },
            )

        # Add the evaluation result to the report
        self.reports[run_id].add_evaluation_result(evaluation_result)

    def get_report(self, run_id: str) -> BenchmarkReport:
        """Get a benchmark report for the given run ID.

        Args:
            run_id: The run ID to get the report for.

        Returns:
            The BenchmarkReport for the specified run.

        Raises:
            ValueError: If no report exists for the given run_id.
        """
        if run_id not in self.reports:
            raise ValueError(f"No report found for run {run_id}")
        return self.reports[run_id]

    def save_report(
        self, report: BenchmarkReport, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Save a benchmark report to disk.

        Args:
            report: The report to save.
            output_path: Path to save the report. If None, generates a filename.

        Returns:
            Path to the saved report.
        """
        if output_path is None:
            # Generate a filename based on model name and timestamp
            safe_model_name = "".join(
                c if c.isalnum() else "_" for c in report.model_name
            )
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{safe_model_name}_{timestamp}.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save(output_path)
        return output_path

    def _aggregate_metrics(self, run_id: str) -> Dict[str, float]:
        """Compute aggregate metrics across all tasks in a run.

        Args:
            run_id: The run ID to aggregate metrics for.

        Returns:
            Dictionary of aggregated metric names to values.

        Note:
            This method is kept for backward compatibility but is no longer used
            internally as aggregation is now handled by the BenchmarkReport class.
        """
        if run_id not in self.reports:
            return {}
        return self.reports[run_id].overall_scores

    # -----------------------
    # Aggregation & Filtering
    # -----------------------
    def aggregate_statistics(
        self,
        run_id: str,
        metrics: Optional[Sequence[str]] = None,
        percentiles: Optional[Sequence[float]] = None,
        tasks: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean/median and optional percentiles for metrics across tasks.

        Args:
            run_id: Report identifier
            metrics: Subset of metrics to aggregate; defaults to all
            percentiles: Optional list of percentiles [0-100]
            tasks: Optional subset of task_ids to include

        Returns:
            Mapping metric -> {"mean": x, "median": y, "p50": z, ...}
        """
        report = self.get_report(run_id)
        task_scores = report.task_scores
        if tasks is not None:
            task_scores = {t: s for t, s in task_scores.items() if t in set(tasks)}
        if not task_scores:
            return {}

        # Determine metrics to compute
        all_metrics = list(next(iter(task_scores.values())).keys())
        target_metrics = list(metrics) if metrics is not None else all_metrics

        results: Dict[str, Dict[str, float]] = {}
        for m in target_metrics:
            values = [scores[m] for scores in task_scores.values() if m in scores]
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            stats: Dict[str, float] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
            }
            if percentiles:
                for p in percentiles:
                    try:
                        stats[f"p{int(p)}"] = float(np.percentile(arr, p))
                    except Exception:  # pragma: no cover - guard against bad input
                        logger.warning("Invalid percentile value: %s", p)
            results[m] = stats
        return results

    def filter_and_sort_tasks(
        self,
        run_id: str,
        tasks: Optional[Sequence[str]] = None,
        metrics: Optional[Sequence[str]] = None,
        sort_by: Optional[str] = None,
        descending: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return task-level rows filtered to tasks/metrics and optionally sorted.

        Args:
            run_id: Report identifier
            tasks: Optional list of task_ids to include
            metrics: Optional list of metric names to include
            sort_by: Metric name to sort by (or "task_id")
            descending: Sort order for numeric sorts

        Returns:
            List of dict rows: {"task_id": str, <metric>: value, ...}
        """
        report = self.get_report(run_id)
        include_tasks = set(tasks) if tasks is not None else None
        include_metrics = set(metrics) if metrics is not None else None

        rows: List[Dict[str, Any]] = []
        for task_id, scores in report.task_scores.items():
            if include_tasks is not None and task_id not in include_tasks:
                continue
            row: Dict[str, Any] = {"task_id": task_id}
            if include_metrics is None:
                row.update(scores)
            else:
                row.update({k: v for k, v in scores.items() if k in include_metrics})
            rows.append(row)

        if sort_by:
            if sort_by == "task_id":
                rows.sort(key=lambda r: r.get("task_id", ""))
            else:
                rows.sort(
                    key=lambda r: r.get(sort_by, float("nan")), reverse=descending
                )
        return rows

    # -----------
    # Exporters
    # -----------
    def export_report_json(
        self, run_id: str, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Export a report to JSON (wrapper around save_report)."""
        report = self.get_report(run_id)
        return self.save_report(report, output_path)

    def export_report_csv(self, run_id: str, output_path: Union[str, Path]) -> Path:
        """Export a report to CSV with per-task rows and an overall row."""
        report = self.get_report(run_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine columns
        metric_names = list(sorted(report.overall_scores.keys()))
        fieldnames = ["row_type", "task_id"] + metric_names

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Per-task rows
            for task_id, scores in report.task_scores.items():
                row: Dict[str, Any] = {"row_type": "task", "task_id": task_id}
                row.update({m: scores.get(m, "") for m in metric_names})
                writer.writerow(row)

            # Overall row
            overall_row: Dict[str, Any] = {"row_type": "overall", "task_id": "OVERALL"}
            overall_row.update(
                {m: report.overall_scores.get(m, "") for m in metric_names}
            )
            writer.writerow(overall_row)

        return output_path

    def export_report_markdown(
        self, run_id: str, output_path: Union[str, Path]
    ) -> Path:
        """Export a report to a Markdown table."""
        report = self.get_report(run_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metric_names = list(sorted(report.overall_scores.keys()))
        headers = ["Task ID"] + metric_names
        sep = "|".join(["---"] * (len(headers)))

        lines: List[str] = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + sep + "|")

        # Task rows
        for task_id, scores in report.task_scores.items():
            row_vals = [task_id] + [f"{scores.get(m, '')}" for m in metric_names]
            lines.append("| " + " | ".join(row_vals) + " |")

        # Overall row
        overall_vals = ["OVERALL"] + [
            f"{report.overall_scores.get(m, '')}" for m in metric_names
        ]
        lines.append("| " + " | ".join(overall_vals) + " |")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def export_report_html(self, run_id: str, output_path: Union[str, Path]) -> Path:
        """Export a simple HTML table for the report (no external deps)."""
        report = self.get_report(run_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metric_names = list(sorted(report.overall_scores.keys()))

        # Build HTML
        def tr(cells: Iterable[str]) -> str:
            return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

        header = (
            "<tr>"
            + "".join(f"<th>{h}</th>" for h in (["Task ID"] + metric_names))
            + "</tr>"
        )
        task_rows = [
            tr([task_id] + [str(scores.get(m, "")) for m in metric_names])
            for task_id, scores in report.task_scores.items()
        ]
        overall_row = tr(
            ["OVERALL"] + [str(report.overall_scores.get(m, "")) for m in metric_names]
        )

        html = (
            "<!DOCTYPE html>\n"
            "<html><head><meta charset='utf-8'>"
            "<title>Benchmark Report</title>"
            "<style>"
            "table{border-collapse:collapse;}"
            "td,th{border:1px solid #ddd;padding:6px;}"
            "th{background:#f6f8fa;text-align:left}"
            "</style></head><body>"
            "<h1>Benchmark Report</h1>"
            "<table>" + header + "".join(task_rows) + overall_row + "</table>"
            "</body></html>"
        )
        output_path.write_text(html, encoding="utf-8")
        return output_path

    # -----------------
    # Run Comparisons
    # -----------------
    def compare_runs(
        self,
        run_a: str,
        run_b: str,
        metrics: Optional[Sequence[str]] = None,
        relative: bool = False,
        eps: float = 1e-9,
    ) -> Dict[str, Any]:
        """Compare two runs and compute diffs for overall and per-task metrics.

        Args:
            run_a: Baseline run id
            run_b: Comparison run id
            metrics: Optional subset of metrics to compare
            relative: If True, compute relative change (b - a) / (|a| + eps)
            eps: Small epsilon for division safety

        Returns:
            Dict with 'overall', 'per_task', and 'metadata'.
        """
        rep_a = self.get_report(run_a)
        rep_b = self.get_report(run_b)

        # Determine metrics
        all_metrics = sorted(
            set(rep_a.overall_scores.keys()) & set(rep_b.overall_scores.keys())
        )
        target_metrics = list(metrics) if metrics is not None else all_metrics

        def diff(a: float, b: float) -> float:
            if relative:
                return float((b - a) / (abs(a) + eps))
            return float(b - a)

        overall: Dict[str, float] = {}
        for m in target_metrics:
            if m in rep_a.overall_scores and m in rep_b.overall_scores:
                overall[m] = diff(rep_a.overall_scores[m], rep_b.overall_scores[m])

        # Per-task: only tasks present in both
        tasks_common = sorted(
            set(rep_a.task_scores.keys()) & set(rep_b.task_scores.keys())
        )
        per_task: Dict[str, Dict[str, float]] = {}
        for t in tasks_common:
            per_task[t] = {}
            for m in target_metrics:
                if m in rep_a.task_scores[t] and m in rep_b.task_scores[t]:
                    per_task[t][m] = diff(
                        rep_a.task_scores[t][m], rep_b.task_scores[t][m]
                    )

        return {
            "overall": overall,
            "per_task": per_task,
            "metadata": {"run_a": run_a, "run_b": run_b, "relative": relative},
        }

    # -----------------
    # Plotting hooks
    # -----------------
    def plot_metric_distribution(
        self, run_id: str, metric: str, output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Plot a metric across tasks. If matplotlib not available, return data only.

        Returns a dict with keys: metric, tasks, values, and optionally 'saved_to'.
        """
        report = self.get_report(run_id)
        tasks = []
        values = []
        for t, scores in report.task_scores.items():
            if metric in scores:
                tasks.append(t)
                values.append(scores[metric])

        data = {"metric": metric, "tasks": tasks, "values": values}

        if output_path is None:
            return data

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 4))
            plt.bar(tasks, values, color="#4c78a8")
            plt.title(f"{metric} by task")
            plt.xlabel("Task")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()
            data["saved_to"] = str(output_path)
            return data
        except Exception as e:  # pragma: no cover - optional dependency path
            logger.warning("Plotting unavailable or failed: %s", e)
            return data

    @classmethod
    def generate_run_id(
        cls,
        model_name: str,
        task_ids: List[str],
        timestamp: Optional[str] = None,
        max_length: int = 32,
    ) -> str:
        """Generate a deterministic run ID.

        Args:
            model_name: Name of the model being evaluated.
            task_ids: List of task IDs included in this run.
            timestamp: Optional timestamp to include in the ID.
            max_length: Maximum length of the generated ID.

        Returns:
            A deterministic run ID string.
        """
        import hashlib

        if timestamp is None:
            timestamp = datetime.now(UTC).isoformat()

        # Create a unique string from the inputs
        unique_str = f"{model_name}:{':'.join(sorted(task_ids))}:{timestamp}"

        # Generate a hash
        hash_obj = hashlib.sha256(unique_str.encode())
        hash_hex = hash_obj.hexdigest()

        # Truncate to max_length if needed
        if len(hash_hex) > max_length:
            return hash_hex[:max_length]
        return hash_hex.lower()
