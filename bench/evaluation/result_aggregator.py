"""Result aggregation for MedAISure benchmark."""

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..models.benchmark_report import BenchmarkReport
from ..models.evaluation_result import EvaluationResult
from ..leaderboard.submission import (
    build_submission_from_report,
    build_and_validate_submission,
    validate_submission,
)

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates and manages evaluation results across multiple tasks.

    This class handles the collection, aggregation, and persistence of evaluation
    results, including computing summary statistics across tasks.

    Example – minimal flow:
        >>> from bench.evaluation.result_aggregator import ResultAggregator
        >>> from bench.models import EvaluationResult
        >>> ra = ResultAggregator(output_dir="results")
        >>> er = EvaluationResult(
        ...     model_id="demo",
        ...     task_id="task1",
        ...     inputs=[{"text": "x"}],
        ...     model_outputs=[{"label": "y"}],
        ...     metrics_results={"accuracy": 1.0},
        ... )
        >>> ra.add_evaluation_result(er, run_id="run-1")
        >>> report = ra.get_report("run-1")
        >>> isinstance(report.overall_scores, dict)
        True
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

        Example:
            >>> ra = ResultAggregator()
            >>> er = EvaluationResult(
            ...     model_id="m",
            ...     task_id="t",
            ...     inputs=[{"q": 1}],
            ...     model_outputs=[{"label": 1}],
            ...     metrics_results={"accuracy": 1.0},
            ... )
            >>> ra.add_evaluation_result(er, run_id="run-A")
            >>> list(ra.reports.keys())
            ['run-A']
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

        Example:
            >>> ra = ResultAggregator()
            >>> er = EvaluationResult(
            ...     model_id="m",
            ...     task_id="t",
            ...     inputs=[{"q": 1}],
            ...     model_outputs=[{"label": 1}],
            ...     metrics_results={"accuracy": 1.0},
            ... )
            >>> ra.add_evaluation_result(er, run_id="run-A")
            >>> rep = ra.get_report("run-A")
            >>> rep.metadata.get("run_id")
            'run-A'
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

        Example:
            >>> ra = ResultAggregator(output_dir="results")
            >>> er = EvaluationResult(
            ...     model_id="m",
            ...     task_id="t",
            ...     inputs=[{"q": 1}],
            ...     model_outputs=[{"label": 1}],
            ...     metrics_results={"accuracy": 1.0},
            ... )
            >>> ra.add_evaluation_result(er, run_id="run-A")
            >>> p = ra.save_report(ra.get_report("run-A"))  # doctest: +ELLIPSIS
            >>> str(p).endswith('.json')
            True
        """
        if output_path is None:
            # Generate a filename based on model id and timestamp
            safe_model_id = "".join(c if c.isalnum() else "_" for c in report.model_id)
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{safe_model_id}_{timestamp}.json"
        else:
            output_path = Path(output_path)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report.save(output_path)
            return output_path
        except Exception as e:
            msg = f"Failed to save report to {output_path}: {e}"
            logger.error(msg)
            raise ValueError(msg) from e

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

        Example:
            >>> ra = ResultAggregator()
            >>> er1 = EvaluationResult(model_id="m", task_id="t1", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 0.8})
            >>> er2 = EvaluationResult(model_id="m", task_id="t2", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 0.6})
            >>> ra.add_evaluation_result(er1, run_id="r")
            >>> ra.add_evaluation_result(er2, run_id="r")
            >>> stats = ra.aggregate_statistics("r", metrics=["acc"])  # doctest: +ELLIPSIS
            >>> round(stats["acc"]["mean"], 3)
            0.7
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

        Example:
            >>> ra = ResultAggregator()
            >>> er1 = EvaluationResult(model_id="m", task_id="t1", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 0.8})
            >>> er2 = EvaluationResult(model_id="m", task_id="t2", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 0.6})
            >>> ra.add_evaluation_result(er1, run_id="r")
            >>> ra.add_evaluation_result(er2, run_id="r")
            >>> rows = ra.filter_and_sort_tasks("r", sort_by="acc")
            >>> [r["task_id"] for r in rows]  # doctest: +ELLIPSIS
            ['t2', 't1']
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
    # -------------------------------
    # Category aggregation utilities
    # -------------------------------
    # Default canonical mapping derived from docs/metrics/metric_categories.md
    _DEFAULT_CATEGORY_MAP: Dict[str, set] = {
        # Diagnostics: core correctness
        "diagnostics": {
            "clinical_accuracy",
            "diagnostic_accuracy",
            "clinical_correctness",
            "final_answer_correct",
            "exact_match",
            # Many tasks use a plain 'accuracy' for diagnostic QA
            "accuracy",
        },
        # Safety: harmfulness avoidance and guideline adherence
        "safety": {
            "safety",
            "harm_avoidance",
            "toxicity_reduction",
            "factuality_safety",
        },
        # Communication: clarity and usefulness
        "communication": {
            "reasoning_quality",
            "communication",
            "coherence",
            "helpfulness",
            "instruction_following",
        },
        # Summarization: quality of summarizing clinical content
        "summarization": {
            "summarization",
            "summary_quality",
            "rouge_l",
            "rouge1",
            "rouge2",
            "bertscore",
            # Heuristic proxies implemented in MetricCalculator
            "clinical_relevance",
            "factual_consistency",
        },
    }

    def add_category_aggregates(
        self,
        run_id: str,
        category_map: Optional[Dict[str, Sequence[str]]] = None,
        *,
        aggregate: str = "mean",
    ) -> None:
        """Compute per-task category aggregates and update overall category scores.

        This maps raw metric keys (e.g., 'accuracy', 'rouge_l') into canonical
        categories (e.g., 'diagnostics', 'summarization') and stores the reduced
        values under those category names in both `task_scores` and
        `overall_scores` of the underlying `BenchmarkReport`.

        Args:
            run_id: The run whose report to update.
            category_map: Optional custom mapping of category -> iterable of metric
                keys. Falls back to the default canonical mapping if None.
            aggregate: Reduction to apply over matched metrics per category.
                Supported: 'mean' (default). Unknown values default to mean.

        Notes:
            - Non-numeric metric values are ignored.
            - Categories with no matched numeric metrics for a task are skipped for
              that task.
            - Overall category scores are computed as the mean over tasks where the
              category aggregate is present.
        """
        if run_id not in self.reports:
            raise ValueError(f"No report found for run {run_id}")

        report = self.reports[run_id]
        cmap: Dict[str, set] = {}
        try:
            # Normalize mapping to category -> set of metric names (lowercased)
            raw_map = (
                category_map
                if category_map is not None
                else {k: sorted(v) for k, v in self._DEFAULT_CATEGORY_MAP.items()}
            )
            for cat, keys in (raw_map or {}).items():
                norm_cat = str(cat).strip()
                key_set = {str(k).strip() for k in (keys or [])}
                cmap[norm_cat] = {k.lower() for k in key_set if k}
        except Exception:
            # If mapping normalization fails, do nothing but log
            logger.warning(
                "Invalid category_map provided; skipping category aggregation"
            )
            return

        if not cmap:
            return

        # Compute per-task category aggregates
        per_task_cat: Dict[str, Dict[str, float]] = {}
        for task_id, metrics in (report.task_scores or {}).items():
            if not isinstance(metrics, dict) or not metrics:
                continue
            # Lowercase key view for matching but keep original dict for values
            lower_key_map = {k.lower(): k for k in metrics.keys()}
            cat_vals: Dict[str, float] = {}
            for cat, metric_names in cmap.items():
                matched_vals: List[float] = []
                for lname in metric_names:
                    orig_key = lower_key_map.get(lname)
                    if orig_key is None:
                        continue
                    try:
                        v = float(metrics[orig_key])
                    except Exception:
                        continue
                    matched_vals.append(v)
                if not matched_vals:
                    # No usable values for this category on this task
                    continue
                if aggregate.lower() != "mean":
                    # Future-proofing: default to mean for unsupported reducers
                    logger.debug("Unknown aggregate '%s'; using mean", aggregate)
                cat_vals[cat] = float(np.mean(np.asarray(matched_vals, dtype=float)))
            if cat_vals:
                # Persist per-task category aggregates directly into task_scores
                report.task_scores[task_id].update(cat_vals)
                per_task_cat[task_id] = cat_vals

        # Compute overall category means over tasks where present
        if per_task_cat:
            # For each category, collect values from tasks having that category
            cat_to_vals: Dict[str, List[float]] = {}
            for _task, cdict in per_task_cat.items():
                for cat, val in cdict.items():
                    cat_to_vals.setdefault(cat, []).append(float(val))
            for cat, vals in cat_to_vals.items():
                if not vals:
                    continue
                report.overall_scores[cat] = float(
                    np.mean(np.asarray(vals, dtype=float))
                )

    def export_report_json(
        self, run_id: str, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Export a report to JSON (wrapper around save_report).

        Example:
            >>> ra = ResultAggregator()
            >>> er = EvaluationResult(model_id="m", task_id="t", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 1.0})
            >>> ra.add_evaluation_result(er, run_id="r")
            >>> p = ra.export_report_json("r")  # doctest: +ELLIPSIS
            >>> p.suffix
            '.json'
        """
        report = self.get_report(run_id)
        return self.save_report(report, output_path)

    def export_leaderboard_submission(
        self,
        run_id: str,
        output_path: Union[str, Path],
        *,
        include_reasoning: bool = True,
        validate: bool = True,
    ) -> Path:
        """Export a leaderboard submission JSON for a given run.

        This builds a submission payload from the underlying BenchmarkReport's
        detailed results, optionally validates it against the lightweight
        submission schema, and writes it to ``output_path``.

        Example:
            >>> ra = ResultAggregator()
            >>> er = EvaluationResult(model_id="m", task_id="t", inputs=[{}], model_outputs=[{"answer": 1}], metrics_results={"acc": 1.0})
            >>> ra.add_evaluation_result(er, run_id="r")
            >>> p = ra.export_leaderboard_submission("r", "results/submission.json")  # doctest: +ELLIPSIS
            >>> p.suffix
            '.json'
        """
        report = self.get_report(run_id)
        # Build payload and validate if requested
        if validate:
            payload = build_and_validate_submission(
                report, include_reasoning=include_reasoning
            )
        else:
            payload = build_submission_from_report(
                report, include_reasoning=include_reasoning
            )
            try:
                # Best-effort validation to surface warnings in logs without raising
                validate_submission(payload)
            except Exception as e:
                logger.warning("Submission validation warning: %s", e)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            import json as _json

            out.write_text(
                _json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return out
        except Exception as e:
            msg = f"Failed to export submission to {out}: {e}"
            logger.error(msg)
            raise ValueError(msg) from e

    def export_report_csv(self, run_id: str, output_path: Union[str, Path]) -> Path:
        """Export a report to CSV with per-task rows and an overall row.

        Example:
            >>> ra = ResultAggregator()
            >>> er = EvaluationResult(model_id="m", task_id="t", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 1.0})
            >>> ra.add_evaluation_result(er, run_id="r")
            >>> _ = ra.export_report_csv("r", "results/demo.csv")  # doctest: +ELLIPSIS
        """
        report = self.get_report(run_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine columns
        metric_names = list(sorted(report.overall_scores.keys()))
        fieldnames = ["row_type", "task_id"] + metric_names

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # Per-task rows
                for task_id, scores in report.task_scores.items():
                    row: Dict[str, Any] = {"row_type": "task", "task_id": task_id}
                    row.update({m: scores.get(m, "") for m in metric_names})
                    writer.writerow(row)

                # Overall row
                overall_row: Dict[str, Any] = {
                    "row_type": "overall",
                    "task_id": "OVERALL",
                }
                overall_row.update(
                    {m: report.overall_scores.get(m, "") for m in metric_names}
                )
                writer.writerow(overall_row)

            return output_path
        except Exception as e:
            msg = f"Failed to export CSV to {output_path}: {e}"
            logger.error(msg)
            raise ValueError(msg) from e

    def export_report_markdown(
        self,
        run_id: str,
        output_path: Union[str, Path],
        *,
        include_examples: bool = False,
        max_examples: int = 5,
        include_validation_errors: bool = False,
    ) -> Path:
        """Export a report to a Markdown table.

        Example:
            >>> ra = ResultAggregator()
            >>> er = EvaluationResult(model_id="m", task_id="t", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 1.0})
            >>> ra.add_evaluation_result(er, run_id="r")
            >>> _ = ra.export_report_markdown("r", "results/demo.md")  # doctest: +ELLIPSIS
        """
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

        # Optionally include example-level section
        if include_examples:
            lines.append("")
            lines.append("## Examples (truncated)")
            # Group results by task for clarity
            by_task: Dict[str, List[EvaluationResult]] = {}
            for res in report.detailed_results or []:
                by_task.setdefault(res.task_id, []).append(res)
            for t_id, results in by_task.items():
                lines.append("")
                lines.append(f"### Task: {t_id}")
                # Flatten pairs from inputs/model_outputs
                pairs: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
                for res in results:
                    n = min(len(res.inputs or []), len(res.model_outputs or []))
                    for i in range(n):
                        pairs.append((res.inputs[i], res.model_outputs[i]))
                # Render up to max_examples
                for idx, (inp, out) in enumerate(pairs[: max(0, int(max_examples))]):
                    lines.append("")
                    lines.append(f"- Example {idx + 1}:")
                    lines.append(f"  - input: `{str(inp)}`")
                    lines.append(f"  - prediction: `{str(out)}`")

        # Optionally include validation errors aggregated at report metadata
        if include_validation_errors:
            errs = (report.metadata or {}).get("validation_errors", []) or []
            if errs:
                lines.append("")
                lines.append("## Validation Errors")
                for e in errs:
                    lines.append(f"- {e}")

        try:
            output_path.write_text("\n".join(lines), encoding="utf-8")
            return output_path
        except Exception as e:
            msg = f"Failed to export Markdown to {output_path}: {e}"
            logger.error(msg)
            raise ValueError(msg) from e

    def export_report_html(
        self,
        run_id: str,
        output_path: Union[str, Path],
        *,
        include_examples: bool = False,
        max_examples: int = 5,
        include_validation_errors: bool = False,
    ) -> Path:
        """Export a simple HTML table for the report (no external deps).

        Example:
            >>> ra = ResultAggregator()
            >>> er = EvaluationResult(model_id="m", task_id="t", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 1.0})
            >>> ra.add_evaluation_result(er, run_id="r")
            >>> _ = ra.export_report_html("r", "results/demo.html")  # doctest: +ELLIPSIS
        """
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

        extra_sections: List[str] = []

        if include_examples:
            sections: List[str] = ["<h2>Examples (truncated)</h2>"]
            # Group by task
            by_task: Dict[str, List[EvaluationResult]] = {}
            for res in report.detailed_results or []:
                by_task.setdefault(res.task_id, []).append(res)
            for t_id, results in by_task.items():
                sections.append(f"<h3>Task: {t_id}</h3>")
                rows: List[str] = []
                count = 0
                for res in results:
                    n = min(len(res.inputs or []), len(res.model_outputs or []))
                    for i in range(n):
                        if count >= max(0, int(max_examples)):
                            break
                        inp = res.inputs[i]
                        out = res.model_outputs[i]
                        rows.append(
                            "<tr>"
                            f"<td><pre>{str(inp)}</pre></td>"
                            f"<td><pre>{str(out)}</pre></td>"
                            "</tr>"
                        )
                        count += 1
                    if count >= max(0, int(max_examples)):
                        break
                if rows:
                    sections.append(
                        "<table><tr><th>input</th><th>prediction</th></tr>"
                        + "".join(rows)
                        + "</table>"
                    )
            extra_sections.append("".join(sections))

        if include_validation_errors:
            errs = (report.metadata or {}).get("validation_errors", []) or []
            if errs:
                lis = "".join(f"<li>{str(e)}</li>" for e in errs)
                extra_sections.append(f"<h2>Validation Errors</h2><ul>{lis}</ul>")

        html = (
            "<!DOCTYPE html>\n"
            "<html><head><meta charset='utf-8'>"
            "<title>Benchmark Report</title>"
            "<style>"
            "table{border-collapse:collapse;}"
            "td,th{border:1px solid #ddd;padding:6px;}"
            "th{background:#f6f8fa;text-align:left}"
            "pre{white-space:pre-wrap;margin:0;}"
            "</style></head><body>"
            "<h1>Benchmark Report</h1>"
            "<table>"
            + header
            + "".join(task_rows)
            + overall_row
            + "</table>"
            + "".join(extra_sections)
            + "</body></html>"
        )
        try:
            output_path.write_text(html, encoding="utf-8")
            return output_path
        except Exception as e:
            msg = f"Failed to export HTML to {output_path}: {e}"
            logger.error(msg)
            raise ValueError(msg) from e

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

        Example:
            >>> ra = ResultAggregator()
            >>> er1 = EvaluationResult(model_id="m", task_id="t", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 0.8})
            >>> er2 = EvaluationResult(model_id="m", task_id="t", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 0.9})
            >>> ra.add_evaluation_result(er1, run_id="A")
            >>> ra.add_evaluation_result(er2, run_id="B")
            >>> diff = ra.compare_runs("A", "B", metrics=["acc"])  # doctest: +ELLIPSIS
            >>> round(diff["overall"]["acc"], 3)
            0.1
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

        Example:
            >>> ra = ResultAggregator()
            >>> er1 = EvaluationResult(model_id="m", task_id="t1", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 0.8})
            >>> er2 = EvaluationResult(model_id="m", task_id="t2", inputs=[{}], model_outputs=[{}], metrics_results={"acc": 0.6})
            >>> ra.add_evaluation_result(er1, run_id="r")
            >>> ra.add_evaluation_result(er2, run_id="r")
            >>> data = ra.plot_metric_distribution("r", "acc")
            >>> set(data.keys()) >= {"metric", "tasks", "values"}
            True
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

        Example:
            >>> rid = ResultAggregator.generate_run_id("m", ["t1", "t2"], timestamp="2024-01-01T00:00:00Z")
            >>> isinstance(rid, str) and len(rid) > 0
            True
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
