from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bench.models.benchmark_report import BenchmarkReport
from bench.models.evaluation_result import EvaluationResult
from .base import ReportGenerator


class CSVReportGenerator(ReportGenerator):
    """
    Generate CSV representations of a BenchmarkReport.

    generate() returns a mapping of file name -> CSV text so callers can choose how to persist.
    save() writes either a single CSV file (when output_path ends with .csv) using task_scores,
    or, when output_path is a directory (recommended), writes multiple CSV files:
      - overall_scores.csv
      - task_scores.csv
      - detailed_metrics.csv
      - detailed_inputs.csv
      - detailed_outputs.csv
    """

    def generate(self, benchmark_report: BenchmarkReport) -> Dict[str, str]:
        br = benchmark_report
        outputs: Dict[str, str] = {}

        # Overall scores: metric,score
        outputs["overall_scores.csv"] = br.overall_scores_to_csv()

        # Task scores: task_id,metric,score
        outputs["task_scores.csv"] = br.task_scores_to_csv()

        # Detailed metrics flattened: model_id,task_id,timestamp,metric,score
        outputs["detailed_metrics.csv"] = self._detailed_metrics_to_csv(
            br.detailed_results or []
        )

        # Aggregate inputs across results: task_id,index,<dynamic input fields>
        inputs_rows, inputs_headers = self._aggregate_rows(
            br.detailed_results or [], field="inputs"
        )
        outputs["detailed_inputs.csv"] = self._rows_to_csv(inputs_headers, inputs_rows)

        # Aggregate outputs across results: task_id,index,<dynamic output fields>
        outs_rows, outs_headers = self._aggregate_rows(
            br.detailed_results or [], field="model_outputs"
        )
        outputs["detailed_outputs.csv"] = self._rows_to_csv(outs_headers, outs_rows)

        return outputs

    def save(self, report: Dict[str, str], output_path: Path) -> None:
        output_path = Path(output_path)
        if output_path.suffix.lower() == ".csv":
            # If a single CSV file was requested, write the primary table (task_scores)
            text = report.get("task_scores.csv", "")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text)
            return

        # Otherwise treat as directory and write all CSV files
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, text in (report or {}).items():
            (output_dir / name).write_text(text)

    def validate(self, report: Dict[str, str]) -> None:
        if not isinstance(report, dict):
            raise ValueError("CSV report must be a dict of filename -> CSV string")
        expected = {
            "overall_scores.csv": ["metric", "score"],
            "task_scores.csv": ["task_id", "metric", "score"],
            "detailed_metrics.csv": [
                "model_id",
                "task_id",
                "timestamp",
                "metric",
                "score",
            ],
            "detailed_inputs.csv": ["task_id", "index"],  # dynamic fields follow
            "detailed_outputs.csv": ["task_id", "index"],  # dynamic fields follow
        }
        missing = [k for k in expected.keys() if k not in report]
        if missing:
            raise ValueError(f"CSV report missing files: {missing}")

        # Validate headers for the fixed schemas; inputs/outputs allow dynamic extra columns
        for name, required_headers in expected.items():
            csv_text = report.get(name, "")
            if not isinstance(csv_text, str):
                raise ValueError(f"CSV content for {name} must be a string")
            # Read header line
            sio = StringIO(csv_text)
            reader = csv.reader(sio)
            try:
                headers = next(reader)
            except StopIteration:
                headers = []
            missing_headers = [h for h in required_headers if h not in headers]
            if missing_headers:
                raise ValueError(
                    f"{name} missing required headers: {missing_headers}; found: {headers}"
                )

    # --- Helpers ---
    @staticmethod
    def _detailed_metrics_to_csv(results: List[EvaluationResult]) -> str:
        headers = ["model_id", "task_id", "timestamp", "metric", "score"]
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
        for res in results:
            ts = res.timestamp.isoformat()
            for metric, score in (res.metrics_results or {}).items():
                writer.writerow(
                    {
                        "model_id": res.model_id,
                        "task_id": res.task_id,
                        "timestamp": ts,
                        "metric": metric,
                        "score": float(score),
                    }
                )
        return buf.getvalue()

    @staticmethod
    def _aggregate_rows(
        results: List[EvaluationResult], *, field: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Aggregate a list-of-dicts field across results into rows with unified headers.
        field: "inputs" | "model_outputs"
        Returns (rows, headers)
        """
        # Collect union of headers
        headers: List[str] = ["task_id", "index"]
        dynamic_headers: List[str] = []

        def _ensure_header(k: str) -> None:
            if k not in dynamic_headers:
                dynamic_headers.append(k)

        # First pass: compute dynamic headers
        for res in results:
            rows = getattr(res, field) or []
            for rec in rows:
                for k in rec.keys():
                    _ensure_header(k)

        full_headers = headers + dynamic_headers

        # Second pass: build rows
        rows_out: List[Dict[str, Any]] = []
        for res in results:
            rows = getattr(res, field) or []
            for i, rec in enumerate(rows):
                row: Dict[str, Any] = {"task_id": res.task_id, "index": i}
                for k in dynamic_headers:
                    v = rec.get(k)
                    # Normalize nested values to strings for CSV friendliness
                    if hasattr(v, "isoformat"):
                        try:
                            v = v.isoformat()
                        except Exception:
                            v = str(v)
                    elif isinstance(v, (dict, list, tuple, set)):
                        v = str(v)
                    row[k] = v
                rows_out.append(row)

        return rows_out, full_headers

    @staticmethod
    def _rows_to_csv(headers: List[str], rows: List[Dict[str, Any]]) -> str:
        if not rows:
            buf = StringIO()
            writer = csv.DictWriter(buf, fieldnames=headers)
            writer.writeheader()
            return buf.getvalue()
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h) for h in headers})
        return buf.getvalue()
