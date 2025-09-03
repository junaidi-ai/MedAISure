from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from bench.models.benchmark_report import BenchmarkReport
from .base import ReportGenerator


class JSONReportGenerator(ReportGenerator):
    """Generate JSON representation of a BenchmarkReport."""

    def generate(self, benchmark_report: BenchmarkReport) -> Dict:
        """Serialize the benchmark report to a JSON-compatible dict.

        Args:
            benchmark_report: Aggregated benchmark results.

        Returns:
            A JSON-serializable dictionary.
        """
        return benchmark_report.to_dict()

    def save(self, report: Dict, output_path: Path) -> None:
        """Write the JSON report to a file.

        Args:
            report: JSON-serializable dictionary.
            output_path: Destination file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str))

    def validate(self, report: Dict) -> None:
        """Validate basic structure and required keys of the JSON payload.

        Args:
            report: JSON dictionary to validate.

        Raises:
            ValueError: If structure or required keys are invalid.
        """
        if not isinstance(report, dict):
            raise ValueError("JSON report must be a dict")
        # Required top-level keys
        required = {
            "schema_version",
            "model_id",
            "timestamp",
            "overall_scores",
            "task_scores",
            "detailed_results",
            "metadata",
        }
        missing = required - set(report.keys())
        if missing:
            raise ValueError(f"JSON report missing keys: {sorted(missing)}")
        if not isinstance(report["model_id"], str) or not report["model_id"].strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(report["overall_scores"], dict):
            raise ValueError("overall_scores must be a dict")
        if not isinstance(report["task_scores"], dict):
            raise ValueError("task_scores must be a dict")
        if not isinstance(report["detailed_results"], list):
            raise ValueError("detailed_results must be a list")
