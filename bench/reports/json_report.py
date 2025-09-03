from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from bench.models.benchmark_report import BenchmarkReport
from .base import ReportGenerator


class JSONReportGenerator(ReportGenerator):
    def generate(self, benchmark_report: BenchmarkReport) -> Dict:
        return benchmark_report.to_dict()

    def save(self, report: Dict, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str))

    def validate(self, report: Dict) -> None:
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
