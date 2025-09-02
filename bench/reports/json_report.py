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
