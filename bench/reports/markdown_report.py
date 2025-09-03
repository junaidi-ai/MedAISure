from __future__ import annotations

from pathlib import Path

from bench.models.benchmark_report import BenchmarkReport
from .base import ReportGenerator


class MarkdownReportGenerator(ReportGenerator):
    """Generate a Markdown report for a benchmark run."""

    def generate(self, benchmark_report: BenchmarkReport) -> str:
        """Render the provided benchmark report to Markdown text.

        Args:
            benchmark_report: Aggregated benchmark results to render.

        Returns:
            Markdown document as a string.
        """
        br = benchmark_report
        lines: list[str] = []
        lines.append(f"# Benchmark Report: {br.model_id}")
        lines.append(f"Generated at: {br.timestamp}\n")

        lines.append("## Overall Scores")
        for metric, score in (br.overall_scores or {}).items():
            lines.append(f"- **{metric}**: {float(score):.4f}")
        lines.append("")

        lines.append("## Task Scores")
        for task_id, scores in (br.task_scores or {}).items():
            lines.append(f"### {task_id}")
            for metric, score in (scores or {}).items():
                lines.append(f"- **{metric}**: {float(score):.4f}")
            lines.append("")

        lines.append("## Detailed Results")
        for result in br.detailed_results or []:
            lines.append(f"### Task: {result.task_id}")
            lines.append("#### Metrics")
            for metric, score in (result.metrics_results or {}).items():
                try:
                    sval = f"{float(score):.4f}"
                except Exception:
                    sval = str(score)
                lines.append(f"- **{metric}**: {sval}")
            lines.append("")

        return "\n".join(lines)

    def save(self, report: str, output_path: Path) -> None:
        """Write the Markdown report to disk.

        Args:
            report: Markdown text content.
            output_path: Destination file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    def validate(self, report: str) -> None:
        """Validate that the Markdown payload contains core sections.

        Args:
            report: Markdown text to validate.

        Raises:
            ValueError: If required markers are missing.
        """
        if not isinstance(report, str):
            raise ValueError("Markdown report must be a string")
        # Minimal structure checks
        required_markers = [
            "# Benchmark Report:",
            "## Overall Scores",
            "## Task Scores",
            "## Detailed Results",
        ]
        missing = [m for m in required_markers if m not in report]
        if missing:
            raise ValueError(f"Markdown report is missing sections: {missing}")
