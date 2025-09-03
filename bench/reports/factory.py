from __future__ import annotations

from .base import ReportGenerator
from .json_report import JSONReportGenerator
from .markdown_report import MarkdownReportGenerator
from .html_report import HTMLReportGenerator
from .csv_report import CSVReportGenerator


class ReportFactory:
    """Factory for creating report generator instances by format.

    Supported formats: "json", "md"/"markdown", "html", "csv".
    """

    @staticmethod
    def create_generator(fmt: str) -> ReportGenerator:
        """Create a report generator for the given format.

        Args:
            fmt: Desired format (e.g., "json", "md", "html", "csv").

        Returns:
            An instance of a concrete `ReportGenerator`.

        Raises:
            ValueError: If the format is not supported.
        """
        f = (fmt or "").lower()
        if f == "json":
            return JSONReportGenerator()
        if f in {"md", "markdown"}:
            return MarkdownReportGenerator()
        if f == "html":
            return HTMLReportGenerator()
        if f == "csv":
            return CSVReportGenerator()
        raise ValueError(f"Unsupported format: {fmt}")
