from __future__ import annotations

from .base import ReportGenerator
from .json_report import JSONReportGenerator
from .markdown_report import MarkdownReportGenerator
from .html_report import HTMLReportGenerator
from .csv_report import CSVReportGenerator


class ReportFactory:
    @staticmethod
    def create_generator(fmt: str) -> ReportGenerator:
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
