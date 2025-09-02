from __future__ import annotations

from .base import ReportGenerator
from .json_report import JSONReportGenerator
from .markdown_report import MarkdownReportGenerator
from .html_report import HTMLReportGenerator


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
        raise ValueError(f"Unsupported format: {fmt}")
