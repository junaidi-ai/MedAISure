from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

from bench.models.benchmark_report import BenchmarkReport


class ReportGenerator(ABC):
    """Abstract base for report generators."""

    @abstractmethod
    def generate(
        self, benchmark_report: BenchmarkReport
    ) -> Any:  # pragma: no cover - interface
        """Generate report data from a BenchmarkReport."""
        raise NotImplementedError

    @abstractmethod
    def save(
        self, report: Any, output_path: Path
    ) -> None:  # pragma: no cover - interface
        """Persist a generated report to a file."""
        raise NotImplementedError
