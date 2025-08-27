"""Data models for MedAISure benchmark.

Exports:
- TaskType
- MedicalTask
- EvaluationResult
- BenchmarkReport
"""

from .benchmark_report import BenchmarkReport
from .evaluation_result import EvaluationResult
from .medical_task import MedicalTask, TaskType

__all__ = [
    "TaskType",
    "MedicalTask",
    "EvaluationResult",
    "BenchmarkReport",
]
