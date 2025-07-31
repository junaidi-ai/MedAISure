"""Result aggregation for MEDDSAI benchmark."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..models.benchmark_report import BenchmarkReport
from ..models.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates and manages evaluation results across multiple tasks.

    This class handles the collection, aggregation, and persistence of evaluation
    results, including computing summary statistics across tasks.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize the ResultAggregator.

        Args:
            output_dir: Directory to save output reports. If None, uses a default
                location.
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage of results
        self.reports: Dict[str, BenchmarkReport] = {}

    def add_evaluation_result(
        self,
        evaluation_result: EvaluationResult,
        run_id: Optional[str] = None,
    ) -> None:
        """Add an evaluation result to the aggregator.

        Args:
            evaluation_result: The evaluation result to add
            run_id: Optional run ID to associate with this result. If not provided,
                   a new run will be created.
        """
        if run_id is None:
            # Generate a deterministic run ID if not provided
            run_id = self.generate_run_id(
                model_name=evaluation_result.metadata.get("model_name", "unknown"),
                task_ids=[evaluation_result.task_id],
                timestamp=evaluation_result.timestamp.isoformat(),
            )

        if run_id not in self.reports:
            # Create a new report for this run
            self.reports[run_id] = BenchmarkReport(
                model_id=evaluation_result.model_id,
                timestamp=evaluation_result.timestamp,
                overall_scores={},
                task_scores={},
                detailed_results=[],
                metadata={
                    "run_id": run_id,
                    "start_time": evaluation_result.timestamp.isoformat(),
                },
            )

        # Add the evaluation result to the report
        self.reports[run_id].add_evaluation_result(evaluation_result)

    def get_report(self, run_id: str) -> BenchmarkReport:
        """Get a benchmark report for the given run ID.

        Args:
            run_id: The run ID to get the report for.

        Returns:
            The BenchmarkReport for the specified run.

        Raises:
            ValueError: If no report exists for the given run_id.
        """
        if run_id not in self.reports:
            raise ValueError(f"No report found for run {run_id}")
        return self.reports[run_id]

    def save_report(
        self, report: BenchmarkReport, output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Save a benchmark report to disk.

        Args:
            report: The report to save.
            output_path: Path to save the report. If None, generates a filename.

        Returns:
            Path to the saved report.
        """
        if output_path is None:
            # Generate a filename based on model name and timestamp
            safe_model_name = "".join(
                c if c.isalnum() else "_" for c in report.model_name
            )
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{safe_model_name}_{timestamp}.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save(output_path)
        return output_path

    def _aggregate_metrics(self, run_id: str) -> Dict[str, float]:
        """Compute aggregate metrics across all tasks in a run.

        Args:
            run_id: The run ID to aggregate metrics for.

        Returns:
            Dictionary of aggregated metric names to values.

        Note:
            This method is kept for backward compatibility but is no longer used
            internally as aggregation is now handled by the BenchmarkReport class.
        """
        if run_id not in self.reports:
            return {}
        return self.reports[run_id].overall_scores

    @classmethod
    def generate_run_id(
        cls,
        model_name: str,
        task_ids: List[str],
        timestamp: Optional[str] = None,
        max_length: int = 32,
    ) -> str:
        """Generate a deterministic run ID.

        Args:
            model_name: Name of the model being evaluated.
            task_ids: List of task IDs included in this run.
            timestamp: Optional timestamp to include in the ID.
            max_length: Maximum length of the generated ID.

        Returns:
            A deterministic run ID string.
        """
        import hashlib

        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        # Create a unique string from the inputs
        unique_str = f"{model_name}:{':'.join(sorted(task_ids))}:{timestamp}"

        # Generate a hash
        hash_obj = hashlib.sha256(unique_str.encode())
        hash_hex = hash_obj.hexdigest()

        # Truncate to max_length if needed
        if len(hash_hex) > max_length:
            return hash_hex[:max_length]
        return hash_hex.lower()
