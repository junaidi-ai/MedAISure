"""Result aggregation for MEDDSAI benchmark."""
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import json
import datetime
import hashlib
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Stores evaluation results for a single task."""
    task_id: str
    metrics: Dict[str, float]
    num_examples: int
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkReport:
    """Stores aggregated evaluation results across multiple tasks."""
    run_id: str
    timestamp: str
    model_name: str
    task_results: Dict[str, TaskResult]
    metrics_summary: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'model_name': self.model_name,
            'task_results': {
                task_id: asdict(result)
                for task_id, result in self.task_results.items()
            },
            'metrics_summary': self.metrics_summary,
            'metadata': self.metadata
        }
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save the report to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved benchmark report to {output_path}")


class ResultAggregator:
    """
    Aggregates and manages evaluation results across multiple tasks.
    
    This class handles the collection, aggregation, and persistence of evaluation
    results, including computing summary statistics across tasks.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize the ResultAggregator.
        
        Args:
            output_dir: Directory to save output reports. If None, uses a default location.
        """
        self.output_dir = Path(output_dir) if output_dir else Path('results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage of results
        self.results: Dict[str, Dict[str, TaskResult]] = {}
    
    def add_result(
        self,
        run_id: str,
        task_id: str,
        metrics: Dict[str, float],
        num_examples: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a task result to the aggregator.
        
        Args:
            run_id: Unique identifier for this evaluation run.
            task_id: Identifier for the evaluated task.
            metrics: Dictionary of metric names to values.
            num_examples: Number of examples evaluated.
            metadata: Additional metadata about the evaluation.
        """
        if run_id not in self.results:
            self.results[run_id] = {}
            
        self.results[run_id][task_id] = TaskResult(
            task_id=task_id,
            metrics=metrics,
            num_examples=num_examples,
            metadata=metadata or {}
        )
    
    def generate_report(
        self,
        run_id: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkReport:
        """Generate a benchmark report for the given run.
        
        Args:
            run_id: The run to generate a report for.
            model_name: Name of the model being evaluated.
            metadata: Additional metadata to include in the report.
            
        Returns:
            A BenchmarkReport object with aggregated results.
        """
        if run_id not in self.results or not self.results[run_id]:
            raise ValueError(f"No results found for run_id: {run_id}")
        
        # Aggregate metrics across tasks
        metrics_summary = self._aggregate_metrics(run_id)
        
        # Create the report
        report = BenchmarkReport(
            run_id=run_id,
            timestamp=datetime.datetime.utcnow().isoformat(),
            model_name=model_name,
            task_results=self.results[run_id].copy(),
            metrics_summary=metrics_summary,
            metadata=metadata or {}
        )
        
        return report
    
    def save_report(
        self,
        report: BenchmarkReport,
        output_path: Optional[Union[str, Path]] = None
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
            safe_model_name = "".join(c if c.isalnum() else "_" for c in report.model_name)
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{safe_model_name}_{timestamp}.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save(output_path)
        return output_path
    
    def _aggregate_metrics(self, run_id: str) -> Dict[str, float]:
        """Compute aggregate metrics across all tasks in a run."""
        if run_id not in self.results or not self.results[run_id]:
            return {}
        
        # Group metrics by name and collect their values
        metric_values = defaultdict(list)
        
        for task_result in self.results[run_id].values():
            for metric_name, value in task_result.metrics.items():
                metric_values[metric_name].append(value)
        
        # Compute mean for each metric
        metrics_summary = {
            f"mean_{metric}": sum(values) / len(values)
            for metric, values in metric_values.items()
        }
        
        # Add count of tasks
        metrics_summary["num_tasks"] = len(self.results[run_id])
        
        return metrics_summary
    
    @classmethod
    def generate_run_id(
        cls,
        model_name: str,
        task_ids: List[str],
        timestamp: Optional[str] = None,
        max_length: int = 32
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
        if timestamp is None:
            timestamp = datetime.datetime.utcnow().isoformat()
        
        # Create a unique string
        unique_str = f"{model_name}:{':'.join(sorted(task_ids))}:{timestamp}"
        
        # Generate a hash
        hash_obj = hashlib.md5(unique_str.encode('utf-8'))
        short_hash = hash_obj.hexdigest()[:8]
        
        # Create a readable prefix
        safe_model = "".join(c if c.isalnum() else "_" for c in model_name.lower())
        prefix = f"{safe_model[:12]}_{short_hash}"
        
        return prefix[-max_length:] if max_length > 0 else prefix
