"""Evaluation harness for running benchmarks on medical AI models."""

import json
import os
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import yaml
from tqdm import tqdm

from ..models.benchmark_report import BenchmarkReport
from ..models.evaluation_result import EvaluationResult
from ..models.medical_task import MedicalTask
from .metric_calculator import MetricCalculator
from .model_runner import ModelRunner
from .result_aggregator import ResultAggregator
from .task_loader import TaskLoader
from .validators import ensure_task_schemas, validate_record_against_schema

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EvaluationHarness:
    """Main class for running evaluations on medical AI models."""

    def __init__(
        self,
        tasks_dir: str = "tasks",
        results_dir: str = "results",
        cache_dir: Optional[str] = None,
        log_level: str = "INFO",
        # Optional event callbacks
        on_task_start: Optional[Callable[[str], None]] = None,
        on_task_end: Optional[Callable[[str, Any], None]] = None,
        on_progress: Optional[Callable[[int, int, Optional[str]], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        on_metrics: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the evaluation harness.

        Args:
            tasks_dir: Directory containing task definitions
            results_dir: Directory to save evaluation results
            cache_dir: Directory for caching model outputs (optional)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Initialize components with type annotations
        self.task_loader: TaskLoader = TaskLoader(tasks_dir)
        self.model_runner: ModelRunner[Any, Any, Any] = ModelRunner[Any, Any, Any]()
        self.metric_calculator: MetricCalculator = MetricCalculator()
        self.result_aggregator: ResultAggregator = ResultAggregator(results_dir)
        self.tasks_dir: Path = Path(tasks_dir)
        self.results_dir: Path = Path(results_dir)
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None

        # Set up directories
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Event callbacks
        self._on_task_start = on_task_start
        self._on_task_end = on_task_end
        self._on_progress = on_progress
        self._on_error = on_error
        self._on_metrics = on_metrics

        # Internal: track the active model id for cleanup
        self._active_model_id: Optional[str] = None
        # Internal: validation behavior flag set per evaluate() call
        self._strict_validation: bool = False

        logger.info(
            "Initialized EvaluationHarness with "
            f"tasks_dir={tasks_dir}, results_dir={results_dir}"
        )

    # ---------------
    # Context manager
    # ---------------
    def __enter__(self) -> "EvaluationHarness":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close()
        except Exception:
            # Best-effort cleanup
            pass

    def close(self) -> None:
        """Cleanup resources, e.g., unload the active model if loaded."""
        if self._active_model_id is not None:
            try:
                self.model_runner.unload_model(self._active_model_id)
            finally:
                self._active_model_id = None

    def evaluate(
        self,
        model_id: str,
        task_ids: List[str],
        model_type: str = "huggingface",
        batch_size: int = 8,
        use_cache: bool = True,
        save_results: bool = True,
        strict_validation: bool = False,
        **model_kwargs: Any,
    ) -> BenchmarkReport:
        """Run evaluation on the specified model and tasks.

        Args:
            model_id: Identifier for the model to evaluate
            task_ids: List of task IDs to evaluate on
            model_type: Type of model ('huggingface', 'local', 'api')
            batch_size: Batch size for model inference
            use_cache: Whether to use cached results if available
            save_results: Whether to save results to disk
            **model_kwargs: Additional arguments for model loading

        Returns:
            BenchmarkReport containing evaluation results
        """
        # Generate a unique run ID
        run_id = self._generate_run_id(model_id, task_ids)
        logger.info(f"Starting evaluation run {run_id}")

        # Initialize the model
        logger.info(f"Initializing model: {model_id} (type: {model_type})")
        # Load the model (modifies the model_runner's internal state)
        self.model_runner.load_model(
            model_name=model_id,  # Use model_name parameter
            model_type=model_type,
            **model_kwargs,
        )
        self._active_model_id = model_id
        # Get the loaded model from the runner's _models dictionary
        model = self.model_runner._models[model_id]

        # Set validation mode for this run
        self._strict_validation = bool(strict_validation)

        # Initialize results
        task_results = {}
        start_time = time.time()

        # Evaluate on each task
        disable_tqdm = os.environ.get("MEDAISURE_NO_RICH") == "1"
        for idx, task_id in enumerate(
            tqdm(task_ids, desc="Evaluating tasks", disable=disable_tqdm), start=1
        ):
            task: Optional[MedicalTask] = None
            try:
                if self._on_progress:
                    self._on_progress(idx, len(task_ids), task_id)
                if self._on_task_start:
                    self._on_task_start(task_id)

                logger.info(f"Evaluating on task: {task_id}")

                # Load task
                task = self.task_loader.load_task(task_id)

                # Check cache
                cache_key = self._get_cache_key(run_id, task_id)
                cached_result = self._load_from_cache(cache_key) if use_cache else None

                if cached_result:
                    logger.info(f"Using cached results for task {task_id}")
                    task_result = cached_result
                else:
                    # Run evaluation
                    task_result = self._evaluate_task(
                        model=model,
                        model_id=model_id,
                        task=task,
                        batch_size=batch_size,
                    )

                    # Cache results
                    if self.cache_dir:
                        self._save_to_cache(cache_key, task_result)

                # Store results
                task_results[task_id] = task_result

                # Call metrics hook
                metrics_payload: Dict[str, Any] = {}
                if hasattr(task_result, "metrics_results") and isinstance(
                    task_result.metrics_results, dict
                ):
                    metrics_payload = task_result.metrics_results
                elif (
                    hasattr(task_result, "detailed_results")
                    and task_result.detailed_results
                ):
                    metrics_payload = task_result.detailed_results[0].metrics_results
                if self._on_metrics and metrics_payload:
                    self._on_metrics(task_id, metrics_payload)

                # Log progress
                if metrics_payload:
                    logger.info(
                        f"Completed task {task_id} - "
                        f"Metrics: {json.dumps(metrics_payload, indent=2)}"
                    )
                else:
                    logger.warning(f"No detailed results available for task {task_id}")

                if self._on_task_end:
                    self._on_task_end(task_id, task_result)

            except Exception as e:
                # In strict validation mode, propagate exceptions to fail fast
                if self._strict_validation:
                    raise
                if self._on_error:
                    try:
                        self._on_error(task_id, e)
                    except Exception:
                        # Avoid callback exceptions breaking flow
                        pass
                logger.error(
                    f"Error evaluating task {task_id}: {str(e)}", exc_info=True
                )
                # Insert a placeholder result so the task key appears in the final report
                try:
                    # Attempt to use the task's declared metrics for placeholder keys
                    metrics_dict: Dict[str, float] = {}
                    if task is not None and getattr(task, "metrics", None):
                        names: List[str] = []
                        for m in task.metrics:  # type: ignore[union-attr]
                            if isinstance(m, dict) and "name" in m:
                                names.append(str(m["name"]))
                            elif isinstance(m, str):
                                names.append(m)
                        metrics_dict = {name: 0.0 for name in names}
                    else:
                        # Fallback if task isn't available
                        metrics_dict = {"score": 0.0}

                    placeholder = EvaluationResult(
                        model_id=model_id,
                        task_id=task_id,
                        inputs=[],
                        model_outputs=[],
                        metrics_results=metrics_dict,
                        metadata={
                            "error": str(e),
                            "note": "Placeholder result due to evaluation failure",
                        },
                    )
                    task_results[task_id] = placeholder
                    # Intentionally do NOT call on_task_end for error placeholders to
                    # maintain semantics expected by tests (only successful tasks end).
                except Exception:
                    # As a last resort, skip adding results for this task
                    pass
                # Continue with other tasks on error
                continue

        # Calculate total time
        total_time = time.time() - start_time

        # Build report via ResultAggregator/BenchmarkReport to get consistent aggregation
        # Use the run_id for grouping
        for task_id, res in task_results.items():
            self.result_aggregator.add_evaluation_result(res, run_id=run_id)

        # Try to get report from aggregator; if mocked or unavailable, fall back to manual construction
        try:
            report = self.result_aggregator.get_report(run_id)
        except Exception:
            report = None

        if not isinstance(report, BenchmarkReport):
            # Fallback: construct a minimal BenchmarkReport from task_results
            task_scores: Dict[str, Dict[str, float]] = {}
            detailed_results: List[EvaluationResult] = []

            for t_id, res in task_results.items():
                # Metrics may be in metrics_results or detailed_results[0].metrics_results
                if hasattr(res, "metrics_results") and isinstance(
                    res.metrics_results, dict
                ):
                    metrics = res.metrics_results
                elif hasattr(res, "detailed_results") and res.detailed_results:
                    metrics = res.detailed_results[0].metrics_results
                else:
                    metrics = {}

                # Compute a simple average if numeric metrics exist; else 0.0
                avg = 0.0
                numeric_vals = [
                    float(v) for v in metrics.values() if isinstance(v, (int, float))
                ]
                if numeric_vals:
                    avg = sum(numeric_vals) / len(numeric_vals)
                task_scores[t_id] = {"average_score": avg}

                # Collect detailed results only if present (to preserve legacy placeholder behavior)
                if hasattr(res, "detailed_results") and res.detailed_results:
                    detailed_results.extend(res.detailed_results)

            overall_avg = 0.0
            if task_scores:
                overall_avg = sum(
                    s["average_score"] for s in task_scores.values()
                ) / len(task_scores)

            # If no detailed results collected, add legacy placeholder result
            if not detailed_results:
                detailed_results = [
                    EvaluationResult(
                        model_id=model_id,
                        task_id="unknown",
                        inputs=[],
                        model_outputs=[],
                        metrics_results={"score": 0.0},
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        metadata={
                            "note": "Placeholder result - no tasks completed successfully"
                        },
                    )
                ]

            report = BenchmarkReport(
                model_id=model_id,
                overall_scores={"average_score": overall_avg},
                task_scores=task_scores,
                detailed_results=detailed_results,
                metadata={},
            )
        # Enrich metadata
        report.metadata.update(
            {
                "run_id": run_id,
                "model_name": model_id,
                "total_time_seconds": total_time,
                "num_tasks": len(task_results),
                "batch_size": batch_size,
                "model_type": model_type,
                **model_kwargs,
            }
        )

        # If any EvaluationResult captured validation issues, surface them at report level
        try:
            all_val_errors: list[str] = []
            for res in report.detailed_results or []:
                errs = []
                meta = getattr(res, "metadata", {}) or {}
                if isinstance(meta, dict):
                    errs = meta.get("validation_errors", []) or []
                if errs:
                    # Flatten to strings
                    for e in errs:
                        all_val_errors.append(str(e))
            if all_val_errors:
                # De-duplicate while preserving order
                seen = set()
                deduped: list[str] = []
                for e in all_val_errors:
                    if e not in seen:
                        deduped.append(e)
                        seen.add(e)
                report.metadata["validation_errors"] = deduped
        except Exception:
            # Best-effort enrichment; never block evaluation on metadata issues
            pass

        # Save results
        try:
            if save_results:
                output_file = self.results_dir / f"{run_id}.json"
                report.save(output_file)
                logger.info(f"Saved evaluation results to {output_file}")
            return report
        finally:
            # Always attempt cleanup
            self.close()

    def _evaluate_task(  # noqa: C901
        self,
        model: Any,
        model_id: str,
        task: MedicalTask,
        batch_size: int = 8,
    ) -> EvaluationResult:
        """Evaluate a single task.

        Args:
            model: The model to evaluate
            model_id: ID of the model
            task: The task to evaluate on
            batch_size: Batch size for evaluation

        Returns:
            EvaluationResult containing evaluation results
        """

        # Prepare inputs
        inputs = task.dataset
        if not inputs:
            raise ValueError(f"No dataset found for task {task.task_id}")

        # Validate inputs against schema (non-strict by default)
        validation_errors: List[str] = []
        try:
            in_schema, out_schema = ensure_task_schemas(
                task.task_type, task.input_schema, task.output_schema
            )
            for idx, rec in enumerate(inputs):
                # Support nested dataset rows with 'input'
                payload = rec.get("input") if isinstance(rec, dict) else None
                payload = payload if isinstance(payload, dict) else rec
                validate_record_against_schema(
                    payload, in_schema, label="inputs", index=idx
                )
        except Exception as e:
            if self._strict_validation:
                raise
            validation_errors.append(str(e))

        # Run model inference
        logger.info(f"Running inference on {len(inputs)} examples...")
        # Extract input payloads for the model (support nested dataset rows)
        model_inputs: List[Dict[str, Any]] = []
        reference_outputs: List[Dict[str, Any]] = []
        for rec in inputs:
            if isinstance(rec, dict):
                in_payload = (
                    rec.get("input") if isinstance(rec.get("input"), dict) else rec
                )
                out_payload = (
                    rec.get("output") if isinstance(rec.get("output"), dict) else {}
                )
                if isinstance(in_payload, dict):
                    model_inputs.append(in_payload)
                else:
                    model_inputs.append(rec)  # fallback
                if isinstance(out_payload, dict):
                    reference_outputs.append(out_payload)
                else:
                    reference_outputs.append({})
            else:
                model_inputs.append({"text": str(rec)})
                reference_outputs.append({})

        predictions = self.model_runner.run_model(
            model_id=model_id, inputs=model_inputs, batch_size=batch_size
        )

        # Validate outputs against schema (non-strict by default)
        try:
            # predictions are list[dict]; support models returning nested structures
            for idx, rec in enumerate(predictions):
                payload = rec.get("output") if isinstance(rec, dict) else None
                payload = payload if isinstance(payload, dict) else rec
                validate_record_against_schema(
                    payload, out_schema, label="model_outputs", index=idx
                )
        except Exception as e:
            if self._strict_validation:
                raise
            validation_errors.append(str(e))

        # Calculate metrics
        logger.info("Calculating metrics...")
        # Extract metric names, handling both dict and str cases
        metric_names: List[str] = []
        if hasattr(task, "metrics") and task.metrics:
            for m in task.metrics:
                if isinstance(m, dict) and "name" in m:
                    metric_names.append(str(m["name"]))
                elif isinstance(m, str):
                    metric_names.append(m)

        metric_results = self.metric_calculator.calculate_metrics(
            task_id=task.task_id,
            predictions=predictions,
            references=reference_outputs if any(reference_outputs) else inputs,
            metric_names=metric_names,
        )

        # Extract metric values
        metrics = {name: result.value for name, result in metric_results.items()}

        # Create evaluation result
        result = EvaluationResult(
            model_id=model_id,
            task_id=task.task_id,
            inputs=inputs,
            model_outputs=predictions,
            metrics_results=metrics,
            metadata={
                "timestamp": time.time(),
                "batch_size": batch_size,
                "metrics_metadata": {
                    name: asdict(result) for name, result in metric_results.items()
                },
                "predictions": predictions[:10],  # First few predictions
            },
        )
        if validation_errors:
            # Attach collected validation issues for user inspection
            result.metadata["validation_errors"] = validation_errors
            logger.warning("Validation issues encountered: %s", validation_errors)
        return result

    def _generate_run_id(self, model_id: str, task_ids: List[str]) -> str:
        """Generate a unique run ID based on model and task IDs.

        Args:
            model_id: ID of the model being evaluated
            task_ids: List of task IDs being evaluated

        Returns:
            A unique 8-character run ID
        """
        import hashlib

        # Create a unique string from model_id and task_ids
        unique_str = f"{model_id}_{'_'.join(sorted(task_ids))}"

        # Create a hash of the unique string
        return hashlib.md5(unique_str.encode()).hexdigest()[:8]

    def _get_cache_key(self, run_id: str, task_id: str) -> Optional[str]:
        """Generate a cache key for storing/retrieving results.

        Args:
            run_id: ID of the current run
            task_id: ID of the task

        Returns:
            Cache key as a string, or None if cache is disabled
        """
        if not self.cache_dir:
            return None
        return str(self.cache_dir / f"{run_id}_{task_id}.json")

    def _load_from_cache(self, cache_key: Optional[str]) -> Optional[Any]:
        """Load results from cache if available.

        Args:
            cache_key: Path to the cache file, or None if caching is disabled

        Returns:
            Cached EvaluationResult if available and valid, None otherwise
        """
        if not cache_key or not Path(cache_key).exists():
            return None

        try:
            with open(cache_key, "r") as f:
                data = json.load(f)

            # Convert the loaded data into an EvaluationResult
            from ..models.evaluation_result import EvaluationResult

            return EvaluationResult(**data)

        except Exception as e:
            logger.warning(f"Error loading from cache {cache_key}: {str(e)}")
            return None

    def _save_to_cache(self, cache_key: Optional[str], data: Any) -> None:
        """Save results to cache.

        Args:
            cache_key: Path to the cache file, or None if caching is disabled
            data: Data to cache (must be an EvaluationResult or serializable dict)
        """
        if not cache_key:
            return

        try:
            # Create parent directories if they don't exist
            cache_path = Path(cache_key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert data to dict if it's an EvaluationResult
            if hasattr(data, "model_dump"):
                data_dict = data.model_dump()
            elif hasattr(data, "dict"):
                data_dict = data.dict()  # Fallback for older Pydantic versions
            else:
                data_dict = data

            with open(cache_path, "w") as f:
                json.dump(data_dict, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Error saving to cache {cache_key}: {str(e)}")

    def list_available_tasks(self) -> List[Dict[str, Any]]:
        """List all available tasks in the tasks directory."""
        tasks = []

        # Look for YAML task definitions
        for task_file in self.tasks_dir.glob("*.yaml"):
            try:
                with open(task_file, "r") as f:
                    task_data = yaml.safe_load(f)
                    tasks.append(
                        {
                            "task_id": task_file.stem,
                            "name": task_data.get("name", ""),
                            "description": task_data.get("description", ""),
                            "metrics": [
                                m["name"] for m in task_data.get("metrics", [])
                            ],
                            "num_examples": len(task_data.get("dataset", [])),
                            "file": str(task_file),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error loading task from {task_file}: {str(e)}")

        # Look for JSON task definitions
        for task_file in self.tasks_dir.glob("*.json"):
            if task_file.stem == "tasks":  # Skip the tasks index file
                continue

            try:
                with open(task_file, "r") as f:
                    task_data = json.load(f)
                    tasks.append(
                        {
                            "task_id": task_file.stem,
                            "name": task_data.get("name", ""),
                            "description": task_data.get("description", ""),
                            "metrics": [
                                m["name"] for m in task_data.get("metrics", [])
                            ],
                            "num_examples": len(task_data.get("dataset", [])),
                            "file": str(task_file),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error loading task from {task_file}: {str(e)}")

        return tasks

    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific task."""
        task = self.task_loader.load_task(task_id)

        return {
            "task_id": task.task_id,
            "name": task.name,
            "description": task.description,
            "input_schema": task.input_schema,
            "output_schema": task.output_schema,
            "metrics": task.metrics,
            "num_examples": len(task.dataset) if task.dataset else 0,
            "example_input": task.dataset[0] if task.dataset else None,
        }
