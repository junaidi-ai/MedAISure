"""Evaluation harness for running benchmarks on medical AI models."""
from typing import Dict, List, Optional, Any, Union
import logging
import json
import yaml
from pathlib import Path
from dataclasses import asdict
import time
from tqdm import tqdm
import numpy as np

from .task_loader import TaskLoader, MedicalTask
from .model_runner import ModelRunner
from .metric_calculator import MetricCalculator, MetricResult
from .result_aggregator import ResultAggregator, BenchmarkReport

logger = logging.getLogger(__name__)

class EvaluationHarness:
    """Main class for running evaluations on medical AI models."""
    
    def __init__(
        self,
        tasks_dir: str = "tasks",
        results_dir: str = "results",
        cache_dir: Optional[str] = None,
        log_level: str = "INFO"
    ):
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
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self.task_loader = TaskLoader(tasks_dir)
        self.model_runner = ModelRunner()
        self.metric_calculator = MetricCalculator()
        self.result_aggregator = ResultAggregator(results_dir)
        
        # Set up directories
        self.tasks_dir = Path(tasks_dir)
        self.results_dir = Path(results_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized EvaluationHarness with tasks_dir={tasks_dir}, results_dir={results_dir}")
    
    def evaluate(
        self,
        model_id: str,
        task_ids: List[str],
        model_type: str = "huggingface",
        batch_size: int = 8,
        use_cache: bool = True,
        save_results: bool = True,
        **model_kwargs
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
        
        # Load the model
        logger.info(f"Loading model: {model_id} (type: {model_type})")
        model = self.model_runner.load_model(model_id, model_type=model_type, **model_kwargs)
        
        # Initialize results
        task_results = {}
        start_time = time.time()
        
        # Evaluate on each task
        for task_id in tqdm(task_ids, desc="Evaluating tasks"):
            try:
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
                        batch_size=batch_size
                    )
                    
                    # Cache results
                    if self.cache_dir:
                        self._save_to_cache(cache_key, task_result)
                
                # Store results
                task_results[task_id] = task_result
                
                # Log progress
                logger.info(f"Completed task {task_id} - Metrics: {json.dumps(task_result.metrics, indent=2)}")
                
            except Exception as e:
                logger.error(f"Error evaluating task {task_id}: {str(e)}", exc_info=True)
                # Continue with other tasks on error
                continue
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Create benchmark report
        report = BenchmarkReport(
            run_id=run_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_name=model_id,
            task_results=task_results,
            metadata={
                'total_time_seconds': total_time,
                'num_tasks': len(task_results),
                'batch_size': batch_size,
                'model_type': model_type,
                **model_kwargs
            }
        )
        
        # Save results
        if save_results:
            output_file = self.results_dir / f"{run_id}.json"
            report.save(output_file)
            logger.info(f"Saved evaluation results to {output_file}")
        
        return report
    
    def _evaluate_task(
        self,
        model: Any,
        model_id: str,
        task: MedicalTask,
        batch_size: int = 8
    ) -> 'TaskResult':
        """Evaluate a single task."""
        from .result_aggregator import TaskResult
        
        # Prepare inputs
        inputs = task.dataset
        if not inputs:
            raise ValueError(f"No dataset found for task {task.task_id}")
        
        # Run model inference
        logger.info(f"Running inference on {len(inputs)} examples...")
        predictions = self.model_runner.run_model(
            model_id=model_id,
            inputs=inputs,
            batch_size=batch_size
        )
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metric_results = self.metric_calculator.calculate_metrics(
            task_id=task.task_id,
            predictions=predictions,
            references=inputs,
            metric_names=[m['name'] for m in task.metrics]
        )
        
        # Extract metric values
        metrics = {name: result.value for name, result in metric_results.items()}
        
        # Create task result
        return TaskResult(
            task_id=task.task_id,
            metrics=metrics,
            num_examples=len(inputs),
            metadata={
                'metrics_metadata': {
                    name: asdict(result) 
                    for name, result in metric_results.items()
                },
                'predictions': predictions[:10]  # Store first few predictions for reference
            }
        )
    
    def _generate_run_id(self, model_id: str, task_ids: List[str]) -> str:
        """Generate a unique run ID based on model and tasks."""
        import hashlib
        import time
        
        # Create a unique string from model ID and task IDs
        unique_str = f"{model_id}_{'_'.join(sorted(task_ids))}_{int(time.time())}"
        
        # Create a hash of the unique string
        return hashlib.md5(unique_str.encode()).hexdigest()[:8]
    
    def _get_cache_key(self, run_id: str, task_id: str) -> str:
        """Generate a cache key for storing/retrieving results."""
        if not self.cache_dir:
            return None
        return str(self.cache_dir / f"{run_id}_{task_id}.json")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load results from cache if available."""
        if not cache_key or not Path(cache_key).exists():
            return None
            
        try:
            with open(cache_key, 'r') as f:
                data = json.load(f)
                
            from .result_aggregator import TaskResult
            return TaskResult(**data)
            
        except Exception as e:
            logger.warning(f"Error loading from cache {cache_key}: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save results to cache."""
        if not cache_key:
            return
            
        try:
            with open(cache_key, 'w') as f:
                json.dump(asdict(data), f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving to cache {cache_key}: {str(e)}")
    
    def list_available_tasks(self) -> List[Dict[str, Any]]:
        """List all available tasks in the tasks directory."""
        tasks = []
        
        # Look for YAML task definitions
        for task_file in self.tasks_dir.glob("*.yaml"):
            try:
                with open(task_file, 'r') as f:
                    task_data = yaml.safe_load(f)
                    tasks.append({
                        'task_id': task_file.stem,
                        'name': task_data.get('name', ''),
                        'description': task_data.get('description', ''),
                        'metrics': [m['name'] for m in task_data.get('metrics', [])],
                        'num_examples': len(task_data.get('dataset', [])),
                        'file': str(task_file)
                    })
            except Exception as e:
                logger.warning(f"Error loading task from {task_file}: {str(e)}")
        
        # Look for JSON task definitions
        for task_file in self.tasks_dir.glob("*.json"):
            if task_file.stem == 'tasks':  # Skip the tasks index file
                continue
                
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    tasks.append({
                        'task_id': task_file.stem,
                        'name': task_data.get('name', ''),
                        'description': task_data.get('description', ''),
                        'metrics': [m['name'] for m in task_data.get('metrics', [])],
                        'num_examples': len(task_data.get('dataset', [])),
                        'file': str(task_file)
                    })
            except Exception as e:
                logger.warning(f"Error loading task from {task_file}: {str(e)}")
        
        return tasks
    
    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific task."""
        task = self.task_loader.load_task(task_id)
        
        return {
            'task_id': task.task_id,
            'name': task.name,
            'description': task.description,
            'input_schema': task.input_schema,
            'output_schema': task.output_schema,
            'metrics': task.metrics,
            'num_examples': len(task.dataset) if task.dataset else 0,
            'example_input': task.dataset[0] if task.dataset else None
        }
