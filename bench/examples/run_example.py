"""Example script demonstrating how to use the MEDDSAI evaluation framework."""
import os
import sys
import logging
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.evaluation import EvaluationHarness

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_example():
    """Run an example evaluation using the medical NLI task."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    tasks_dir = base_dir / "examples"
    results_dir = base_dir / "results"
    
    # Initialize the evaluation harness
    harness = EvaluationHarness(
        tasks_dir=str(tasks_dir),
        results_dir=str(results_dir),
        cache_dir=str(results_dir / "cache"),
        log_level="INFO"
    )
    
    # List available tasks
    print("\n=== Available Tasks ===")
    tasks = harness.list_available_tasks()
    for task in tasks:
        print(f"- {task['task_id']}: {task['name']} ({task['num_examples']} examples)")
    
    if not tasks:
        print("No tasks found. Make sure you have task files in the examples directory.")
        return
    
    # Use the first available task
    task_id = tasks[0]['task_id']
    
    # Show task details
    print(f"\n=== Task Details: {task_id} ===")
    task_info = harness.get_task_info(task_id)
    print(f"Name: {task_info['name']}")
    print(f"Description: {task_info['description']}")
    print(f"Metrics: {[m['name'] for m in task_info['metrics']]}")
    print(f"Example input: {task_info['example_input']}")
    
    # Define a simple model for demonstration
    class SimpleNLIModel:
        """A simple rule-based model for demonstration purposes."""
        def __init__(self):
            self.entailment_phrases = [
                ("has a history of", "has"),
                ("prescribed", "taking"),
                ("experiencing", "has")
            ]
            self.contradiction_phrases = [
                ("has a history of", "does not have"),
                ("prescribed", "allergic to"),
                ("elevated", "normal")
            ]
        
        def __call__(self, inputs, **kwargs):
            """Make predictions on a batch of inputs."""
            results = []
            for item in inputs:
                premise = item['premise'].lower()
                hypothesis = item['hypothesis'].lower()
                
                # Simple rule-based classification
                if any(p in premise and h in hypothesis for p, h in self.entailment_phrases):
                    label = "entailment"
                elif any(p in premise and h in hypothesis for p, h in self.contradiction_phrases):
                    label = "contradiction"
                else:
                    label = "neutral"
                
                results.append({
                    'label': label,
                    'confidence': 0.9  # Dummy confidence
                })
            
            return results
    
    # Register our simple model
    model_id = "simple_nli_model"
    model = SimpleNLIModel()
    
    # Run evaluation
    print("\n=== Starting Evaluation ===")
    report = harness.evaluate(
        model_id=model_id,
        task_ids=[task_id],
        model_type="local",  # We're using a local Python model
        model=model,  # Pass our model instance directly
        batch_size=4,
        use_cache=True,
        save_results=True
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Model: {report.model_name}")
    print(f"Run ID: {report.run_id}")
    print(f"Timestamp: {report.timestamp}")
    
    for task_id, result in report.task_results.items():
        print(f"\nTask: {task_id}")
        print(f"Number of examples: {result.num_examples}")
        print("Metrics:")
        for metric_name, value in result.metrics.items():
            print(f"  - {metric_name}: {value:.4f}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    run_example()
