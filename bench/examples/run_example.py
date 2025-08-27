"""
Example script demonstrating how to use the MedAISure evaluation framework
with a Hugging Face model.
"""

import logging
import sys
import time
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging before other imports that might log
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from bench.evaluation import EvaluationHarness  # noqa: E402

logger = logging.getLogger(__name__)


def run_example() -> None:
    """
    Run an example evaluation using the medical NLI task
    with a Hugging Face model.
    """
    # Define paths
    base_dir: Path = Path(__file__).parent.parent
    tasks_dir: Path = base_dir / "tasks"
    results_dir: Path = base_dir / "results"

    # Ensure results directory exists
    results_dir.mkdir(exist_ok=True, parents=True)

    # Initialize the evaluation harness
    harness = EvaluationHarness(
        tasks_dir=str(tasks_dir),
        results_dir=str(results_dir),
        cache_dir=str(results_dir / "cache"),
        log_level="INFO",
    )

    # List available tasks
    print("\n=== Available Tasks ===")
    available_tasks = harness.list_available_tasks()
    for available_task in available_tasks:
        task_id = available_task["task_id"]
        task_name = available_task["name"]
        num_examples = available_task["num_examples"]
        print(f"- {task_id}: {task_name} ({num_examples} examples)")

    if not available_tasks:
        print("No tasks found. Make sure you have task files in the tasks directory.")
        return

    # Use the first available task
    task_id = available_tasks[0]["task_id"]

    # Show task details
    print(f"\n=== Task Details: {task_id} ===")
    task_info = harness.get_task_info(task_id)
    print(f"Name: {task_info['name']}")
    print(f"Description: {task_info['description']}")
    print(f"Metrics: {[m['name'] for m in task_info['metrics']]}")
    print(f"Example input: {task_info['example_input']}")

    # Run evaluation with a Hugging Face model
    print("\n=== Starting Evaluation ===")
    print("This may take a while as it downloads the model if not already cached...")

    try:
        # Use a pre-fine-tuned model for NLI tasks
        model_id = "textattack/bert-base-uncased-MNLI"
        print(f"Loading model: {model_id}")

        # Define label mapping for the model's output
        label_map = {
            "LABEL_0": "contradiction",
            "LABEL_1": "neutral",
            "LABEL_2": "entailment",
        }

        # Disable cache to ensure we get fresh results
        use_cache = False

        # Load the task to inspect the dataset
        from bench.models.medical_task import MedicalTask

        task: MedicalTask = harness.task_loader.load_task(task_id)
        print(f"\nTask dataset has {len(task.dataset) if task.dataset else 0} examples")
        if task.dataset:
            sample_labels = [item.get("label", "N/A") for item in task.dataset[:5]]
            print("Sample ground truth labels:", sample_labels)

        # Run evaluation with debug information
        print("\nRunning evaluation...")
        report = harness.evaluate(
            model_id=model_id,
            task_ids=[task_id],
            model_type="huggingface",
            model_kwargs={
                "num_labels": 3,  # For entailment, contradiction, neutral
            },
            pipeline_kwargs={
                "top_k": 1,  # Only return the top prediction
            },
            label_map=label_map,  # Add label mapping
            batch_size=4,
            use_cache=use_cache,  # Disable cache to ensure fresh results
        )

        # Save results
        timestamp = int(time.time())
        results_file = results_dir / f"evaluation_results_{timestamp}.json"

        # Convert report to dict for JSON serialization
        report_dict = {
            "model_name": report.model_name,
            "task_results": {
                task_id: {
                    "metrics": task_result.metrics,
                    "num_examples": task_result.num_examples,
                }
                for task_id, task_result in report.task_results.items()
            },
            "timestamp": timestamp,
        }

        with open(results_file, "w") as f:
            import json

            json.dump(report_dict, f, indent=2)

        print("\n=== Evaluation Complete ===")
        print(f"Results saved to: {results_file}")
        print(f"Model: {report.model_name}")
        print(f"Tasks evaluated: {list(report.task_results.keys())}")
        print("\nTask Results:")
        for task_id, task_result in report.task_results.items():
            print(f"- {task_id}:")
            for metric_name, metric_value in task_result.metrics.items():
                print(f"  - {metric_name}: {metric_value:.4f}")

    except Exception as e:
        logger.error("Error during evaluation: %s", str(e), exc_info=True)
        raise


def main() -> None:
    """Main entry point for the example script."""
    try:
        run_example()
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
