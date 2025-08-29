"""
Demonstrate registering an API-based model.
The endpoint should accept POST JSON body (list of inputs) and return a list of prediction dicts.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.evaluation import EvaluationHarness  # noqa: E402


def main() -> None:
    base = Path(__file__).parent.parent
    h = EvaluationHarness(
        tasks_dir=str(base / "tasks"),
        results_dir=str(base / "results"),
        cache_dir=str(base / "results" / "cache"),
    )

    tasks = h.list_available_tasks()
    if not tasks:
        print("No tasks found.")
        return

    task_id = tasks[0]["task_id"]

    report = h.evaluate(
        model_id="my_api_model",
        task_ids=[task_id],
        model_type="api",
        endpoint=os.environ.get("MY_API_ENDPOINT", "https://example.com/predict"),
        api_key=os.environ.get("MY_API_KEY", "demo-token"),
        timeout=15.0,
        max_retries=1,
        backoff_factor=0.5,
    )

    print("overall:", report.overall_scores)


if __name__ == "__main__":
    main()
