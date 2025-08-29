"""
End-to-end example using a local Python model (model_type="local").
"""

import logging
import sys
from pathlib import Path

# Ensure package imports work when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from bench.evaluation import EvaluationHarness  # noqa: E402


def main() -> None:
    base = Path(__file__).parent.parent
    harness = EvaluationHarness(
        tasks_dir=str(base / "tasks"),
        results_dir=str(base / "results"),
        cache_dir=str(base / "results" / "cache"),
        log_level="INFO",
    )

    tasks = harness.list_available_tasks()
    if not tasks:
        print("No tasks found in tasks dir.")
        return

    task_id = tasks[0]["task_id"]

    # The local model loader lives in bench/examples/mypkg/mylocal.py
    report = harness.evaluate(
        model_id="my_local_model",
        task_ids=[task_id],
        model_type="local",
        module_path="bench.examples.mypkg.mylocal",
        # optional: load_func="load_model", model_path="/path/to/weights",
        batch_size=4,
        use_cache=False,
    )

    print("overall:", report.overall_scores)


if __name__ == "__main__":
    main()
