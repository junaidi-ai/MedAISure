"""
Run a Hugging Face summarization pipeline on a summarization-style task.
"""

import logging
import sys
from pathlib import Path

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

    # pick a task; ensure dataset has 'summary' reference fields for rouge
    task_id = tasks[0]["task_id"]

    model_id = "sshleifer/distilbart-cnn-12-6"
    report = harness.evaluate(
        model_id=model_id,
        task_ids=[task_id],
        model_type="huggingface",
        hf_task="summarization",
        pipeline_kwargs={"max_new_tokens": 64},
        batch_size=2,
        use_cache=False,
    )
    print("overall:", report.overall_scores)


if __name__ == "__main__":
    main()
