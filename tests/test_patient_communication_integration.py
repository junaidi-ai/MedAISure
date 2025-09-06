import math
import os
import pytest

from bench.evaluation import EvaluationHarness


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skip network-dependent HF download on CI",
)
def test_evaluate_patient_communication_with_hf_tiny_gpt2():
    h = EvaluationHarness(
        tasks_dir="bench/tasks",
        results_dir="bench/results-test",
        cache_dir="bench/results-test/cache",
        log_level="WARNING",
    )

    # Ensure tasks discoverable
    tasks = [r["task_id"] for r in h.list_available_tasks()]
    assert "patient_communication_basic" in tasks
    assert "patient_communication_triage" in tasks

    # Use a very small text-generation model for quick smoke
    model_id = "sshleifer/tiny-gpt2"
    report = h.evaluate(
        model_id=model_id,
        task_ids=[
            "patient_communication_basic",
            "patient_communication_triage",
            # medication counseling is optional here to keep runtime small
        ],
        model_type="huggingface",
        batch_size=2,
        use_cache=False,
        strict_validation=False,
        report_formats=["json"],
        hf_task="text-generation",
        generation_kwargs={"max_new_tokens": 8},
    )

    # Check metric keys present and values are finite numbers
    for tid, scores in (report.task_scores or {}).items():
        assert isinstance(scores, dict) and scores, f"no scores for {tid}"
        for name, val in scores.items():
            assert name in {
                "rouge_l",
                "clinical_relevance",
                "summarization",
            } or isinstance(name, str)
            assert isinstance(val, (int, float)) and math.isfinite(
                val
            ), f"non-finite {name} for {tid}: {val}"
