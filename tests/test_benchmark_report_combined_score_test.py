from bench.models.benchmark_report import BenchmarkReport


def make_report():
    # Create a minimal report with two tasks and base metrics
    return BenchmarkReport(
        model_id="test-model",
        overall_scores={},
        task_scores={
            "task1": {"diagnostics": 0.8, "safety": 0.6},
            "task2": {"diagnostics": 0.4},  # safety missing on task2
        },
        detailed_results=[],
        metadata={},
    )


def test_combined_score_with_renormalization():
    r = make_report()
    weights = {"diagnostics": 0.5, "safety": 0.5}

    r.add_combined_score(
        weights, metric_name="combined_score", renormalize_missing=True
    )

    # task1: 0.5*0.8 + 0.5*0.6 = 0.7
    # task2: renormalize -> diagnostics weight becomes 1.0 -> 1.0*0.4 = 0.4
    assert abs(r.task_scores["task1"]["combined_score"] - 0.7) < 1e-9
    assert abs(r.task_scores["task2"]["combined_score"] - 0.4) < 1e-9

    # overall is mean over tasks that have combined_score
    assert abs(r.overall_scores["combined_score"] - ((0.7 + 0.4) / 2.0)) < 1e-9


def test_combined_score_without_renormalization():
    r = make_report()
    weights = {"diagnostics": 0.5, "safety": 0.5}

    r.add_combined_score(weights, metric_name="combo_raw", renormalize_missing=False)

    # task1: same as above -> 0.7 because total_w=1.0
    # task2: total_w=1.0 but only diagnostics present -> 0.5*0.4 = 0.2
    assert abs(r.task_scores["task1"]["combo_raw"] - 0.7) < 1e-9
    assert abs(r.task_scores["task2"]["combo_raw"] - 0.2) < 1e-9
    assert abs(r.overall_scores["combo_raw"] - ((0.7 + 0.2) / 2.0)) < 1e-9
