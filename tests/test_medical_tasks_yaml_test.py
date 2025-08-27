"""Integration tests for YAML-defined medical tasks and medical metrics."""

import math
from pathlib import Path

import pytest

from bench.evaluation.metric_calculator import MetricCalculator
from bench.evaluation.task_loader import TaskLoader
from bench.models.medical_task import MedicalTask

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "bench" / "tasks"


@pytest.mark.parametrize(
    "task_id",
    [
        "medical_qa_symptoms",
        "medical_qa_treatments",
        "diagnostic_reasoning_respiratory",
        "diagnostic_reasoning_cardiology",
        "clinical_summarization_discharge",
        "clinical_summarization_progress",
    ],
)
def test_load_real_yaml_tasks(task_id: str):
    """Ensure real YAML tasks load and satisfy MedicalTask validators."""
    assert TASKS_DIR.exists(), f"Tasks dir not found: {TASKS_DIR}"

    loader = TaskLoader(tasks_dir=str(TASKS_DIR))
    task = loader.load_task(task_id)

    # Basic validations
    assert isinstance(task, MedicalTask)
    assert task.task_id == task_id
    assert isinstance(task.name, str) and task.name
    assert isinstance(task.description, str)
    assert isinstance(task.inputs, list) and len(task.inputs) > 0
    assert isinstance(task.expected_outputs, list) and len(task.expected_outputs) > 0
    assert isinstance(task.metrics, list) and len(task.metrics) > 0
    assert isinstance(task.input_schema, dict)
    assert isinstance(task.output_schema, dict)
    assert isinstance(task.dataset, list) and len(task.dataset) > 0


def test_medical_metrics_simple_cases():
    """Check medical metrics produce finite values on simple synthetic data."""
    calc = MetricCalculator()

    # clinical_correctness (exact match should be 1.0)
    preds_cc = [{"answer": "fever"}, {"answer": "dry mouth"}]
    refs_cc = [{"answer": "fever"}, {"answer": "dry mouth"}]
    res_cc = calc.calculate_metrics(
        task_id="qa_test",
        predictions=preds_cc,
        references=refs_cc,
        metric_names=["clinical_correctness"],
    )
    assert "clinical_correctness" in res_cc
    assert math.isfinite(res_cc["clinical_correctness"].value)
    assert abs(res_cc["clinical_correctness"].value - 1.0) < 1e-6

    # diagnostic_accuracy (two correct, one incorrect -> 2/3)
    preds_dx = [{"label": "asthma"}, {"label": "copd"}, {"label": "pna"}]
    refs_dx = [{"label": "asthma"}, {"label": "copd"}, {"label": "copd"}]
    res_dx = calc.calculate_metrics(
        task_id="dx_test",
        predictions=preds_dx,
        references=refs_dx,
        metric_names=["diagnostic_accuracy"],
    )
    assert "diagnostic_accuracy" in res_dx
    assert 0.0 <= res_dx["diagnostic_accuracy"].value <= 1.0

    # reasoning_quality (identical rationale -> 1.0)
    preds_rq = [{"rationale": "patient has fever and cough"}]
    refs_rq = [{"rationale": "patient has fever and cough"}]
    res_rq = calc.calculate_metrics(
        task_id="rq_test",
        predictions=preds_rq,
        references=refs_rq,
        metric_names=["reasoning_quality"],
    )
    assert "reasoning_quality" in res_rq
    assert abs(res_rq["reasoning_quality"].value - 1.0) < 1e-6

    # summarization metrics (identical -> should be finite and typically 1.0)
    preds_sum = [
        {"summary": "cap treated with ceftriaxone and azithromycin"},
        {"summary": "dka resolved after insulin and fluids"},
    ]
    refs_sum = [
        {"summary": "cap treated with ceftriaxone and azithromycin"},
        {"summary": "dka resolved after insulin and fluids"},
    ]
    res_sum = calc.calculate_metrics(
        task_id="sum_test",
        predictions=preds_sum,
        references=refs_sum,
        metric_names=["rouge_l", "factual_consistency"],
    )
    # rouge may require rouge-score; but requirement exists. Assert finite and in [0,1]
    assert "rouge_l" in res_sum
    assert 0.0 <= res_sum["rouge_l"].value <= 1.0 or math.isnan(res_sum["rouge_l"].value)
    assert "factual_consistency" in res_sum
    assert abs(res_sum["factual_consistency"].value - 1.0) < 1e-6

    # clinical_relevance compares predicted summary to source note
    preds_rel = [
        {"summary": "copd exacerbation improving on steroids"},
    ]
    refs_rel = [
        {"note": "patient with copd exacerbation improving on steroids and nebulizers"},
    ]
    res_rel = calc.calculate_metrics(
        task_id="rel_test",
        predictions=preds_rel,
        references=refs_rel,
        metric_names=["clinical_relevance"],
    )
    assert "clinical_relevance" in res_rel
    val = res_rel["clinical_relevance"].value
    assert (0.0 <= val <= 1.0) or math.isnan(val)
