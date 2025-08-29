"""Property-based tests for MedAISure data models.

Covers randomized validation and round-trip serialization for:
- MedicalTask
- EvaluationResult
- BenchmarkReport
"""

from __future__ import annotations

from typing import Dict, Any, List

from hypothesis import given, settings
from hypothesis import strategies as st

from bench.models import BenchmarkReport, EvaluationResult, MedicalTask, TaskType


# ---- Strategies ----


def non_empty_strs():
    return st.text(min_size=1).filter(lambda s: s.strip() != "")


def dicts_of_simple_values():
    # Keep values simple to avoid serialization edge issues
    return st.dictionaries(
        keys=non_empty_strs(),
        values=st.one_of(
            st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text()
        ),
    )


@st.composite
def medical_task_strategy(draw):
    task_id = draw(non_empty_strs())
    task_type = draw(st.sampled_from(list(TaskType)))
    inputs = draw(st.lists(dicts_of_simple_values(), min_size=1, max_size=5))
    # metrics must be unique, non-empty
    metrics = draw(st.lists(non_empty_strs(), min_size=1, max_size=5, unique=True))
    # To keep constraints simpler, omit expected_outputs and schemas (optional)
    data = {
        "task_id": task_id,
        "task_type": task_type.value,
        "description": "",
        "inputs": inputs,
        "metrics": metrics,
    }
    return data


@st.composite
def evaluation_result_strategy(draw):
    model_id = draw(non_empty_strs())
    task_id = draw(non_empty_strs())
    inputs = draw(st.lists(dicts_of_simple_values(), min_size=0, max_size=5))
    # outputs must match inputs length if inputs provided
    if inputs:
        outs = draw(
            st.lists(
                dicts_of_simple_values(), min_size=len(inputs), max_size=len(inputs)
            )
        )
    else:
        outs = draw(st.lists(dicts_of_simple_values(), min_size=0, max_size=5))
    metrics_results = draw(
        st.dictionaries(
            keys=non_empty_strs(),
            values=st.floats(allow_nan=False, allow_infinity=False),
            max_size=5,
        )
    )
    return {
        "model_id": model_id,
        "task_id": task_id,
        "inputs": inputs,
        "model_outputs": outs,
        "metrics_results": metrics_results,
    }


# ---- Property tests ----


@settings(max_examples=50)
@given(medical_task_strategy())
def test_medical_task_round_trip(task_data: Dict[str, Any]):
    task = MedicalTask.model_validate(task_data)
    dumped = task.model_dump()
    # JSON round-trip
    rtrip = MedicalTask.model_validate_json(task.model_dump_json())
    assert rtrip.model_dump() == dumped


@settings(max_examples=50)
@given(evaluation_result_strategy())
def test_evaluation_result_round_trip(eval_data: Dict[str, Any]):
    res = EvaluationResult.model_validate(eval_data)
    dumped = res.model_dump(exclude={"timestamp"})
    rtrip = EvaluationResult.model_validate_json(res.model_dump_json())
    assert rtrip.model_dump(exclude={"timestamp"}) == dumped


@settings(max_examples=25)
@given(
    st.builds(
        lambda model_id, evals: {
            "model_id": model_id,
            "detailed_results": evals,
        },
        model_id=non_empty_strs(),
        evals=st.lists(evaluation_result_strategy(), min_size=0, max_size=5),
    )
)
def test_benchmark_report_add_and_overall(data: Dict[str, Any]):
    # Ensure detailed_results model_id matches report.model_id for validation to pass
    model_id = data["model_id"]
    aligned_results: List[Dict[str, Any]] = []
    for item in data["detailed_results"]:
        item = dict(item)
        item["model_id"] = model_id
        aligned_results.append(item)

    report = BenchmarkReport(model_id=model_id, detailed_results=[])
    for r in aligned_results:
        report.add_evaluation_result(EvaluationResult(**r))

    # overall_scores keys should be subset of union of all metric keys
    metric_union = set()
    for r in aligned_results:
        metric_union |= set((r.get("metrics_results") or {}).keys())
    assert set(report.overall_scores.keys()).issubset(metric_union)
