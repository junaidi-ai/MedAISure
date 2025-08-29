import math
from typing import List

import hypothesis.strategies as st
from hypothesis import given, settings

from bench.evaluation.metrics import (
    ClinicalAccuracyMetric,
    ReasoningQualityMetric,
    DiagnosticAccuracyMetric,
    ClinicalRelevanceMetric,
)


def _dicts_same_length(keys_left: List[str], keys_right: List[str]):
    """Return a SearchStrategy that yields (expected, outputs) lists of equal length."""

    def builder(n: int):
        base_text = st.one_of(
            st.text(min_size=0, max_size=50), st.integers(), st.none()
        )
        left_item = st.fixed_dictionaries({k: base_text for k in keys_left}) | st.just(
            {}
        )
        right_item = st.fixed_dictionaries(
            {k: base_text for k in keys_right}
        ) | st.just({})
        left_list = st.lists(left_item, min_size=n, max_size=n)
        right_list = st.lists(right_item, min_size=n, max_size=n)
        return st.tuples(left_list, right_list)

    return st.integers(min_value=1, max_value=20).flatmap(builder)


@settings(max_examples=60)
@given(_dicts_same_length(["answer"], ["prediction"]))
def test_clinical_accuracy_property_score_bounds(data):
    expected, outputs = data
    m = ClinicalAccuracyMetric()
    score = m.calculate(expected, outputs)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0


@settings(max_examples=60)
@given(_dicts_same_length(["rationale"], ["rationale"]))
def test_reasoning_quality_property_score_bounds_and_determinism(data):
    expected, outputs = data
    m = ReasoningQualityMetric()
    score1 = m.calculate(expected, outputs)
    score2 = m.calculate(expected, outputs)
    assert isinstance(score1, float) and isinstance(score2, float)
    assert not math.isnan(score1) and not math.isnan(score2)
    assert 0.0 <= score1 <= 1.0
    assert 0.0 <= score2 <= 1.0
    assert score1 == score2  # deterministic for same inputs


@settings(max_examples=60)
@given(_dicts_same_length(["diagnosis", "specialty"], ["prediction", "specialty"]))
def test_diagnostic_accuracy_property_score_bounds(data):
    expected, outputs = data
    m = DiagnosticAccuracyMetric()
    score = m.calculate(expected, outputs)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0


@settings(max_examples=60)
@given(_dicts_same_length(["note", "specialty"], ["summary", "specialty"]))
def test_clinical_relevance_property_score_bounds(data):
    expected, outputs = data
    m = ClinicalRelevanceMetric()
    score = m.calculate(expected, outputs)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0
