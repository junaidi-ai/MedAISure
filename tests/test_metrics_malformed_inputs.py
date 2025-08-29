import math
import pytest

from bench.evaluation.metrics import (
    ClinicalAccuracyMetric,
    ReasoningQualityMetric,
    DiagnosticAccuracyMetric,
    ClinicalRelevanceMetric,
)


@pytest.mark.parametrize(
    "metric_cls, exp_key, out_key",
    [
        (ClinicalAccuracyMetric, "answer", "prediction"),
        (ReasoningQualityMetric, "rationale", "rationale"),
        (DiagnosticAccuracyMetric, "diagnosis", "prediction"),
        (ClinicalRelevanceMetric, "note", "summary"),
    ],
)
def test_metrics_handle_malformed_and_unexpected_inputs(metric_cls, exp_key, out_key):
    m = metric_cls()

    # Mix of malformed dictionaries and unexpected types in values
    expected = [
        {},  # missing keys
        {exp_key: None},  # None value
        {exp_key: 123},  # non-string value
        {exp_key: ""},  # empty string
        {exp_key: "some text", "extra": {"nested": [1, 2, 3]}},  # extras
    ]
    outputs = [
        {},
        {out_key: None},
        {out_key: 456},
        {out_key: ""},
        {out_key: "some text", "weird": object()},
    ]

    score = m.calculate(expected, outputs)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    "metric_cls, exp_key, out_key",
    [
        (ClinicalAccuracyMetric, "answer", "prediction"),
        (ReasoningQualityMetric, "rationale", "rationale"),
        (DiagnosticAccuracyMetric, "diagnosis", "prediction"),
        (ClinicalRelevanceMetric, "note", "summary"),
    ],
)
def test_metrics_reject_list_shape_issues(metric_cls, exp_key, out_key):
    m = metric_cls()

    # Length mismatch should raise ValueError via Metric.validate_inputs
    with pytest.raises(ValueError):
        m.calculate([{exp_key: "a"}], [{out_key: "a"}, {out_key: "b"}])

    # Non-dict elements should raise TypeError
    with pytest.raises(TypeError):
        m.calculate([{exp_key: "a"}], ["bad"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        m.calculate(["bad"], [{out_key: "a"}])  # type: ignore[list-item]
