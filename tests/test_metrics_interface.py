import math

from bench.evaluation.metrics import (
    ClinicalAccuracyMetric,
    ClinicalRelevanceMetric,
    DiagnosticAccuracyMetric,
    ReasoningQualityMetric,
    MetricRegistry,
)


def test_metric_registry_and_metrics_basic():
    reg = MetricRegistry()
    reg.register_metric(ClinicalAccuracyMetric())
    reg.register_metric(ReasoningQualityMetric())
    reg.register_metric(DiagnosticAccuracyMetric())
    reg.register_metric(ClinicalRelevanceMetric())

    expected = [
        {
            "answer": "Pneumonia",
            "rationale": "Because X",
            "label": "A",
            "note": "cough fever",
        },
        {
            "answer": "Sepsis",
            "rationale": "Because Y",
            "label": "B",
            "note": "tachycardia hypotension",
        },
    ]
    outputs = [
        {
            "prediction": "pneumonia",
            "rationale": "because x",
            "label": "A",
            "summary": "fever cough",
        },
        {
            "prediction": "flu",
            "explanation": "because y",
            "label": "B",
            "summary": "tachycardia hypotension patient",
        },
    ]

    results = reg.calculate_metrics(
        [
            "clinical_accuracy",
            "reasoning_quality",
            "diagnostic_accuracy",
            "clinical_relevance",
        ],
        expected,
        outputs,
    )

    assert set(results.keys()) == {
        "clinical_accuracy",
        "reasoning_quality",
        "diagnostic_accuracy",
        "clinical_relevance",
    }

    # Scores should be finite numbers (0..1) where defined
    for name, val in results.items():
        assert isinstance(val, float)
        assert not math.isnan(val)
        assert 0.0 <= val <= 1.0
