import math

from bench.evaluation.metrics.clinical import ClinicalAccuracyMetric


def test_clinical_accuracy_weighted_entities_basic():
    m = ClinicalAccuracyMetric()
    expected = [
        {"answer": "Pneumonia with fever and cough. Start antibiotics."},
        {"answer": "Sepsis with hypotension; give IV fluids and antibiotics."},
        {"answer": "Myocardial infarction treated with aspirin and heparin."},
    ]
    outputs = [
        {"prediction": "pneumonia patient has cough and fever, on antibiotics"},
        {"prediction": "sepsis case hypotension managed with iv fluids"},
        {"prediction": "MI patient on aspirin"},
    ]

    score = m.calculate(expected, outputs)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0

    breakdown = m.get_last_breakdown()
    assert isinstance(breakdown, list)
    assert len(breakdown) == 3
    # Ensure category scores are present for at least one item
    assert "category_scores" in breakdown[0]


def test_clinical_accuracy_fallback_string_match():
    m = ClinicalAccuracyMetric()
    expected = [{"answer": "flu"}]
    outputs = [{"prediction": "the patient likely has flu"}]

    score = m.calculate(expected, outputs)
    # substring fallback should give 1.0
    assert score == 1.0


def test_clinical_accuracy_empty_inputs_behaviour():
    m = ClinicalAccuracyMetric()
    expected = [{"answer": ""}]
    outputs = [{"prediction": ""}]

    score = m.calculate(expected, outputs)
    assert score == 1.0
