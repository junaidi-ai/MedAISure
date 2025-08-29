import math

from bench.evaluation.metrics import DiagnosticAccuracyMetric, ClinicalRelevanceMetric


def test_diagnostic_accuracy_with_specialty_weighting_and_breakdown():
    m = DiagnosticAccuracyMetric()
    expected = [
        {"diagnosis": "myocardial infarction", "specialty": "cardiology"},
        {"diagnosis": "sepsis", "specialty": "infectious_disease"},
        {"diagnosis": "pneumonia", "specialty": "infectious_disease"},
    ]
    outputs = [
        {"prediction": "mi", "specialty": "cardiology"},  # correct (normalized)
        {"prediction": "flu", "specialty": "infectious_disease"},  # incorrect critical
        {"prediction": "pneumonia", "specialty": "infectious_disease"},  # correct
    ]

    score = m.calculate(expected, outputs)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0

    br = m.get_last_breakdown()
    assert isinstance(br, list) and len(br) == 3
    assert "weight" in br[0] and "correct" in br[0]


def test_clinical_relevance_thresholds_and_boost_with_breakdown():
    m = ClinicalRelevanceMetric()
    expected = [
        {
            "note": "patient with fever cough tachycardia",
            "specialty": "infectious_disease",
        },
        {"note": "chest pain troponin elevated", "specialty": "cardiology"},
    ]
    outputs = [
        {
            "summary": "fever and cough present; tachycardia noted"
        },  # high overlap + keywords
        {
            "summary": "patient has chest pain and elevated troponin levels"
        },  # overlap + boost
    ]

    score = m.calculate(expected, outputs)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0

    br = m.get_last_breakdown()
    assert isinstance(br, list) and len(br) == 2
    assert "final_score" in br[0]
    assert "threshold" in br[0]
