from bench.evaluation.metrics import (
    ClinicalAccuracyMetric,
    ReasoningQualityMetric,
    DiagnosticAccuracyMetric,
    ClinicalRelevanceMetric,
)


def _make_bulk(n: int, exp_key: str, out_key: str):
    expected = []
    outputs = []
    for i in range(n):
        expected.append({exp_key: f"patient {i} with fever cough and tachycardia"})
        outputs.append({out_key: f"patient {i} fever and cough present tachycardia"})
    return expected, outputs


def test_perf_clinical_accuracy_large_inputs(benchmark):
    expected, outputs = _make_bulk(1000, "answer", "prediction")
    m = ClinicalAccuracyMetric()

    def run():
        return m.calculate(expected, outputs)

    score = benchmark(run)
    assert 0.0 <= score <= 1.0


def test_perf_reasoning_quality_large_inputs(benchmark):
    expected, outputs = _make_bulk(1000, "rationale", "rationale")
    m = ReasoningQualityMetric()

    def run():
        return m.calculate(expected, outputs)

    score = benchmark(run)
    assert 0.0 <= score <= 1.0


def test_perf_diagnostic_accuracy_large_inputs(benchmark):
    n = 1000
    expected = [
        {"diagnosis": "pneumonia", "specialty": "infectious_disease"} for _ in range(n)
    ]
    outputs = [
        {"prediction": "pneumonia", "specialty": "infectious_disease"} for _ in range(n)
    ]
    m = DiagnosticAccuracyMetric()

    def run():
        return m.calculate(expected, outputs)

    score = benchmark(run)
    assert 0.0 <= score <= 1.0


def test_perf_clinical_relevance_large_inputs(benchmark):
    n = 1000
    expected = [
        {"note": "fever cough tachycardia", "specialty": "infectious_disease"}
        for _ in range(n)
    ]
    outputs = [
        {
            "summary": "fever and cough present tachycardia",
            "specialty": "infectious_disease",
        }
        for _ in range(n)
    ]
    m = ClinicalRelevanceMetric()

    def run():
        return m.calculate(expected, outputs)

    score = benchmark(run)
    assert 0.0 <= score <= 1.0
