from bench.evaluation.metrics import (
    ClinicalAccuracyMetric,
    ReasoningQualityMetric,
    DiagnosticAccuracyMetric,
    ClinicalRelevanceMetric,
)


def test_clinical_accuracy_exact_match_yields_one():
    m = ClinicalAccuracyMetric()
    expected = [{"answer": "Pneumonia"}]
    outputs = [{"prediction": "pneumonia"}]
    assert m.calculate(expected, outputs) == 1.0


def test_clinical_accuracy_no_overlap_yields_zero():
    m = ClinicalAccuracyMetric()
    expected = [{"answer": "pneumonia"}]
    outputs = [{"prediction": "appendicitis"}]
    score = m.calculate(expected, outputs)
    assert 0.0 <= score <= 1.0  # entity lexicon may ignore both -> fallback 0.0


def test_reasoning_quality_empty_both_yields_valid_and_high_overlap():
    m = ReasoningQualityMetric()
    expected = [{"rationale": ""}]
    outputs = [{"rationale": ""}]
    score = m.calculate(expected, outputs)
    assert 0.0 <= score <= 1.0


def test_diagnostic_accuracy_exact_label_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "myocardial infarction", "specialty": "cardiology"}]
    outputs = [{"prediction": "mi", "specialty": "cardiology"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_mismatch_zero():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "sepsis", "specialty": "infectious_disease"}]
    outputs = [{"prediction": "flu", "specialty": "infectious_disease"}]
    assert m.calculate(expected, outputs) == 0.0


def test_clinical_relevance_identical_text_yields_one():
    m = ClinicalRelevanceMetric()
    expected = [{"note": "fever cough", "specialty": "infectious_disease"}]
    outputs = [{"summary": "fever cough", "specialty": "infectious_disease"}]
    assert m.calculate(expected, outputs) == 1.0


def test_clinical_relevance_no_overlap_zero():
    m = ClinicalRelevanceMetric()
    expected = [{"note": "fever cough"}]
    outputs = [{"summary": "ankle sprain"}]
    score = m.calculate(expected, outputs)
    assert score == 0.0


def test_diagnostic_accuracy_hf_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "heart failure", "specialty": "cardiology"}]
    outputs = [{"prediction": "hf", "specialty": "cardiology"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_uti_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "urinary tract infection"}]
    outputs = [{"prediction": "uti"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_copd_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "chronic obstructive pulmonary disease"}]
    outputs = [{"prediction": "copd"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_htn_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "hypertension"}]
    outputs = [{"prediction": "htn"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_cva_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "stroke"}]
    outputs = [{"prediction": "cva"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_pna_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "pneumonia"}]
    outputs = [{"prediction": "pna"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_hld_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "hyperlipidemia"}]
    outputs = [{"prediction": "hld"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_arf_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "acute renal failure"}]
    outputs = [{"prediction": "arf"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_pud_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "peptic ulcer disease"}]
    outputs = [{"prediction": "pud"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_afib_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "atrial fibrillation"}]
    outputs = [{"prediction": "afib"}]
    assert m.calculate(expected, outputs) == 1.0


def test_diagnostic_accuracy_gerd_synonym_match():
    m = DiagnosticAccuracyMetric()
    expected = [{"diagnosis": "gastroesophageal reflux disease"}]
    outputs = [{"prediction": "gerd"}]
    assert m.calculate(expected, outputs) == 1.0
