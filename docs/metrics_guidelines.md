# Metrics Guidelines

- **Centralized synonyms**: Use the shared diagnosis label map `LABEL_SYNONYMS` for clinical abbreviations and common variants.
  - Location: `bench/evaluation/metrics/clinical.py` (exported via `bench.evaluation.metrics`)
  - Import example:
    ```python
    from bench.evaluation.metrics import LABEL_SYNONYMS
    # normalized = LABEL_SYNONYMS.get(_normalize_text(label), _normalize_text(label))
    ```
- **Why**: Ensures consistent normalization across metrics (e.g., `mi → myocardial infarction`, `copd → chronic obstructive pulmonary disease`, `htn → hypertension`).
- **Where to use**:
  - Anywhere you compare predicted/expected diagnoses or labels.
  - Prefer mapping after lowercasing/basic punctuation cleanup (use `_normalize_text()` if available).
- **Extending the map**:
  - Add new entries in `bench/evaluation/metrics/clinical.py` under the `LABEL_SYNONYMS` dict.
  - Keep entries lowercased, punctuation-free keys and values.
  - Group additions by domain (cardiology, pulmonary, etc.) with comments.
- **Testing requirements**:
  - Add known-value tests in `tests/test_metrics_known_values.py` for each new synonym, asserting a 1.0 match for the normalized pair.
  - Example:
    ```python
    def test_diagnostic_accuracy_htn_synonym_match():
        m = DiagnosticAccuracyMetric()
        expected = [{"diagnosis": "hypertension"}]
        outputs = [{"prediction": "htn"}]
        assert m.calculate(expected, outputs) == 1.0
    ```
- **Coordination with other metrics**:
  - If a metric relies on diagnosis-specific rules, consider normalizing inputs via `LABEL_SYNONYMS` before rule checks to reduce duplication.
  - Avoid changing existing lexicon logic unless tests are added to cover behavior changes.
- **Contribution tip**:
  - When proposing new synonyms, include short clinical justification or references if ambiguity is possible.

> See also: category mapping and combined scoring
>
> - Metric Categories: [docs/metrics/metric_categories.md](metrics/metric_categories.md)
> - CLI combined score usage: [docs/api/cli.md#combined-score-via-cli-typer](api/cli.md#combined-score-via-cli-typer)
