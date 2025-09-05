# Metrics Testing Coverage and How to Run

This document summarizes the unit/property/performance tests added for metric implementations in `bench/evaluation/metrics/` and how to run them.

## Implemented Tests

- **Malformed/Unexpected Inputs**
  - File: `tests/test_metrics_malformed_inputs.py`
  - Covers resilience to missing keys, `None`, non-string values, and extras for:
    - `ClinicalAccuracyMetric`
    - `ReasoningQualityMetric`
    - `DiagnosticAccuracyMetric`
    - `ClinicalRelevanceMetric`
  - Also asserts list shape/type validation via `Metric.validate_inputs()`.

- **Property-Based Tests (Hypothesis)**
  - File: `tests/test_metrics_property_based.py`
  - Generates randomized inputs of equal lengths; checks:
    - Scores are floats in `[0, 1]`
    - Determinism for identical inputs (ReasoningQuality)

- **Known-Values / Deterministic Scenarios**
  - File: `tests/test_metrics_known_values.py`
  - Asserts exact scores for simple deterministic cases (e.g., identical texts -> 1.0, no overlap -> 0.0).

- **Performance Benchmarks (pytest-benchmark)**
  - File: `tests/test_metrics_performance_benchmark.py`
  - Benchmarks runtime on ~1000-item batches for each metric and asserts valid score ranges.

## How to Run

- Run only new tests quickly:
```bash
pytest -q tests/test_metrics_malformed_inputs.py \
          tests/test_metrics_property_based.py \
          tests/test_metrics_known_values.py \
          tests/test_metrics_performance_benchmark.py
```

- Run benchmarks with more iterations:
```bash
pytest tests/test_metrics_performance_benchmark.py --benchmark-min-time=0.1
```

See also: category mapping and combined scoring — [metrics/metric_categories.md](metrics/metric_categories.md) and [api/cli.md#combined-score-via-cli-typer](api/cli.md#combined-score-via-cli-typer).

## Notes

- `DiagnosticAccuracyMetric` now normalizes the label `"mi"` to `"myocardial infarction"` to align synonyms before comparison.
- All tests keep metrics dependency-light and validate score normalization `[0, 1]`.
