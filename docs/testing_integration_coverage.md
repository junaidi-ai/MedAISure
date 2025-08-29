# Integration Tests Coverage for Metrics System

This document summarizes the integration tests that validate the end-to-end metrics system in MedAISure.

## Scope Covered

- **End-to-end pipeline**
  - `EvaluationHarness.evaluate()` across tasks, including callbacks, caching, and strict validation.
  - Reference: `tests/test_integration_end_to_end_local_model_test.py`

- **Model Runner integration**
  - Model loading (local/HF mocked), batched inference, error propagation, and resource cleanup.
  - Reference: `tests/test_integration_end_to_end_local_model_test.py`

- **Metric aggregation and reports**
  - Aggregation, statistics, filtering/sorting, and exporters (JSON/CSV/Markdown/HTML).
  - References:
    - `tests/test_result_aggregator_extended_test.py`
    - `bench/evaluation/result_aggregator.py`

- **Performance smoke / large datasets**
  - Smoke performance and large dataset scenarios validated.
  - Reference: `tests/test_integration_end_to_end_local_model_test.py`

- **Human judgment comparison**
  - Custom `human_judgment` metric compared across runs using `ResultAggregator.compare_runs()`.
  - Reference: `tests/test_integration_human_judgment_and_regression_test.py::test_human_judgment_comparison_via_compare_runs`

- **Regression tests (comparisons)**
  - Absolute and relative diffs between baseline and current runs using `compare_runs(relative=True)`.
  - Reference: `tests/test_integration_human_judgment_and_regression_test.py::test_regression_detection_relative_diff`

## How to Run Tests

```bash
pytest -q
```

All tests should pass; as of the latest run: 174 passed.

## Using Run Comparisons in Practice

- **Absolute diff**
  ```python
  diff = agg.compare_runs("baseline", "current", metrics=["accuracy"], relative=False)
  print(diff["overall"]["accuracy"])  # positive => improvement
  ```

- **Relative diff**
  ```python
  diff = agg.compare_runs("baseline", "current", metrics=["accuracy"], relative=True)
  # (b - a) / (|a| + eps)
  ```

## Exported Reports

- `ResultAggregator` supports exporting a run report to multiple formats:
  - **JSON**: `export_report_json(run_id, path)`
  - **CSV**: `export_report_csv(run_id, path)`
  - **Markdown**: `export_report_markdown(run_id, path)`
  - **HTML**: `export_report_html(run_id, path)`

See `tests/test_result_aggregator_extended_test.py` for usage examples.

## Notes

- Human judgment comparisons use a dedicated metric name (`human_judgment`) so they do not interfere with other metrics.
- Tests are designed to be lightweight and avoid heavy model dependencies.
