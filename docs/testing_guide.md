# MedAISure Testing Guide

This document summarizes the test coverage added for the core data models under `bench/models/` and how to run them.

## Scope Covered

- MedicalTask (`bench/models/medical_task.py`)
- EvaluationResult (`bench/models/evaluation_result.py`)
- BenchmarkReport (`bench/models/benchmark_report.py`)

## Test Categories

- Property-based tests: `tests/test_models_property_based_test.py`
  - Randomized generation validates model constraints and JSON round-trips.
- Serialization round-trip tests: `tests/test_models_serialization_roundtrip_test.py`
  - Validates `to_dict`/`from_dict`, `to_json`/`from_json`, `to_yaml`/`from_yaml`, and file I/O helpers.
  - Verifies CSV helpers where applicable.
- Edge case tests: `tests/test_models_edge_cases_test.py`
  - Invalid inputs, boundary conditions, timezone normalization, metric validation strictness.
- Performance smoke tests: `tests/test_models_performance_test.py`
  - Aggregation throughput for `BenchmarkReport.add_evaluation_result`.
  - JSON/YAML serialization throughput for `BenchmarkReport`.
- Integration tests: `tests/test_models_integration_test.py`
  - `EvaluationResult.validate_against_task()` with `MedicalTask` schemas.
  - `BenchmarkReport.validate_against_tasks()` and aggregate correctness across tasks.

## Running Tests

- Fast model-focused subset:

```bash
pytest -q tests/test_models_property_based_test.py \
         tests/test_models_serialization_roundtrip_test.py \
         tests/test_models_edge_cases_test.py \
         tests/test_models_integration_test.py
```

- Include performance (benchmarks):

```bash
pytest -q tests/test_models_performance_test.py --benchmark-only
```

## Dev Dependencies

- Property-based tests require `hypothesis`. Ensure you have dev deps installed:

```bash
pip install -r requirements-dev.txt
```

## Coverage

- Coverage is enabled via `pytest-cov` in `requirements-dev.txt`.

Generate an HTML coverage report for the entire suite:

```bash
pytest --cov=bench --cov-report=html
open htmlcov/index.html  # or use a file viewer
```

## Notes

- `EvaluationResult.metrics_results` now rejects non-numeric, NaN, and Infinity values.
- Timestamp fields are normalized to timezone-aware UTC; tests compare with `exclude={"timestamp"}` when appropriate.
