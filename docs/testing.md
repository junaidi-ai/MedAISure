# Testing Guide

This project includes a comprehensive testing suite for dataset connectors and related utilities.

## Running Tests

- Local: `pytest -q`
- With coverage (default via pytest.ini): `pytest -v`
- Generate HTML coverage report: opens automatically at `htmlcov/index.html`.

CI runs tests on each push/PR and uploads the HTML coverage report as an artifact named `coverage-html`.

## Test Structure

- `tests/test_data_connectors.py`: Unit tests for `JSONDataset` and `CSVDataset` load paths (plain, gzip, zip), validation, and batching.
- `tests/test_connector_encrypted_local_datasets.py`: Ensures encrypted JSON/CSV files decrypt end-to-end when an `encryption_key` is provided.
- `tests/test_medical_connectors.py`: Unit tests for `MIMICConnector` (SQLite) and `PubMedConnector` with mocked HTTP responses.
- `tests/test_mimic_performance_and_redaction.py`: Integration/performance test for `MIMICConnector` including PHI redaction and cache acceleration check.
- `tests/test_security_handler.py`: Security-focused tests for `SecureDataHandler` including algorithms, include/exclude fields, anonymization, audit logging, key rotation, and batch helpers.
- `tests/test_encrypted_negative_cases.py`: Negative-path tests for encrypted inputs, including corrupted ciphertexts, wrong keys, mixed partially encrypted rows, and truncated base64 tokens.
- `tests/test_large_batched_reads.py`: Large-scale batched read tests for JSON/CSV with plain, gzip, and zip variants; parameterized batch sizes and CI-safe performance bounds.

Additional end-to-end and performance tests exist across the suite (models, metrics, harness).

## Fixtures and Factories

Defined in `tests/conftest.py`:

- `sample_json_file`, `sample_csv_file`: Create simple local datasets.
- `encrypted_json_file`, `encrypted_csv_file`: Create encrypted datasets using `SecureDataHandler("test-pass")` for end-to-end decryption tests.
- Existing `temp_tasks_dir`, `example_task_definition` support task-related tests.

## External Services and Mocking

- PubMed API is mocked in unit tests via `monkeypatch`.
- A live smoke test exists in `tests/test_pubmed_live.py` guarded by `RUN_PUBMED_LIVE=1` and optional `NCBI_API_KEY`.
- Databases are simulated using temporary SQLite files created in tests.

## Security Considerations

- PHI redaction is validated in both connectors and security handler tests.
- Encryption/decryption is tested across multiple algorithms; audit logging and key rotation are verified.

## Performance Benchmarks

- Lightweight performance checks exist for connectors and core components.
- Run `pytest -k perf -q` to filter performance-oriented tests.
- Large batched read tests enforce modest time upper bounds tuned for CI to catch regressions without flakiness.

## Coverage

- Pytest is configured to collect coverage for `bench/*` and produce terminal and HTML reports.
- CI uploads `htmlcov` so you can review annotated coverage on PRs.

## Tips

- Use `-k <expr>` to focus on a subset, e.g., `pytest -k medical -q`.
- Use `-m integration` to include integration tests; mark heavy tests with `@pytest.mark.integration`.
- CI splits unit vs. integration suites: the main job excludes `integration`, while a separate job runs only `integration` tests.
