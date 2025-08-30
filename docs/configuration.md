- **MEDAISURE_NO_RICH**
  - Purpose: Disable Rich live console rendering during tests/CI, and force plain stdout output for both text and JSON from the Typer CLI in `bench/cli_typer.py`.
  - Values: `"1"` (enabled) or unset/other (disabled).
  - Behavior when set to `1`:
    - Rich status/progress bars are not created (`_status()` becomes no-op).
    - `_print()` writes plain text to `sys.stdout` (captured by pytest/CliRunner).
    - `_print_json()` always writes raw JSON to `sys.stdout` (no ANSI styling), ensuring robust parsing in tests.
    - Python logging is disabled via `logging.disable(logging.CRITICAL)` to avoid noisy writes to captured streams.
  - Recommended usage:
    - Local runs of the CLI can omit it for nicer output.
    - Enable for pytest and CI runs to avoid flaky captures: `export MEDAISURE_NO_RICH=1`.
  - CI: The default GitHub Actions workflow sets this for the test job (see `.github/workflows/tests.yml`).

# Configuration Reference

Key parameters and where they apply.

## Environment Variables
- `MEDAISURE_NO_RICH`: Disable Rich live console rendering during tests/CI, and force plain stdout output for both text and JSON from the Typer CLI in `bench/cli_typer.py`.

## EvaluationHarness
- `tasks_dir`: where task YAML/JSON files are discovered
- `results_dir`: where `<run_id>.json` reports and exports are written
- `cache_dir`: if set, caches per-task predictions for reuse
- `log_level`: python logging level string
- callbacks: `on_task_start`, `on_task_end`, `on_progress`, `on_error`, `on_metrics`

## ModelRunner (Hugging Face)
- `hf_task`: `text-classification` (default), `summarization` (alias of `text2text-generation`), `text-generation`
- `model_kwargs`: passed to `AutoModel.from_pretrained`
- `tokenizer_kwargs`: passed to `AutoTokenizer.from_pretrained`
- `pipeline_kwargs`: passed to `transformers.pipeline`
- `device`: -1 (CPU) or CUDA index
- `num_labels`: classification heads

## ModelRunner (Local)
- `model_path`: arbitrary path to your weights/artifacts
- `module_path`: import path to your loader module
- `load_func`: optional function name (default `load_model`)

## ModelRunner (API)
- `endpoint`: POST URL
- `api_key`: bearer token used by `_call_api_model`
- optional: `headers`, `timeout`, `max_retries`, `backoff_factor`

## MetricCalculator
- `register_metric(name, fn, **defaults)`
- `calculate_metrics(task_id, predictions, references, metric_names=None, **metric_kwargs)`
  - `metric_kwargs` can be a mapping keyed by metric name for per-metric overrides

## ResultAggregator
- `export_report_json|csv|markdown|html(run_id, output_path)`
- `aggregate_statistics(run_id, metrics=None, percentiles=None, tasks=None)`
- `filter_and_sort_tasks(run_id, tasks=None, metrics=None, sort_by=None, descending=True)`
- `compare_runs(run_a, run_b, metrics=None, relative=False)`
