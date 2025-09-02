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
- `MEDAISURE_HTML_OPEN_METADATA`: When set to `"1"`, HTML reports open the Metadata section by default; otherwise collapsed.
- `MEDAISURE_HTML_PREVIEW_LIMIT`: Integer. Max number of list items to show in HTML previews for `inputs`/`model_outputs` before truncation. Default `5`.

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

## BenchmarkConfig (CLI config file)

You can provide defaults for CLI runs via a JSON/YAML config file and pass it with `--config-file` to the Typer CLI `evaluate` command.

Fields (subset):

- `model_id`: string
- `tasks`: list of task IDs
- `output_dir`: path for results
- `output_format`: `json|yaml|md|csv`
- `model_type`: `huggingface|local|api`
- `batch_size`: int
- `use_cache`: bool
- `save_results`: bool
- `extra_reports`: optional list of extra export formats, e.g. `["html", "md"]`
- `report_dir`: optional path where extra reports are written (defaults to `output_dir`)
 - `html_open_metadata`: optional bool; if true, open Metadata sections by default in HTML reports
 - `html_preview_limit`: optional int; controls truncation length for inputs/outputs previews in HTML

Example YAML:

```yaml
model_id: textattack/bert-base-uncased-MNLI
tasks: [clinical_icd10_classification]
output_dir: results
output_format: json
model_type: huggingface
batch_size: 4
use_cache: true
save_results: true
extra_reports: [html, md]
report_dir: reports
html_open_metadata: true
html_preview_limit: 10
```

### Supported formats (extra reports)

- json
- md (markdown)
- html

Reference: extra reports are generated via `ReportFactory` in `bench/reports/factory.py`.

CLI usage with config file:

```bash
MEDAISURE_NO_RICH=1 python -m bench.cli_typer evaluate textattack/bert-base-uncased-MNLI \
  --tasks clinical_icd10_classification \
  --config-file config.yaml
```
