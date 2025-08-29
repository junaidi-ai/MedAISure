# Configuration Reference

Key parameters and where they apply.

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
