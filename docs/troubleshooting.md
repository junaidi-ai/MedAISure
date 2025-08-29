# Troubleshooting

## Common Issues

- Missing `transformers` or `torch` when using Hugging Face
  - Install: `pip install transformers torch`

- `rouge-score not available` in logs
  - Install: `pip install rouge-score`

- Model outputs length != inputs length
  - Ensure your model returns one prediction dict per input example.

- Metrics are NaN
  - Check that reference fields exist in your task dataset (`label`, `answer`, `summary`, `note`, `rationale`)
  - Make sure prediction dicts include a compatible field (`label`, `prediction`, `text`, `summary`)

- Local model loader errors
  - Provide `module_path` and ensure it exposes a callable loader (default `load_model(model_path, **kwargs)`)

- API request failures
  - Verify `endpoint`, `api_key`, and network access
  - Tune `timeout`, `max_retries`, `backoff_factor`

## Debug Tips

- Set `log_level="DEBUG"` in `EvaluationHarness`
- Use `use_cache=False` to re-run fresh predictions
- Inspect cached files in `<cache_dir>/<run_id>_<task_id>.json`
- Enable `on_metrics` callback to stream metrics after each task
