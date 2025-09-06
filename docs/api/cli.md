# CLI

Command-line usage for running evaluations and utilities. MedAISure does not ship a standalone CLI binary; use Python module invocations or short scripts.

## Common invocations

- GPU smoke test
```bash
python scripts/gpu_smoke.py
```

- Run unit tests
```bash
pytest -q
```

- Evaluate via a short Python one-liner
```bash
python - <<'PY'
from bench.evaluation.harness import EvaluationHarness

h = EvaluationHarness(tasks_dir="tasks", results_dir="results", log_level="INFO")
report = h.evaluate(
    model_id="hf-sum",
    task_ids=["medical_qa"],
    model_type="huggingface",
    batch_size=8,
    use_cache=True,
    save_results=True,
    strict_validation=False,
    report_formats=["json"],
    model_path="sshleifer/tiny-t5",
    hf_task="summarization",
    generation_kwargs={"max_new_tokens": 64},
)
print(report.overall_scores)
PY
```

## Combined score via CLI (Typer)

The Typer-based CLI supports computing a weighted combined score across categories using `--combined-weights` and `--combined-metric-name`. The weights can be provided as JSON or comma-separated `key=value` pairs. See `bench/cli_typer.py` for details.

- JSON weights:
```bash
python -m bench.cli_typer evaluate <model-id> \
  --tasks <task-id> \
  --tasks-dir tasks \
  --model-type local \
  --output-dir results \
  --format json \
  --save-results \
  --combined-weights '{"diagnostics": 0.4, "safety": 0.3, "communication": 0.2, "summarization": 0.1}' \
  --combined-metric-name combined_score
```

- Comma-separated pairs:
```bash
python -m bench.cli_typer evaluate <model-id> \
  --tasks <task-id> \
  --tasks-dir tasks \
  --model-type local \
  --output-dir results \
  --format json \
  --save-results \
  --combined-weights diagnostics=0.4,safety=0.3,communication=0.2,summarization=0.1 \
  --combined-metric-name combined_score
```

Notes:
- Weights must be non-negative and sum to 1.0 (±1e-6). Invalid inputs will be rejected with a clear error.
- When a task lacks some categories, remaining present weights are re-normalized by default so scores stay comparable.

### Category mapping overrides

You can override how raw metric names are mapped to high-level categories that feed into the combined score. This is useful when tasks define custom metric keys.

- Inline JSON:
```bash
python -m bench.cli_typer evaluate <model-id> \
  --tasks <task-id> \
  --tasks-dir tasks \
  --category-map '{"diagnostics":["accuracy","exact_match"],"summarization":["rouge_l"]}' \
  --combined-weights diagnostics=0.7,summarization=0.3 \
  --combined-metric-name combined_score
```

- JSON/YAML file:
```bash
python -m bench.cli_typer evaluate <model-id> \
  --tasks <task-id> \
  --tasks-dir tasks \
  --category-map-file .taskmaster/configs/category_map.yaml \
  --combined-weights diagnostics=0.7,summarization=0.3 \
  --combined-metric-name combined_score
```

- Config file (`BenchmarkConfig`):
Add a `category_map` section to your config (`--config-file`), and the CLI will use it unless overridden by CLI flags.

See tuned defaults and more examples in `docs/metrics/metric_categories.md`.

### Config file alternative

You can also set combined score options in a config file that the Typer CLI consumes via `--config-file`. The config maps to `bench/cli_typer.BenchmarkConfig` and supports `combined_weights` and `combined_metric_name`.

YAML:

```yaml
model_id: test-local
tasks:
  - medical_qa
combined_weights:
  diagnostics: 0.4
  safety: 0.3
  communication: 0.2
  summarization: 0.1
combined_metric_name: combined_score
```

JSON:

```json
{
  "model_id": "test-local",
  "tasks": ["medical_qa"],
  "combined_weights": {
    "diagnostics": 0.4,
    "safety": 0.3,
    "communication": 0.2,
    "summarization": 0.1
  },
  "combined_metric_name": "combined_score"
}
```

Invoke with the config file:

```bash
python -m bench.cli_typer evaluate test-local \
  --config-file config.yaml \
  --tasks-dir tasks \
  --model-type local \
  --output-dir results \
  --format json \
  --save-results
```

Resolution order: CLI flags > config file > defaults (`diagnostics=0.4`, `safety=0.3`, `communication=0.2`, `summarization=0.1`).

## Key parameters (map to EvaluationHarness.evaluate)

- `model_id` (str): identifier used to register/load the model in `ModelRunner`.
- `task_ids` (list[str]): tasks to evaluate.
- `model_type` (str): `huggingface`, `local`, or `api`.
- `batch_size` (int): batch size for inference.
- `use_cache` (bool): use cached results if available.
- `save_results` (bool): write results to `results_dir`.
- `strict_validation` (bool): raise on schema validation errors.
- `report_formats` (list[str]): optional extra outputs (e.g., `json`, `md`).
- `report_dir` (str): optional output directory for extra formats.
- `**model_kwargs`: forwarded to `ModelRunner.load_model(...)` (e.g., `model_path`, `hf_task`, `generation_kwargs`, `endpoint`, `api_key`).

## Environment variables

- `MEDAISURE_NO_RICH=1` disables the tqdm progress bar/animations during evaluation.

## Troubleshooting

- Missing dependencies: ensure requirements are installed
  ```bash
  pip install -r requirements.txt
  ```
- HF model fails to load: verify `model_path`/`revision`/`trust_remote_code`; check GPU availability and `device_map`.
- Unexpected HF text output: set appropriate `hf_task` and `generation_kwargs` (e.g., `max_new_tokens`, `do_sample=False`).
- API model errors: verify `endpoint`, auth (`api_key`/`headers`), and response shape `{ "outputs": [ ... ] }`.
- Validation errors: use `strict_validation=True` to fail fast and inspect schemas in [Tasks overview](../tasks/overview.md) and [API → Task schema](../api/reference.md#task-schema).
- Empty metrics: confirm the task defines metrics; see [Metrics overview](../metrics/overview.md) and [API → Metrics (clinical)](../api/reference.md#metrics-clinical).

- Pytest marks missing: running `pytest -c /dev/null` bypasses the project `pytest.ini`, so custom marks like `@pytest.mark.integration` may warn as unknown. Run `pytest` without `-c /dev/null` (or register the mark in a provided `pytest.ini`) to avoid the warning.
