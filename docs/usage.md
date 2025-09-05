# Usage Guide

This guide shows how to run evaluations with the MedAISure framework using the public API in `bench/evaluation/`.

- Core class: `EvaluationHarness` in `bench/evaluation/harness.py`
- Supporting components: `TaskLoader`, `ModelRunner`, `MetricCalculator`, `ResultAggregator`

## Quick Start

```python
from bench.evaluation import EvaluationHarness

h = EvaluationHarness(
    tasks_dir="bench/tasks",
    results_dir="bench/results",
    cache_dir="bench/results/cache",
    log_level="INFO",
)

# Discover tasks and pick one
tasks = h.list_available_tasks()
print([t["task_id"] for t in tasks])

report = h.evaluate(
    model_id="textattack/bert-base-uncased-MNLI",
    task_ids=[tasks[0]["task_id"]],
    model_type="huggingface",
    model_kwargs={"num_labels": 3},
    pipeline_kwargs={"top_k": 1},
    batch_size=4,
    use_cache=False,
)
print(report.overall_scores)
```

## Running with Different Model Types

- Hugging Face: `model_type="huggingface"` with `model_kwargs`, `pipeline_kwargs` (see `ModelRunner._load_huggingface_model()`)
- Local model: `model_type="local"` with `model_path`, `module_path`, optional `load_func` (see `ModelRunner._load_local_model()`)
- API model: `model_type="api"` with `endpoint`, `api_key` and optional retry settings (see `ModelRunner._load_api_model()`)

## HuggingFace advanced options & generation quick start

When loading Hugging Face models via `ModelRunner.load_model(..., model_type="huggingface", ...)`, you can pass:

- `hf_task`: `text-classification | summarization | text-generation`
- `generation_kwargs`: forwarded during inference for generative tasks (e.g., `max_new_tokens`, `temperature`, `do_sample`, `top_p`, `top_k`)
- Advanced loading: `device_map`, `torch_dtype`, `low_cpu_mem_usage`, `revision`, `trust_remote_code`

Quick start (generation parameters):

```python
from bench.evaluation.model_runner import ModelRunner

mr = ModelRunner()
mr.load_model(
    "facebook/bart-large-cnn",
    model_type="huggingface",
    hf_task="summarization",
    generation_kwargs={"max_new_tokens": 96, "temperature": 0.7, "do_sample": True, "top_p": 0.9},
    device_map="auto",
    torch_dtype="auto",
)

inputs = [{"document": "Long clinical note ..."}]
preds = mr.run_model("facebook/bart-large-cnn", inputs, batch_size=1)
print(preds[0]["summary"])
```

See also the runnable examples:
- `bench/examples/run_hf_summarization_gen.py`
- `bench/examples/run_hf_text_generation_gen.py`

## Task Selection

`EvaluationHarness.list_available_tasks()` scans `tasks_dir` for YAML/JSON files. For advanced loading (file path or URL), use `TaskLoader.load_task()` directly.

## Results and Reports

- Per-task results: collected as `EvaluationResult` in `bench/models/evaluation_result.py`
- Aggregation: `ResultAggregator` builds a `BenchmarkReport` and supports CSV/Markdown/HTML export
- Save path: `<results_dir>/<run_id>.json`

> Tip: You can compute a weighted combined score across categories during evaluation. From the CLI, use `--combined-weights` and optionally `--combined-metric-name`. See [CLI combined score](api/cli.md#combined-score-via-cli-typer) and [Metric Categories](metrics/metric_categories.md).

### Config-based combined score (YAML/JSON)

For users who prefer config files, you can set `combined_weights` and `combined_metric_name` in a YAML or JSON config and pass it to the Typer CLI via `--config-file`. These map to `bench/cli_typer.BenchmarkConfig`.

YAML example:

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

JSON example:

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

Run with the config file:

```bash
python -m bench.cli_typer evaluate test-local \
  --config-file config.yaml \
  --tasks-dir tasks \
  --model-type local \
  --output-dir results \
  --format json \
  --save-results
```

Resolution order for combined score settings: CLI flags > config file > defaults (`diagnostics=0.4`, `safety=0.3`, `communication=0.2`, `summarization=0.1`).

Programmatic (no config file): pass directly to `EvaluationHarness.evaluate(...)`:

```python
report = h.evaluate(
    model_id="test-local",
    task_ids=["medical_qa"],
    model_type="local",
    combined_weights={
        "diagnostics": 0.4,
        "safety": 0.3,
        "communication": 0.2,
        "summarization": 0.1,
    },
    combined_metric_name="combined_score",
)
```

## Caching

If `cache_dir` is set, predictions per task are cached to JSON (`<run_id>_<task_id>.json`). Disable with `use_cache=False`.

## Callbacks

`EvaluationHarness` accepts optional callbacks:
- `on_task_start(task_id)`, `on_task_end(task_id, result)`
- `on_progress(idx, total, current_task)`
- `on_error(task_id, exception)`
- `on_metrics(task_id, metrics_dict)`

See their usage throughout `EvaluationHarness.evaluate()`.
