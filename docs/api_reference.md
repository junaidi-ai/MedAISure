# API Reference (concise)

References for primary public classes. Read the code for full details.

## EvaluationHarness (`bench/evaluation/harness.py`)
- `__init__(tasks_dir, results_dir, cache_dir=None, log_level="INFO", callbacks...)`
- `evaluate(model_id, task_ids, model_type="huggingface", batch_size=8, use_cache=True, save_results=True, **model_kwargs) -> BenchmarkReport`
- `list_available_tasks() -> List[dict]`
- `get_task_info(task_id) -> dict`

Notes:
- `evaluate()` loads the model via `ModelRunner.load_model()` then iterates tasks.
- Metrics are computed via `MetricCalculator.calculate_metrics()`.
- Results aggregated via `ResultAggregator.add_evaluation_result()` and returned as `BenchmarkReport`.

Example – basic run:
```python
from bench.evaluation import EvaluationHarness
h = EvaluationHarness(tasks_dir="bench/tasks", results_dir="bench/results")
tasks = h.list_available_tasks()
report = h.evaluate(
    model_id="textattack/bert-base-uncased-MNLI",
    task_ids=[tasks[0]["task_id"]],
    model_type="huggingface",
    batch_size=4,
)
print(report.overall_scores)
```

Example – get task info:
```python
info = h.get_task_info(tasks[0]["task_id"])
print(info["name"], info.get("metrics"))
```

## TaskLoader (`bench/evaluation/task_loader.py`)
- `load_task(task_id: str) -> MedicalTask` (task_id can be id, local file path, or HTTP(S) URL)
- `load_tasks(task_ids: List[str]) -> Dict[str, MedicalTask>`
- `discover_tasks() -> Dict[str, str]`
- `list_available_tasks() -> List[dict]`

Examples:
```python
from bench.evaluation.task_loader import TaskLoader
tl = TaskLoader(tasks_dir="bench/tasks")
task = tl.load_task("clinical_summarization_discharge")
all_tasks = tl.list_available_tasks()
```

## ModelRunner (`bench/evaluation/model_runner.py`)
- `load_model(model_name, model_type="local", model_path=None, **kwargs)`
  - `model_type in {"huggingface", "local", "api"}`
  - HF: supports `hf_task`, `model_kwargs`, `tokenizer_kwargs`, `pipeline_kwargs`, `device`, `num_labels`
  - Local: requires `module_path`, optional `load_func` (default `load_model`)
  - API: requires `endpoint`, `api_key`, optional `timeout`, `max_retries`, `backoff_factor`, `headers`
- `run_model(model_id, inputs, batch_size=8, **kwargs) -> List[dict]`
- `unload_model(model_name)`

Examples:
```python
from bench.evaluation.model_runner import ModelRunner
mr = ModelRunner()
# HF
mr.load_model("sshleifer/distilbart-cnn-12-6", model_type="huggingface", hf_task="summarization")
preds = mr.run_model("sshleifer/distilbart-cnn-12-6", inputs=[{"text": "note"}], batch_size=1)

# Local
mr.load_model("my_local", model_type="local", module_path="bench.examples.mypkg.mylocal")
preds = mr.run_model("my_local", inputs=[{"text": "x"}], batch_size=2)
```

Returned prediction dicts typically include `{"label", "score"}` for classification or `{"summary"|"text"|"prediction"}` for generative tasks.

## LocalModel (`bench/evaluation/model_interface.py`)

- `LocalModel(predict_fn: Optional[Callable]=None, model_path: Optional[str]=None, loader: Optional[Callable]=None, model_id: Optional[str]=None)`
  - Load local models directly from files or wrap a Python callable.
  - Auto-loaders by extension:
    - `.pkl`/`.pickle` → `pickle.load`
    - `.joblib` → `joblib.load` (optional dependency)
    - `.pt`/`.pth` → `torch.load` (optional dependency)
  - Prediction: calls the model object (if callable) or its `.predict()` method; otherwise uses `predict_fn`.
  - `metadata` includes `file_path`, `ext`, `file_size`, `file_mtime`, `object_class`, `object_module`, and inferred `framework`.

Examples (runnable scripts):

```bash
python bench/examples/run_localmodel_pickle.py
python bench/examples/run_localmodel_joblib.py   # requires joblib
python bench/examples/run_localmodel_torch.py    # requires torch
```

Inline usage – Pickle:
```python
from bench.evaluation.model_interface import LocalModel
lm = LocalModel(model_path=".local_models/echo.pkl")
preds = lm.predict([{"text": "hello"}, {"text": "world"}])
print(preds, lm.metadata)
```

Inline usage – Joblib:
```python
from bench.evaluation.model_interface import LocalModel
lm = LocalModel(model_path=".local_models/echo.joblib")
preds = lm.predict([{"a": 1}, {"bbb": 2}])
print(preds)
```

Inline usage – Torch:
```python
from bench.evaluation.model_interface import LocalModel
lm = LocalModel(model_path=".local_models/echo.pt")
preds = lm.predict([{"x": 0.1}, {"x": 0.9}])
print(preds)
```

## MetricCalculator (`bench/evaluation/metric_calculator.py`)
- Built-ins: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `average_precision`, `mse`, `mae`, `r2`
- Medical: `diagnostic_accuracy`, `clinical_correctness`, `reasoning_quality`, `rouge_l`, `clinical_relevance`, `factual_consistency`
- `register_metric(name, fn, **default_kwargs)`
- `calculate_metrics(task_id, predictions, references, metric_names=None, **metric_kwargs) -> Dict[str, MetricResult]`
- `aggregate_metrics(metric_results, aggregation="mean") -> Dict[str, MetricResult]`

Examples:
```python
from bench.evaluation.metric_calculator import MetricCalculator
mc = MetricCalculator()

preds = [{"label": "yes"}, {"label": "no"}]
refs = [{"label": "yes"}, {"label": "no"}]
res = mc.calculate_metrics("demo", preds, refs, metric_names=["accuracy"])

def exact_match(y_true, y_pred, **kw):
    return float(sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true))
mc.register_metric("exact_match", exact_match)
res2 = mc.calculate_metrics("demo", preds, refs, metric_names=["exact_match"])
```

## ResultAggregator (`bench/evaluation/result_aggregator.py`)
- `add_evaluation_result(EvaluationResult, run_id=None)`
- `get_report(run_id) -> BenchmarkReport`
- `export_report_json|csv|markdown|html(run_id, output)`
- `aggregate_statistics(run_id, metrics=None, percentiles=None, tasks=None)`
- `filter_and_sort_tasks(run_id, tasks=None, metrics=None, sort_by=None, descending=True)`
- `compare_runs(run_a, run_b, metrics=None, relative=False)`

Examples:
```python
from bench.evaluation.result_aggregator import ResultAggregator
from bench.models import EvaluationResult

ra = ResultAggregator(output_dir="bench/results")
er = EvaluationResult(
    model_id="demo",
    task_id="task1",
    inputs=[{"text": "a"}],
    model_outputs=[{"label": "y"}],
    metrics_results={},
)
ra.add_evaluation_result(er, run_id="run-1")
report = ra.get_report("run-1")
ra.export_report_json("run-1", output_path="bench/results/run-1.json")
```

## Data Models (`bench/models/`)
- `MedicalTask`: fields include `task_id`, `task_type`, `name`, `description`, `inputs`, `expected_outputs`, `metrics`, `input_schema`, `output_schema`, `dataset`
- `EvaluationResult`: `model_id`, `task_id`, `inputs`, `model_outputs`, `metrics_results`, `metadata`, `timestamp`
- `BenchmarkReport`: overall/task scores + `detailed_results`, JSON save/load helpers

### Advanced Serialization (Models)

- `schema_version`: int, default `1` on all models for forward compatibility

- Partial serialization (all models):
  - `to_dict(include=None, exclude=None)`
  - `to_json(indent=None, include=None, exclude=None)`

- YAML support (all models):
  - `to_yaml()` → YAML string
  - `from_yaml(text: str)` → instance

- File I/O and conversion (all models):
  - `save(path, format=None)` → format inferred from extension (`.json`, `.yaml`, `.yml`) or by `format`
  - `from_file(path)` → loads JSON/YAML by extension
  - `convert(to)` → returns string in `json` or `yaml`

- CSV helpers:
  - `EvaluationResult`: `inputs_to_csv()`, `outputs_to_csv()`, `from_inputs_csv(model_id, task_id, csv_text)`, `from_outputs_csv(model_id, task_id, csv_text)`
  - `MedicalTask`: `dataset_to_csv()`
  - `BenchmarkReport`: `overall_scores_to_csv()`, `task_scores_to_csv()`

Examples:
```python
from bench.models import MedicalTask, EvaluationResult, BenchmarkReport

# YAML round-trip
task_yaml = task.to_yaml()
task2 = MedicalTask.from_yaml(task_yaml)

# Partial JSON (only overall_scores)
report_json = report.to_json(indent=2, include={"overall_scores": True})

# Save/Load
result.save("res.yaml")
res2 = EvaluationResult.from_file("res.yaml")

# CSV exports
csv_inputs = res2.inputs_to_csv()
csv_overall = report.overall_scores_to_csv()
```
