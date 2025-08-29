# Extension Guide

How to extend MedAISure with custom tasks, metrics, and models.

## Add a Custom Task

1) Create `your_task.yaml` in your `tasks_dir` with fields used by `MedicalTask`:
```yaml
name: "My Clinical NLI"
description: "NLI over clinical text"
task_type: qa
metrics: [accuracy, clinical_correctness]
dataset:
  - text: "Patient denies chest pain."
    hypothesis: "The patient has chest pain."
    label: contradiction
```
2) Load by id (file stem) or path:
```python
from bench.evaluation import EvaluationHarness
h = EvaluationHarness(tasks_dir="bench/tasks")
report = h.evaluate(model_id="textattack/bert-base-uncased-MNLI", task_ids=["your_task"], model_type="huggingface")
```

## Register a Custom Metric

```python
from bench.evaluation.metric_calculator import MetricCalculator

def my_metric(y_true, y_pred, **kwargs):
    # return a float, or (float, metadata_dict)
    return float(sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true))

mc = MetricCalculator()
mc.register_metric("my_metric", my_metric)
```
Use it by including `my_metric` in your task's `metrics:` or by passing `metric_names=[...]` to `calculate_metrics()` manually.

## Use a Local Model

Your local module should expose a loader (default name: `load_model`):
```python
# mypkg/mylocal.py
class MyModel:
    def __call__(self, batch, **kwargs):
        # return list[dict] with keys like "label" or "prediction"
        return [{"label": "entailment", "score": 0.9} for _ in batch]

def load_model(model_path, **kwargs):
    return MyModel()
```
Load via `ModelRunner` (automatically used by `EvaluationHarness.evaluate()`):
```python
report = h.evaluate(
    model_id="my_local_model",
    task_ids=["your_task"],
    model_type="local",
    model_path="/path/to/artifacts",
    module_path="mypkg.mylocal",
)
```

## Use an API Model

Provide `endpoint` and `api_key`. Predictions should be a list of dicts with `label`/`score` or `prediction/text/summary`.
```python
report = h.evaluate(
    model_id="my_api",
    task_ids=["your_task"],
    model_type="api",
    endpoint="https://api.example.com/predict",
    api_key=os.environ["MY_API_KEY"],
)
```

See `bench/examples/` for runnable samples.
