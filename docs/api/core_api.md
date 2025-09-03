# Core API

This section documents the primary Python interfaces used to run tasks, models, and evaluations.

Key Modules
- `bench.evaluation.harness.EvaluationHarness`: orchestrates task loading, model execution, metrics, reports
- `bench.evaluation.model_runner.ModelRunner`: unified runner for HuggingFace, local modules, API models
- `bench.evaluation.metrics.*`: concrete metric implementations (e.g., `clinical.ClinicalAccuracyMetric`)
- `bench.models.medical_task.MedicalTask`: canonical task schema

Typical Flow (end-to-end evaluate)
```python
from bench.evaluation.harness import EvaluationHarness

h = EvaluationHarness(
    tasks_dir="bench/tasks",
    results_dir="bench/results",
    cache_dir="bench/results/cache",
    log_level="INFO",
)

# Evaluate a HuggingFace pipeline on one or more tasks
report = h.evaluate(
    model_id="hf-sum",
    task_ids=["clinical_summarization_basic"],
    model_type="huggingface",
    model_path="sshleifer/tiny-t5",
    hf_task="summarization",
    # Optional advanced HF settings (see ModelRunner):
    device=-1,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    low_cpu_mem_usage=True,
    generation_kwargs={"max_new_tokens": 64, "do_sample": False},
)
print(report.overall_scores)
```

Using local Python model
```python
from bench.evaluation.harness import EvaluationHarness

h = EvaluationHarness(tasks_dir="bench/tasks", results_dir="bench/results")
report = h.evaluate(
    model_id="my_local_model",
    task_ids=["some_task_id"],
    model_type="local",
    module_path="bench.examples.mypkg.mylocal",  # provides load_model()
    # optional: load_func="load_model", model_path="/path/to/weights",
    batch_size=4,
)
```

Using a simple HTTP API model
```python
from bench.evaluation.harness import EvaluationHarness

h = EvaluationHarness(tasks_dir="bench/tasks", results_dir="bench/results")
report = h.evaluate(
    model_id="api-demo",
    task_ids=["medical_qa_basic"],
    model_type="api",
    endpoint="https://api.example/v1/predict",
    api_key="sk-...",
    timeout=30.0,
    max_retries=1,
)
```

Notes
- Generation kwargs pass-through for HF pipelines is supported via `generation_kwargs` in `ModelRunner`.
- Advanced HF loading options supported: `device_map`, `torch_dtype`, `trust_remote_code`, `low_cpu_mem_usage`, `revision`.
- Metadata extraction (e.g., config fields, parameter count, dtype, device) is attempted when available.

See also: API reference (mkdocstrings)
- Python API → Evaluation Harness: api/reference.md#evaluation-harness
- Python API → Model Runner: api/reference.md#model-runner
- Python API → Metrics (clinical): api/reference.md#metrics-clinical
- Python API → Task schema: api/reference.md#task-schema

Source modules:
- `bench/evaluation/harness.py`
- `bench/evaluation/model_runner.py`
- `bench/evaluation/metrics/clinical.py`
- `bench/models/medical_task.py`

Examples
- End-to-end local model run: `bench/examples/run_local_model.py`
