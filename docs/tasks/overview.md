# Tasks Overview

Overview of task types and how to add custom tasks.

- Medical QA
- Diagnostic Reasoning
- Clinical Summarization

See also: `bench/examples/` and `docs/usage.md`.

## Task schema (MedicalTask)

Core schema lives in `bench/models/medical_task.py` with enum `TaskType` and model `MedicalTask`.

TaskType values
```python
from bench.models.medical_task import TaskType

list(TaskType)
# [TaskType.DIAGNOSTIC_REASONING, TaskType.QA, TaskType.SUMMARIZATION, TaskType.COMMUNICATION]
```

MedicalTask fields (essentials)
```python
from bench.models.medical_task import MedicalTask, TaskType

t = MedicalTask(
    task_id="qa-demo",
    task_type=TaskType.QA,
    description="Answer short medical questions",
    inputs=[{"question": "What is BP?"}],
    expected_outputs=[{"answer": "blood pressure"}],
    metrics=["clinical_accuracy"],
    input_schema={"required": ["question"]},
    output_schema={"required": ["answer"]},
)
```

Validation rules (enforced by Pydantic validators)
- inputs: non-empty list
- metrics: non-empty, unique values (normalized spacing)
- expected_outputs: if provided, length must match inputs (1:1)
- input_schema/output_schema: dict with optional `required: [str, ...]`

Serialization helpers
```python
data = t.to_dict()
json_s = t.to_json(indent=2)
yaml_s = t.to_yaml()
t.save("/tmp/task.yaml")        # infers format from extension
u = MedicalTask.from_file("/tmp/task.yaml")
print(u.convert("json").splitlines()[0])  # pretty JSON string
```

Lightweight inline dataset (optional)
```python
MedicalTask(
    task_id="sum-1",
    task_type=TaskType.SUMMARIZATION,
    inputs=[{"document": "Patient note"}],
    expected_outputs=[{"summary": "Short note"}],
    metrics=["clinical_relevance"],
    input_schema={"required": ["document"]},
    output_schema={"required": ["summary"]},
    dataset=[{"input": {"document": "HPI..."}, "output": {"summary": "..."}}],
)
```

Example run
- See `bench/examples/run_local_model.py` for an end-to-end run using `EvaluationHarness` and a local model.

See also
- Python API → Task schema: api/reference.md#task-schema
