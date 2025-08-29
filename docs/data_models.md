# MedAISure Benchmark Data Models

This document provides detailed documentation for the core data models used in the MedAISure Benchmark system.

## Table of Contents

- [MedicalTask](#medicaltask)
- [EvaluationResult](#evaluationresult)
- [BenchmarkReport](#benchmarkreport)
- [Usage Examples](#usage-examples)
- [Serialization](#serialization)

## MedicalTask

Represents a medical task that a model needs to perform.

### Fields

- `task_id` (str): Unique identifier for the task
- `task_type` (TaskType): Type of the task (e.g., "diagnostic_reasoning", "qa")
- `description` (str): Human-readable description of the task
- `inputs` (List[Dict]): List of input examples for the task
- `expected_outputs` (List[Dict]): List of expected outputs corresponding to inputs
- `metrics` (List[str]): List of metric names to evaluate the task

### Example

```python
from bench.models import MedicalTask, TaskType

task = MedicalTask(
    task_id="task_123",
    task_type=TaskType.QA,
    description="Answer medical questions",
    inputs=[{"question": "What are the symptoms of COVID-19?"}],
    expected_outputs=[{"answer": "Common symptoms include fever, cough, and fatigue."}],
    metrics=["accuracy", "f1_score"]
)
```

## EvaluationResult

Represents the result of evaluating a model on a specific task.

### Fields

- `model_id` (str): Identifier for the evaluated model
- `task_id` (str): Identifier of the task being evaluated
- `inputs` (List[Dict]): Inputs used for evaluation
- `model_outputs` (List[Dict]): Model's outputs for the given inputs
- `metrics_results` (Dict[str, float]): Evaluation metrics and their values
- `metadata` (Dict[str, Any]): Additional metadata about the evaluation
- `timestamp` (datetime): When the evaluation was performed (auto-generated)

### Example

```python
from datetime import datetime, timezone
from bench.models import EvaluationResult

result = EvaluationResult(
    model_id="gpt-4",
    task_id="task_123",
    inputs=[{"question": "What are the symptoms of COVID-19?"}],
    model_outputs=[{"answer": "Symptoms include fever, cough, and fatigue."}],
    metrics_results={"accuracy": 0.9, "f1_score": 0.85},
    metadata={"model_version": "1.0.0"},
    timestamp=datetime.now(timezone.utc)
)
```

## BenchmarkReport

Aggregates evaluation results across multiple tasks for a model.

### Fields

- `model_id` (str): Identifier for the evaluated model
- `timestamp` (datetime): When the benchmark was run (auto-generated)
- `overall_scores` (Dict[str, float]): Aggregated scores across all tasks
- `task_scores` (Dict[str, Dict[str, float]]): Scores for each task
- `detailed_results` (List[EvaluationResult]): Individual evaluation results
- `metadata` (Dict[str, Any]): Additional metadata about the benchmark

### Methods

- `add_evaluation_result(result: EvaluationResult)`: Add a new evaluation result
- `to_file(file_path: Union[str, Path])`: Save the report to a JSON file
- `from_file(file_path: Union[str, Path]) -> 'BenchmarkReport'`: Load a report from a JSON file

### Example

```python
from pathlib import Path
from bench.models import BenchmarkReport

# Create a new report
report = BenchmarkReport(
    model_id="gpt-4",
    overall_scores={"accuracy": 0.85, "f1_score": 0.8},
    task_scores={
        "task_1": {"accuracy": 0.9, "f1_score": 0.85},
        "task_2": {"accuracy": 0.8, "f1_score": 0.75}
    },
    metadata={"run_id": "run_123"}
)

# Add evaluation results
report.add_evaluation_result(result1)
report.add_evaluation_result(result2)

# Save to file
report.to_file("benchmark_results.json")

# Load from file
loaded_report = BenchmarkReport.from_file("benchmark_results.json")
```

## Usage Examples

### Creating a Medical Task

```python
from bench.models import MedicalTask, TaskType

task = MedicalTask(
    task_id="diagnosis_001",
    task_type=TaskType.DIAGNOSTIC_REASONING,
    description="Diagnose the most likely condition based on symptoms",
    inputs=[
        {
            "symptoms": ["fever", "cough", "shortness of breath"],
            "age": 45,
            "gender": "M"
        }
    ],
    expected_outputs=[
        {
            "diagnosis": "pneumonia",
            "confidence": 0.92,
            "differential_diagnosis": ["influenza", "bronchitis"]
        }
    ],
    metrics=["accuracy", "precision", "recall", "f1_score"]
)
```

### Running an Evaluation

```python
from bench.models import EvaluationResult

# After running the model on the task
model_outputs = [
    {
        "diagnosis": "pneumonia",
        "confidence": 0.89,
        "differential_diagnosis": ["influenza", "bronchitis", "asthma"]
    }
]

# Calculate metrics (simplified example)
metrics = {
    "accuracy": 1.0,  # Correct diagnosis
    "precision": 0.9,
    "recall": 0.85,
    "f1_score": 0.87
}

# Create evaluation result
eval_result = EvaluationResult(
    model_id="our-model-1.0",
    task_id=task.task_id,
    inputs=task.inputs,
    model_outputs=model_outputs,
    metrics_results=metrics,
    metadata={
        "model_version": "1.0.0",
        "evaluation_time_seconds": 2.5
    }
)
```

## Serialization

All models support JSON serialization and deserialization:

```python
# To JSON string
json_str = task.model_dump_json(indent=2)

# From JSON string
loaded_task = MedicalTask.model_validate_json(json_str)

# To dictionary
data = task.model_dump()

# From dictionary
loaded_task = MedicalTask.model_validate(data)
```

## Validation

Models include built-in validation:

```python
from pydantic import ValidationError

try:
    invalid_task = MedicalTask(
        task_id="",  # Empty string not allowed
        task_type="invalid_type",  # Not a valid TaskType
        inputs=[],  # Cannot be empty
        expected_outputs=[{}],
        metrics=[""]  # Empty metric name not allowed
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Best Practices

1. Always use the provided model classes instead of raw dictionaries
2. Take advantage of built-in validation
3. Use type hints for better IDE support and code clarity
4. Store and share evaluation results using the serialization methods
5. Include relevant metadata for traceability

## Task-Type Schema Requirements

The evaluation framework applies minimal default schemas per `TaskType` when tasks do not provide explicit `input_schema`/`output_schema`. These defaults are enforced during task load and evaluation. Required keys are:

- **QA**
  - Inputs require: `question`
  - Outputs require: `answer`

- **Summarization**
  - Inputs require: `document`
  - Outputs require: `summary`

- **Diagnostic Reasoning**
  - Inputs require: `symptoms`
  - Outputs require: `diagnosis`

- **Communication**
  - Inputs require: `prompt`
  - Outputs require: `response`

Notes:
- These defaults are defined in `bench/evaluation/validators.py` under `DEFAULT_SCHEMAS` and are used by `ensure_task_schemas()`.
- Inline datasets may use either flat rows or nested form `{ "input": { ... }, "output": { ... } }`. See `validate_task_dataset()` for details.
- Strict validation mode in the harness will raise on schema violations; non-strict mode attaches validation errors to result metadata.

## Task-Type Examples

Below are concise, runnable examples for each built-in task type.

### QA

```python
from bench.models.task_types import MedicalQATask

task = MedicalQATask("qa-demo")
task.dataset = [
    {"input": {"question": "What is BP?"}, "output": {"answer": "blood pressure"}}
]
metrics = task.evaluate([{ "answer": "blood pressure" }])  # {"accuracy": 1.0, "clinical_correctness": 1.0}
```

### Diagnostic Reasoning

```python
from bench.models.task_types import DiagnosticReasoningTask

task = DiagnosticReasoningTask("dx-demo")
task.dataset = [
    {"input": {"case": "60M chest pain"}, "output": {"diagnosis": "ACS"}}
]
metrics = task.evaluate([{ "diagnosis": "ACS", "explanation": "because ECG shows ST elevation" }])
```

### Summarization

```python
from bench.models.task_types import ClinicalSummarizationTask

task = ClinicalSummarizationTask("sum-demo")
task.dataset = [
    {"input": {"document": "Patient with HTN and DM."}, "output": {"summary": "HTN, DM."}}
]
metrics = task.evaluate([{ "summary": "HTN, DM." }])
```

## Creating Custom Task Instances

You can define your own tasks using `MedicalTask` (YAML/JSON) and validate/load them via the loader/validators.

### Minimal YAML Example

```yaml
schema_version: 1
task_id: qa-custom
task_type: qa
description: Answer short questions
inputs:
  - { question: "What is HR?" }
expected_outputs:
  - { answer: "heart rate" }
metrics: [accuracy]
input_schema:
  required: [question]
output_schema:
  required: [answer]
dataset:
  - input:  { question: "What is BP?" }
    output: { answer: "blood pressure" }
```

### Load and Validate

```python
from bench.models.medical_task import MedicalTask
from bench.evaluation.validators import validate_task_dataset

task = MedicalTask.from_file("qa-custom.yaml")
validate_task_dataset(task)  # raises if required keys missing
```

See also:
- `bench/evaluation/task_loader.py` for loading tasks from files/URLs
- `bench/evaluation/model_runner.py` for running models on tasks

## Task-Specific Metrics

This framework ships with lightweight, task-specific metrics implemented in the task classes under `bench/models/task_types.py`. These are intentionally simple placeholders and should be replaced or extended for production use.

- **QA (`MedicalQATask`)**
  - `accuracy`: case-insensitive exact match of `answer`
  - `clinical_correctness`: proxy equal to `accuracy`

- **Diagnostic Reasoning (`DiagnosticReasoningTask`)**
  - `diagnostic_accuracy`: case-insensitive exact match of `diagnosis`
  - `reasoning_quality`: heuristic based on presence of explanation cues (e.g., "because", "due to") in `explanation`/`rationale`

- **Summarization (`ClinicalSummarizationTask`)**
  - `rouge_l`: unigram-overlap proxy (not true ROUGE-L)
  - `clinical_relevance`: overlap ratio with a small set of medical keywords
  - `factual_consistency`: penalizes hallucinated numbers not present in the reference

For full implementations, consider integrating standard NLP metrics (e.g., official ROUGE, BERTScore) and clinical factuality checks.

## End-to-End Usage

To see how tasks are loaded and models are executed over them, refer to:

- `bench/evaluation/task_loader.py`
  - Key methods: `TaskLoader.load_task()`, `TaskLoader.load_tasks()`, `TaskLoader.list_available_tasks()`
  - Supports loading by ID, local path, or HTTP(S) URL with validation

- `bench/evaluation/model_runner.py`
  - Key methods: `ModelRunner.load_model()`, `ModelRunner.run_model()` / `run_model_async()`, `ModelRunner.unload_model()`
  - Supports HuggingFace pipelines, local Python module models, and API-based models

You can combine these to evaluate a model end-to-end:

```python
from bench.evaluation.task_loader import TaskLoader
from bench.evaluation.model_runner import ModelRunner

loader = TaskLoader()
task = loader.load_task("bench/tasks/clinical_summarization_basic.yaml")

runner = ModelRunner()
runner.load_model({"type": "local", "path": "tests/fixtures/simple_local_model.py", "callable": "predict"})

outputs = runner.run_model(task.inputs)
metrics = task.evaluate(outputs)
```

## Task Registry

For dynamic registration, discovery, and filtered listing of tasks, use `bench/evaluation/task_registry.py`.

- `TaskRegistry.register()` / `register_from_file()` / `register_from_url()`
- `TaskRegistry.get()` to retrieve by ID
- `TaskRegistry.discover()` to scan a tasks directory
- `TaskRegistry.list_available(task_type=..., min_examples=..., has_metrics=...)` for simple filtering

Example:

```python
from bench.evaluation.task_registry import TaskRegistry
from bench.models.medical_task import TaskType

reg = TaskRegistry(tasks_dir="bench/tasks")
reg.discover()

# Filter to QA tasks that declare metrics and have >= 1 example
rows = reg.list_available(task_type=TaskType.QA, min_examples=1, has_metrics=True)

# Load a specific task (from discovery or previously registered)
task = reg.get(rows[0].task_id)
```
