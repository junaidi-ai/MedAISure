# MEDDSAI Benchmark Data Models

This document provides detailed documentation for the core data models used in the MEDDSAI Benchmark system.

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
