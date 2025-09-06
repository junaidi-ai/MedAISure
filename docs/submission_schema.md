# Leaderboard Submission Schema

This document describes the JSON schema used for MedAISure leaderboard submissions and shows concrete examples.

The exporter builds this schema from a `BenchmarkReport` and validates it strictly using `jsonschema`.

## Top-level structure

- `schema_version` (int, >=1)
- `run_id` (string)
- `model_id` (string)
- `created_at` (ISO 8601 string)
- `submissions` (array of per-task objects)

## Per-task structure

Each entry in `submissions` has:

- `task_id` (string)
- `items` (non-empty array of per-item records)

Per item:

- `input_id` (string)
- `prediction` (object; non-empty)
- `reasoning` (optional string; included when present in outputs and enabled via CLI flags)

## JSON Schema (excerpt)

The exporter uses the following JSON Schema (Draft 2020-12) for validation:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["schema_version", "run_id", "model_id", "created_at", "submissions"],
  "properties": {
    "schema_version": {"type": "integer", "minimum": 1},
    "run_id": {"type": "string", "minLength": 1},
    "model_id": {"type": "string", "minLength": 1},
    "created_at": {"type": "string"},
    "submissions": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["task_id", "items"],
        "properties": {
          "task_id": {"type": "string", "minLength": 1},
          "items": {
            "type": "array",
            "minItems": 1,
            "items": {
              "type": "object",
              "required": ["input_id", "prediction"],
              "properties": {
                "input_id": {"type": "string", "minLength": 1},
                "prediction": {"type": "object", "minProperties": 1},
                "reasoning": {"type": "string", "minLength": 1}
              },
              "additionalProperties": true
            }
          }
        },
        "additionalProperties": true
      }
    }
  },
  "additionalProperties": true
}
```

Download the full schema as a JSON file: [submission.schema.json](schema/submission.schema.json)

## Example

```json
{
  "schema_version": 1,
  "run_id": "run-20250906-abc",
  "model_id": "hf-sum",
  "created_at": "2025-09-06T02:15:00+00:00",
  "submissions": [
    {
      "task_id": "medical_qa",
      "items": [
        {
          "input_id": "ex1",
          "prediction": {"answer": "A"},
          "reasoning": "Short rationale"
        },
        {
          "input_id": "ex2",
          "prediction": {"answer": "B"}
        }
      ]
    }
  ]
}
```

## CLI usage

- Generate from a saved report:

```bash
# Include reasoning traces (default)
python -m bench.cli_typer generate-submission \
  --run-id <run-id> \
  --results-dir ./results \
  --out submission.json \
  --include-reasoning

# Or explicitly disable reasoning traces
python -m bench.cli_typer generate-submission \
  --run-id <run-id> \
  --results-dir ./results \
  --out submission.json \
  --no-include-reasoning
```

- Export directly during evaluation (no disk reload):

```bash
python -m bench.cli_typer evaluate <model-id> \
  --tasks <task-id> \
  --tasks-dir bench/tasks \
  --output-dir results \
  --format json --save-results \
  --export-submission results/submission.json \
  --export-submission-include-reasoning
```

## Programmatic usage

- From `ResultAggregator`:

```python
from bench.evaluation.result_aggregator import ResultAggregator

ra = ResultAggregator()
# ... add EvaluationResult to a run_id ...
ra.export_leaderboard_submission("run-xyz", "results/submission.json")
```

- From a `BenchmarkReport` instance:

```python
from bench.leaderboard.submission import build_and_validate_submission

payload = build_and_validate_submission(report, include_reasoning=True)
```
