# API Models

Use hosted HTTP APIs by configuring `ModelRunner` with an endpoint and optional auth.

## Configure API model

```python
from bench.evaluation.model_runner import ModelRunner

mr = ModelRunner()
mr.load_model(
    "api-demo",
    model_type="api",
    endpoint="https://api.example.com/v1/predict",
    api_key="sk_example",          # optional; sent as Authorization: Bearer <key>
    headers={"X-Client": "MedAISure"},  # merged with auth header
    timeout=30.0,
    max_retries=1,
)
```

## Request/response format

- Input to `run_model(model_id, inputs)` is a list of dicts (`inputs: list[dict]`).
- `ModelRunner` will POST JSON: `{ "inputs": [ ... ] }` to the configured `endpoint`.
- The API must return a JSON body with a list of outputs aligned 1:1, e.g. `{ "outputs": [ {"answer": "..."}, ... ] }`.

Example call
```python
batch = [
    {"question": "What is BP?"},
    {"question": "Define tachycardia."},
]
outs = mr.run_model("api-demo", inputs=batch)
# outs -> list[dict], e.g., [{"answer": "blood pressure"}, {"answer": "HR > 100 bpm"}]
```

## Error handling & retries

- Non-2xx responses or malformed JSON raise an error in the runner.
- `max_retries` uses simple retry with backoff for transient failures.
- `timeout` applies per HTTP request.

## Security notes

- API keys are sent as `Authorization: Bearer <api_key>` by default; you can override via `headers`.
- Avoid logging raw keys; prefer environment variables in your invocation script.

## Harness example

```python
from bench.evaluation.harness import EvaluationHarness

h = EvaluationHarness(tasks=["medical_qa"])  # or provide a MedicalTask
h.evaluate(
    model_id="api-demo",
    runner=mr,
)
```

See also
- Core API: `docs/api/core_api.md`
- Python API → Model Runner: `api/reference.md#model-runner`
