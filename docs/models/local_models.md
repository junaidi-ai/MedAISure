# Local Models

Use local models with `ModelRunner` either via HuggingFace pipelines or a Python callable you provide.

## HuggingFace pipelines

Load a HF model with advanced options and generation kwargs (stored for later):
```python
from bench.evaluation.model_runner import ModelRunner

mr = ModelRunner()
mr.load_model(
    "hf-sum",
    model_type="huggingface",
    model_path="sshleifer/tiny-t5",
    hf_task="summarization",
    # Advanced options mapped to AutoModel/AutoTokenizer/pipe
    device=-1,                 # CPU by default
    device_map="auto",        # GPU mapping if available
    torch_dtype="auto",       # or "float16", etc.
    trust_remote_code=False,
    low_cpu_mem_usage=True,
    revision=None,
    generation_kwargs={"max_new_tokens": 64, "do_sample": False},
)

outs = mr.run_model(
    "hf-sum",
    inputs=[{"document": "Patient note ..."}],
    batch_size=2,
)
```

Notes
- `generation_kwargs` are automatically passed for `summarization`/`text-generation` pipelines.
- Metadata like dtype/device may be captured for reporting where available.

## Python callable (fully local)

Implement and expose a loader that returns a callable `predict(batch, **kwargs)` or a simple function accepting `batch`:
```python
# bench/examples/mypkg/mylocal.py
def load_model(model_path: str | None = None, **kwargs):
    resources = {"path": model_path}

    def _predict(batch, **_):
        # batch: list[dict]; return list[dict] aligned 1:1
        results = []
        for item in batch:
            if "document" in item:
                results.append({"summary": item["document"][:50]})
            elif "question" in item:
                results.append({"answer": "placeholder"})
            else:
                results.append({"text": "ok"})
        return results

    return _predict
```

Register and run it via `ModelRunner`:
```python
mr = ModelRunner()
mr.load_model(
    "local-demo",
    model_type="local",
    module_path="bench.examples.mypkg.mylocal",
    model_path="/opt/models/demo",
    load_func="load_model",  # optional if named load_model
)

outs = mr.run_model(
    "local-demo",
    inputs=[{"question": "What is BP?"}],
)
```

## Harness example

```python
from bench.evaluation.harness import EvaluationHarness

h = EvaluationHarness(tasks=["medical_qa"])  # or provide a MedicalTask
h.evaluate(
    model_id="hf-sum",
    runner=mr,
)
```

See also
- Core API: `docs/api/core_api.md`
- Python API → Model Runner: `api/reference.md#model-runner`
