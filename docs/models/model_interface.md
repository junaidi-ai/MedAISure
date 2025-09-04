# Model Interface

All models should implement a consistent interface compatible with the Evaluation Harness.

Key aspects:
- Loading/configuration
- Generate/invoke API
- Metadata and resource management

## Loading models with ModelRunner

`bench.evaluation.model_runner.ModelRunner` supports HuggingFace, local Python modules, simple HTTP APIs, and interface models.

HuggingFace (pipeline) with advanced options
```python
from bench.evaluation.model_runner import ModelRunner

mr = ModelRunner()
pipe = mr.load_model(
    "hf-sum",
    model_type="huggingface",
    model_path="sshleifer/tiny-t5",
    hf_task="summarization",
    # Advanced HF loading options
    device=-1,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    low_cpu_mem_usage=True,
    revision=None,
    # Generation kwargs stored and later passed during run
    generation_kwargs={"max_new_tokens": 64, "do_sample": False},
)
```

Local Python module
```python
model = mr.load_model(
    "local-demo",
    model_type="local",
    model_path="/path/to/weights-or-dir",
    module_path="bench.examples.mypkg.mylocal",
    # optional: load_func="load_model",
)
```

Simple HTTP API
```python
cfg = mr.load_model(
    "api-demo",
    model_type="api",
    endpoint="https://api.example/v1/predict",
    api_key="sk-...",
    timeout=30.0,
    max_retries=1,
)
```

## Running inference

`run_model()` signature
```python
def run_model(
    self,
    model_id: str,
    inputs: list[dict[str, Any]],
    batch_size: int = 8,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    ...
```

HuggingFace generation kwargs passthrough
```python
# If generation_kwargs were supplied to load_model(...), they will be
# passed to the pipeline call for summarization/text-generation tasks.
out = mr.run_model(
    "hf-sum",
    inputs=[{"document": "Patient note ..."}],
    batch_size=2,
)
```

Local/API/interface models
- Local: your callable returned by `load_model()` is invoked as `model(batch, **kwargs)`.
- API: a generated callable posts `batch` to `endpoint` and parses response.
- Interface: objects implementing `ModelInterface` can be registered and used via `.predict(batch)`.

See source: `bench/evaluation/model_runner.py`.
