# Performance Tips

This page collects practical tips to make MedAISure evaluations faster and more memory‑efficient. It applies to both the example notebooks in `bench/examples/notebooks/` and your own scripts.

- Path: `docs/04c_performance_tips.md`
- Related files: `bench/evaluation/harness.py`, `bench/evaluation/model_runner.py`

## Batching

- Use smaller `batch_size` if you see OOM (out‑of‑memory) errors.
- For CPU or small GPUs, start at `batch_size=1..4` and increase gradually.
- In notebooks calling `EvaluationHarness.evaluate(...)`, pass `batch_size=<N>`.
- Keep batch sizes consistent across runs when comparing models to ensure fairness.

## Caching

- Enable caching to avoid recomputing results when re‑running the same model+task:
  - `EvaluationHarness(..., cache_dir='cache')` and `evaluate(..., use_cache=True)`.
- The harness caches per‑task results. Subsequent runs will load from cache if inputs, model ID, and task ID match.
- Clear or change the cache directory if you want to force fresh computation.

## HuggingFace dtype/device

When using `model_type='huggingface'` with `ModelRunner`:

- Device selection:
  - CPU: `device=-1`
  - Single GPU: `device=0`
  - Multi‑GPU or automatic placement: set `device_map='auto'` in loading kwargs.
- Dtype for memory/perf trade‑offs:
  - `torch.float16` (FP16) or `torch.bfloat16` often reduce memory and can speed up inference on supported GPUs.
  - Pass as `torch_dtype='float16'` or `torch_dtype='bfloat16'` in loading kwargs.
- Additional loading options (forwarded by `ModelRunner`):
  - `low_cpu_mem_usage=True` reduces peak CPU RAM during loading.
  - `trust_remote_code=False` by default; set `True` only if you trust the model repo.
  - `revision='<tag-or-commit>'` to pin a specific model revision.
- Generation kwargs (for generative tasks):
  - Use smaller `max_new_tokens` to reduce latency and cost.
  - Disable sampling for deterministic baselines: `do_sample=False`.

Example (see `04_advanced_configuration.ipynb`):

```python
rep = h.evaluate(
    model_id='hf-demo',
    task_ids=['clinical_summarization_basic'],
    model_type='huggingface',
    hf_task='summarization',
    model_path='sshleifer/tiny-t5',
    batch_size=2,
    use_cache=True,
    generation_kwargs={'max_new_tokens': 64, 'do_sample': False},
    device=0,                 # GPU 0 (use -1 for CPU)
    torch_dtype='float16',    # or 'bfloat16' if supported
    low_cpu_mem_usage=True,
)
```

## Dataset sizing and sampling

- For quick iteration, work with a small dataset subset (e.g., 20–100 examples) to validate your pipeline.
- Once metrics are stable, scale up to the full dataset for final results.
- If constructing tasks programmatically (see `02b_python_task_interface_and_registration.ipynb`), you can:
  - Provide a small `dataset` inline for rapid tests.
  - Store the full dataset on disk and create a second, larger task YAML for final runs.

## Logging and reporting

- Use `log_level='INFO'` (default) to monitor progress without excessive verbosity.
- Save reports in lightweight formats during iteration:
  - `report_formats=['json']` while prototyping
  - Add `['csv','md','html']` only when needed for sharing/visualization.

## Troubleshooting common performance issues

- GPU OOM during HF model load:
  - Try `torch_dtype='float16'` or `'bfloat16'`, reduce `batch_size`, or set `device_map='auto'`.
- Slow CPU runs:
  - Reduce `max_new_tokens`, `batch_size`, and dataset size; enable caching.
- Frequent re‑runs:
  - Ensure `use_cache=True` and `cache_dir` is set; avoid modifying inputs between runs unless necessary.

For more details, check:
- `bench/evaluation/model_runner.py` for loading and generation kwargs
- `bench/evaluation/harness.py` for evaluation parameters and caching
- `bench/examples/notebooks/04_advanced_configuration.ipynb` for a concrete configuration example
