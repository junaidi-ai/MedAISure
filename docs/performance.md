# Performance Tips

- Prefer batching via `batch_size` in `EvaluationHarness.evaluate()`; default is 8
- Use GPU for HF models by setting `device` (e.g., `device=0`) in `model_kwargs`/`pipeline_kwargs`
- Cache predictions by providing `cache_dir`; reuse with `use_cache=True`
- Minimize heavy metrics; some (e.g., ROUGE) are slower and optional
- Filter tasks to a subset during development
- For large runs, export CSV/Markdown and analyze externally
