# MedAISure Example Notebooks

This folder contains example Jupyter notebooks demonstrating key workflows with the MedAISure benchmark.

Notebooks:
- 01_basic_evaluation.ipynb — Quick start: load tasks, run a local demo model, view results
- 02_custom_task_creation.ipynb — How to define and load a custom task (YAML/JSON schema)
- 02b_python_task_interface_and_registration.ipynb — Define a MedicalTask in Python, save to YAML, register/load via TaskRegistry, and evaluate
- 03_result_analysis_visualization.ipynb — Load results, compare runs, and visualize metrics
- 04_advanced_configuration.ipynb — Batch size, caching, report export, and HuggingFace models
- 04b_custom_metrics_and_registry.ipynb — Register a custom metric with MetricCalculator, compute it post-hoc on evaluation outputs, and combine with built-ins

## Quickstart

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Launch Jupyter (use your preferred environment)

```bash
jupyter lab
# or
jupyter notebook
```

3) Open a notebook from this folder and run all cells in order.

Notes:
- The examples default to a simple local model (`bench.examples.mypkg.mylocal`) to avoid external downloads.
- You can switch to HuggingFace models by setting `model_type='huggingface'` and providing `model_path` and `hf_task`.
- Results are saved to the `results/` directory by default; you can change paths via `EvaluationHarness`.
