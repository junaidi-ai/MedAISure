# Tutorials and Example Notebooks

This page links to a set of runnable Jupyter notebooks under `bench/examples/notebooks/` that demonstrate common workflows with MedAISure.

Notebooks:
- 01 — Basic Evaluation: `bench/examples/notebooks/01_basic_evaluation.ipynb`
- 02 — Custom Task Creation: `bench/examples/notebooks/02_custom_task_creation.ipynb`
- 02b — Python Task Interface & Registration: `bench/examples/notebooks/02b_python_task_interface.ipynb`
- 03 — Result Analysis & Visualization: `bench/examples/notebooks/03_result_analysis_visualization.ipynb`
- 04 — Advanced Configuration: `bench/examples/notebooks/04_advanced_configuration.ipynb`
- 04b — Custom Metrics (Registration & Usage): `bench/examples/notebooks/04b_custom_metrics_and_registry.ipynb`

Quick steps to run locally:
1. Install project dependencies: `pip install -r requirements.txt`
2. Launch Jupyter: `jupyter lab` or `jupyter notebook`
3. Open the notebooks from the paths above and run cells top-to-bottom.

Notes:
- The examples default to a minimal local model at `bench/examples/mypkg/mylocal.py` so they run without external downloads.
- You can switch to HuggingFace models by setting `model_type='huggingface'` and adding `model_path` and `hf_task`. See the advanced configuration notebook for an example.
- Reports are saved under `results/` and can be exported to JSON/CSV/Markdown/HTML. Plotting requires `matplotlib` (optional).
