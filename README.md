# MedAISure Benchmark: Evaluating Large Language Models in Healthcare

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/medaisure/medaisure-benchmark/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/medaisure/medaisure-benchmark)

A comprehensive framework for evaluating Large Language Models (LLMs) in healthcare applications. The MedAISure Benchmark provides a standardized way to assess model performance on various medical NLP tasks.

## ✨ Features

- **Modular Architecture**: Easily extensible with custom models, tasks, and metrics
- **Support for Multiple Model Types**: Local models, Hugging Face models, and API-based models
- **Comprehensive Evaluation**: Multiple metrics and aggregation methods
- **Reproducible Results**: Caching and versioning support
- **Pre-configured Medical Tasks**: Includes tasks for clinical text classification, NLI, and more

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/junaidi-ai/MedAISure.git
cd MedAISure

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .[dev]  # Includes development dependencies
```

### Get Started

```python
from bench.evaluation import EvaluationHarness

h = EvaluationHarness(
    tasks_dir="bench/tasks",
    results_dir="bench/results",
    cache_dir="bench/results/cache",
    log_level="INFO",
)

# Discover tasks and run one with a Hugging Face model
tasks = h.list_available_tasks()
report = h.evaluate(
    model_id="textattack/bert-base-uncased-MNLI",
    task_ids=[tasks[0]["task_id"]],
    model_type="huggingface",
    batch_size=4,
    use_cache=False,
)
print(report.overall_scores)
```

## 📚 Documentation

For detailed documentation, including API reference and advanced usage, please visit our [documentation site](https://junaidi-ai.github.io/MedAISure/).

Also see local docs in this repo:

- Usage Guide: [docs/usage.md](docs/usage.md)
- API Reference: [docs/api_reference.md](docs/api_reference.md)
- Extension Guide: [docs/extensions.md](docs/extensions.md)
- Configuration: [docs/configuration.md](docs/configuration.md)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md)
- Performance Tips: [docs/performance.md](docs/performance.md)

Examples you can run:

- HF text classification: [bench/examples/run_hf_text_classification.py](bench/examples/run_hf_text_classification.py)
- HF summarization: [bench/examples/run_hf_summarization.py](bench/examples/run_hf_summarization.py)
- Local model: [bench/examples/run_local_model.py](bench/examples/run_local_model.py)
- API model: [bench/examples/run_api_model.py](bench/examples/run_api_model.py)
- Custom metric: [bench/examples/register_custom_metric.py](bench/examples/register_custom_metric.py)


## 👋 Overview

The MedAISure Benchmark (Medical Domain-Specific AI Benchmark) is a comprehensive framework for evaluating Large Language Models (LLMs) in healthcare. Inspired by SWE-bench, it challenges models to solve real-world medical tasks, such as clinical question-answering, diagnostic reasoning, summarization, and patient communication, validated by physician expertise.

With a public leaderboard, reproducible evaluation harness, and support for federated testing, MedAISure aims to drive innovation in medical AI while ensuring clinical relevance and data privacy.

MedAISure draws from leading healthcare benchmarks like HealthBench, Open Medical-LLM, SD Bench, MedPerf, and Stanford's MedArena. It combines real-world tasks, a transparent leaderboard, and privacy-conscious evaluation to meet the needs of researchers, clinicians, and developers.

To access the initial MedAISure dataset, run:

```python
from datasets import load_dataset
medaisure = load_dataset('MedAISure/MedAISure-Core', split='test')
```

## 🚀 Set Up

MedAISure uses Docker for reproducible evaluations, ensuring consistency across environments. For sensitive medical data, a federated evaluation option is planned, inspired by MedPerf.

### Prerequisites

- **Docker**: Install Docker following the [Docker setup guide](https://docs.docker.com/get-docker/). For Linux, see [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).
- **System Requirements**: Recommended x86_64 machine with 120GB free storage, 16GB RAM, and 8 CPU cores. ARM64 support (e.g., MacOS M-series) is experimental.
- **Python**: Version 3.8+ with pip.



Test your installation with a sample task:

```bash
python -m medaisure.harness.run_evaluation \
    --predictions_path gold \
    --max_workers 1 \
    --instance_ids medqa_001 \
    --run_id validate-gold
```

> **Note**: For ARM-based systems, add `--namespace ''` to build evaluation images locally.

## 💽 Usage

Evaluate model predictions on the MedAISure-Core dataset (200 tasks) with:

```bash
python -m medaisure.harness.run_evaluation \
    --dataset_name MedAISure/MedAISure-Core \
    --predictions_path <path_to_predictions> \
    --max_workers <num_workers> \
    --run_id <run_id>
```

- Use `--predictions_path 'gold'` to verify ground-truth solutions.
- Use `--run_id` to name the evaluation run.
- Use `--federated true` (planned) for privacy-preserving evaluation.

Evaluation generates logs (`logs/evaluation`) and results (`evaluation_results`) in the current directory.

> **Warning**: MedAISure evaluation is resource-intensive. Use fewer than `min(0.75 * os.cpu_count(), 24)` for `--max_workers`. Ensure 120GB free disk space for Docker.

For all arguments, run:

```bash
python -m medaisure.harness.run_evaluation --help
```

See the [evaluation tutorial](docs/evaluation.md) for details on datasets and cloud-based evaluation (coming soon).

## 📊 Task Overview

MedAISure includes tasks reflecting real-world healthcare scenarios, validated by medical experts:

- **Medical Question-Answering**: Answer USMLE-style questions (e.g., MedQA, PubMedQA).
- **Diagnostic Reasoning**: Generate differential diagnoses from symptoms or EHR data.
- **Clinical Summarization**: Summarize patient notes or radiology reports.
- **Patient Communication**: Draft empathetic responses to patient queries.
- **Medical Entity Recognition**: Identify diseases, medications, or procedures in text.
- **Multimodal Tasks (Planned)**: Analyze text with imaging or wearable data.

### Datasets

- **MedAISure-Core**: 200 tasks across diagnostics, summarization, and communication.
- **MedAISure-Hard (Planned)**: 100 challenging tasks for advanced models.
- **MedAISure-Specialty (Planned)**: Domain-specific tasks (e.g., oncology, cardiology).
- **MedAISure-Multimodal (Planned)**: Tasks combining text and imaging.

**Sources**: Public datasets (MIMIC-IV, MedQA), synthetic personas (inspired by HealthBench), and de-identified clinical cases, validated by physicians.

### Metrics

- **Accuracy**: For question-answering and diagnostics.
- **F1-Score**: For entity recognition.
- **BLEU/ROUGE**: For summarization and communication.
- **Clinical Relevance**: Expert-scored metric for guideline alignment.
- **Combined Score**: Weighted average (40% diagnostics, 30% safety, 20% communication, 10% summarization).

## 🏆 Leaderboard

The [MedAISure Leaderboard](https://www.medaisure.org/leaderboard) ranks models based on combined scores and task-specific metrics, inspired by SWE-bench and HealthBench. Submit predictions and reasoning traces for transparency.

- **Subsets**: MedAISure-Core, Hard, Specialty, Multimodal (planned).
- **Submission**: Via [Hugging Face](https://huggingface.co/MedAISure) or [GitHub](https://github.com/MedAISure/MedAISure), with automated and expert validation.
- **Transparency**: Evaluation logs and results are publicly available.

## ⬇️ Downloads

| Datasets | Baseline Models | Tools |
|----------|----------------|-------|
| 💿 [MedAISure-Core](https://huggingface.co/datasets/MedAISure/MedAISure-Core) | 🩺 [BioBERT](https://huggingface.co/bert-base-uncased) (Baseline) | 🛠️ [MedAISure-Toolkit](https://github.com/MEDAISURE/medaisure-tools) (Planned) |
| 💿 [MedAISure-Hard](https://huggingface.co/datasets/MedAISure/MedAISure-Hard) (Planned) | 🩺 [Med-PaLM](https://ai.google/healthcare/research/med-palm/) (Baseline, Planned) | 🛠️ [Federated Evaluation API](https://medperf.ai) (Planned) |

## 💫 Contributions

We welcome contributions from the medical, AI, and open-source communities! To contribute:

1. Submit new tasks, datasets, or metrics via [pull requests](https://github.com/MedAISure/MedAISure/pulls).
2. Report issues or suggest improvements on [GitHub Issues](https://github.com/MedAISure/MedAISure/issues).
3. Join our community on [X](https://twitter.com/medaisurebench) or [Hugging Face](https://huggingface.co/MedAISure).

**Contact**: Dr. Kresna Sucandra (kresnasucandra@gmail.com), AI Research Lead.

## ✍️ Citation & License

- **License**: MIT. See [LICENSE.md](LICENSE.md).
- **Cite MedAISure as**:

```bibtex
@inproceedings{
    medaisure2025,
    title={{MedAISure}: A Benchmark for Evaluating Large Language Models in Healthcare},
    author={Kresna Sucandra and Team MedAISure},
    booktitle={TBD},
    year={2025},
    url={https://medaisure.junaidi.ai}
}
```

## 📚 Related Projects

- [HealthBench](https://github.com/openai/healthbench) – OpenAI's healthcare conversation benchmark.
- [Open Medical-LLM](https://huggingface.co/spaces/openmedllm-leaderboard/leaderboard) – Hugging Face's medical LLM leaderboard.
- [SD Bench](https://aka.ms/sdbench) – Microsoft's sequential diagnosis benchmark.
- [MedPerf](https://medperf.ai) – Federated evaluation platform.
- [MedArena](https://medarena.stanford.edu) – Stanford's clinician-driven platform.

---

**About**

MedAISure Benchmark: Advancing healthcare AI through transparent, clinically relevant evaluation of LLMs.

© 2025 [MedAISure Team](https://medaisure.junaidi.ai/team)
