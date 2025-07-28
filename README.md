# MEDDSAI Benchmark: Evaluating Large Language Models in Healthcare

[![Documentation](https://img.shields.io/badge/docs-Read%20the%20Docs-blue)](https://meddsai.readthedocs.io/)
[![Leaderboard](https://img.shields.io/badge/leaderboard-View%20Rankings-brightgreen)](https://www.meddsai.org/leaderboard)

[日本語](README_ja.md) | [中文简体](README_zh.md) | [Bahasa Indonesia](README_id.md)

## 📰 News

- **July 28, 2025**: Initial development of MEDDSAI Benchmark announced! Includes 200 tasks across diagnostics, summarization, and patient communication, with a future public leaderboard on [dashboard.meddsai.org](https://dashboard.meddsai.org).
- **Future**: Planning federated evaluation track inspired by MedPerf for privacy-sensitive data, and multimodal tasks with imaging, inspired by MedArena.

## 👋 Overview

The MEDDSAI Benchmark (Medical Domain-Specific AI Benchmark) is a comprehensive framework for evaluating Large Language Models (LLMs) in healthcare. Inspired by SWE-bench, it challenges models to solve real-world medical tasks, such as clinical question-answering, diagnostic reasoning, summarization, and patient communication, validated by physician expertise.

With a public leaderboard, reproducible evaluation harness, and support for federated testing, MEDDSAI aims to drive innovation in medical AI while ensuring clinical relevance and data privacy.

MEDDSAI draws from leading healthcare benchmarks like HealthBench, Open Medical-LLM, SD Bench, MedPerf, and Stanford's MedArena. It combines real-world tasks, a transparent leaderboard, and privacy-conscious evaluation to meet the needs of researchers, clinicians, and developers.

To access the initial MEDDSAI dataset, run:

```python
from datasets import load_dataset
meddsai = load_dataset('MEDDSAI/MEDDSAI-Core', split='test')
```

## 🚀 Set Up

MEDDSAI uses Docker for reproducible evaluations, ensuring consistency across environments. For sensitive medical data, a federated evaluation option is planned, inspired by MedPerf.

### Prerequisites

- **Docker**: Install Docker following the [Docker setup guide](https://docs.docker.com/get-docker/). For Linux, see [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).
- **System Requirements**: Recommended x86_64 machine with 120GB free storage, 16GB RAM, and 8 CPU cores. ARM64 support (e.g., MacOS M-series) is experimental.
- **Python**: Version 3.8+ with pip.

### Installation

Clone and install MEDDSAI:

```bash
git clone git@github.com:MEDDSAI/MEDDSAI-bench.git
cd MEDDSAI-bench
pip install -e .
```

Test your installation with a sample task:

```bash
python -m meddsai.harness.run_evaluation \
    --predictions_path gold \
    --max_workers 1 \
    --instance_ids medqa_001 \
    --run_id validate-gold
```

> **Note**: For ARM-based systems, add `--namespace ''` to build evaluation images locally.

## 💽 Usage

Evaluate model predictions on the MEDDSAI-Core dataset (200 tasks) with:

```bash
python -m meddsai.harness.run_evaluation \
    --dataset_name MEDDSAI/MEDDSAI-Core \
    --predictions_path <path_to_predictions> \
    --max_workers <num_workers> \
    --run_id <run_id>
```

- Use `--predictions_path 'gold'` to verify ground-truth solutions.
- Use `--run_id` to name the evaluation run.
- Use `--federated true` (planned) for privacy-preserving evaluation.

Evaluation generates logs (`logs/evaluation`) and results (`evaluation_results`) in the current directory.

> **Warning**: MEDDSAI evaluation is resource-intensive. Use fewer than `min(0.75 * os.cpu_count(), 24)` for `--max_workers`. Ensure 120GB free disk space for Docker.

For all arguments, run:

```bash
python -m meddsai.harness.run_evaluation --help
```

See the [evaluation tutorial](docs/evaluation.md) for details on datasets and cloud-based evaluation (coming soon).

## 📊 Task Overview

MEDDSAI includes tasks reflecting real-world healthcare scenarios, validated by medical experts:

- **Medical Question-Answering**: Answer USMLE-style questions (e.g., MedQA, PubMedQA).
- **Diagnostic Reasoning**: Generate differential diagnoses from symptoms or EHR data.
- **Clinical Summarization**: Summarize patient notes or radiology reports.
- **Patient Communication**: Draft empathetic responses to patient queries.
- **Medical Entity Recognition**: Identify diseases, medications, or procedures in text.
- **Multimodal Tasks (Planned)**: Analyze text with imaging or wearable data.

### Datasets

- **MEDDSAI-Core**: 200 tasks across diagnostics, summarization, and communication.
- **MEDDSAI-Hard (Planned)**: 100 challenging tasks for advanced models.
- **MEDDSAI-Specialty (Planned)**: Domain-specific tasks (e.g., oncology, cardiology).
- **MEDDSAI-Multimodal (Planned)**: Tasks combining text and imaging.

**Sources**: Public datasets (MIMIC-IV, MedQA), synthetic personas (inspired by HealthBench), and de-identified clinical cases, validated by physicians.

### Metrics

- **Accuracy**: For question-answering and diagnostics.
- **F1-Score**: For entity recognition.
- **BLEU/ROUGE**: For summarization and communication.
- **Clinical Relevance**: Expert-scored metric for guideline alignment.
- **Combined Score**: Weighted average (40% diagnostics, 30% safety, 20% communication, 10% summarization).

## 🏆 Leaderboard

The [MEDDSAI Leaderboard](https://www.meddsai.org/leaderboard) ranks models based on combined scores and task-specific metrics, inspired by SWE-bench and HealthBench. Submit predictions and reasoning traces for transparency.

- **Subsets**: MEDDSAI-Core, Hard, Specialty, Multimodal (planned).
- **Submission**: Via [Hugging Face](https://huggingface.co/MEDDSAI) or [GitHub](https://github.com/MEDDSAI/MEDDSAI-bench), with automated and expert validation.
- **Transparency**: Evaluation logs and results are publicly available.

## ⬇️ Downloads

| Datasets | Baseline Models | Tools |
|----------|----------------|-------|
| 💿 [MEDDSAI-Core](https://huggingface.co/datasets/MEDDSAI/MEDDSAI-Core) | 🩺 [BioBERT](https://huggingface.co/bert-base-uncased) (Baseline) | 🛠️ [MEDDSAI-Toolkit](https://github.com/MEDDSAI/meddsai-tools) (Planned) |
| 💿 [MEDDSAI-Hard](https://huggingface.co/datasets/MEDDSAI/MEDDSAI-Hard) (Planned) | 🩺 [Med-PaLM](https://ai.google/healthcare/research/med-palm/) (Baseline, Planned) | 🛠️ [Federated Evaluation API](https://medperf.ai) (Planned) |

## 💫 Contributions

We welcome contributions from the medical, AI, and open-source communities! To contribute:

1. Submit new tasks, datasets, or metrics via [pull requests](https://github.com/MEDDSAI/MEDDSAI-bench/pulls).
2. Report issues or suggest improvements on [GitHub Issues](https://github.com/MEDDSAI/MEDDSAI-bench/issues).
3. Join our community on [X](https://twitter.com/meddsaibench) or [Hugging Face](https://huggingface.co/MEDDSAI).

**Contact**: Dr. Junaidi Ahmad (junaidi@meddsai.org), AI Research Lead.

## ✍️ Citation & License

- **License**: MIT. See [LICENSE.md](LICENSE.md).
- **Cite MEDDSAI as**:

```bibtex
@inproceedings{
    meddsai2025,
    title={{MEDDSAI}: A Benchmark for Evaluating Large Language Models in Healthcare},
    author={Kresna Sucandra and Team MEDDSAI},
    booktitle={TBD},
    year={2025},
    url={https://www.meddsai.org}
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

MEDDSAI Benchmark: Advancing healthcare AI through transparent, clinically relevant evaluation of LLMs.

© 2025 [MEDDSAI Team](https://www.meddsai.org/team)