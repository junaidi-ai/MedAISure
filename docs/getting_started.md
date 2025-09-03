# Getting Started with MedAISure

This project ships a Quick Start guide. For a concise intro, see [Quick Start](quick_start.md).

## Installation

```bash
# From PyPI (recommended when published)
pip install medaisure-benchmark

# Or from source
git clone https://github.com/junaidi-ai/MedAISure.git
cd MedAISure
pip install -e .
```

## CLI Usage

```bash
# List available tasks
medaisure list-tasks

# Evaluate a model on a task
medaisure evaluate my-model --tasks medical_qa_basic

# Generate a report
medaisure generate-report ./results/my-model_results.json --format md
```

## Python API

```python
from bench.evaluation.harness import EvaluationHarness
from bench.models import registry as model_registry

h = EvaluationHarness()
model = model_registry.load("gpt2")
report = h.evaluate(model, tasks=["medical_qa_basic"])
print(report.overall_scores)
```
