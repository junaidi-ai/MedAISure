# Patient Communication Tasks

The Patient Communication task family evaluates models on generating empathetic, safe, and clear responses to patient questions or concerns.

- Task type: `communication`
- Input schema: `{ patient_message: string, context?: string, age?: integer, risk_factors?: string }`
- Output schema: `{ response: string }`
- Suggested metrics: `rouge_l`, `clinical_relevance`

## Included Tasks

- `patient_communication_basic` — General patient responses to common medical questions
- `patient_communication_triage` — Triage-oriented guidance with red flags and next steps (non-diagnostic)

## Examples

```yaml
# bench/tasks/patient_communication_basic.yaml
name: "Patient Communication: Basic Responses"
description: "Generate empathetic, clear patient-facing responses to common medical questions."
```

```yaml
# bench/tasks/patient_communication_triage.yaml
name: "Patient Communication: Triage Advice"
description: "Generate clear, empathetic triage guidance for patient-reported symptoms (non-diagnostic)."
```

## Running the tasks

Python API:

```python
from bench.evaluation import EvaluationHarness

h = EvaluationHarness(tasks_dir="bench/tasks", results_dir="bench/results")
print([t["task_id"] for t in h.list_available_tasks()])

report = h.evaluate(
    model_id="textattack/bert-base-uncased-MNLI",  # replace with a suitable model for generation
    task_ids=[
        "patient_communication_basic",
        "patient_communication_triage",
    ],
    model_type="huggingface",
    batch_size=4,
    use_cache=False,
)
print(report.overall_scores)
```

CLI (Typer):

```bash
python -m bench.cli_typer evaluate \
  --tasks patient_communication_basic,patient_communication_triage \
  --tasks-dir bench/tasks \
  --output-dir bench/results \
  --save-results
```

## Notes

- These sample datasets are small and intended as scaffold tasks. For robust evaluation, expand datasets and references.
- Ensure responses avoid diagnostic claims unless the task explicitly permits and includes proper validation checks.
