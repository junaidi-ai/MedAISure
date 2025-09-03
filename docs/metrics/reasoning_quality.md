# Reasoning Quality

Definition: Evaluates structure, coherence, and clinical plausibility of reasoning traces.

Rubric (0-5)
- Structure: clear steps, justified transitions
- Plausibility: aligns with clinical guidelines/pathophysiology
- Use of Data: incorporates key findings; avoids overfitting
- Calibration: expresses uncertainty appropriately

Scoring Approaches
- Rule-based checks (section headers, presence of justification)
- LLM-judge with anchored examples
- Pair with Clinical Accuracy for final scorecard

## Code reference

Implementation: `bench/evaluation/metrics/clinical.py` → `ReasoningQualityMetric`

Excerpt
```python
class ReasoningQualityMetric(Metric):
    def calculate(self, expected_outputs: list[dict], model_outputs: list[dict]) -> float:
        # overlap F1 + structure + evidence + factual consistency - fallacy_penalty
        ...

    # Components (weights in _W_WEIGHTS)
    # - _f1(a, b): token-overlap F1
    # - _structure_score(text): markers like "because", numbered steps
    # - _evidence_score(text): labs/imaging/vitals mentions
    # - _factual_consistency_score(diagnosis, text): simple rules
    # - _fallacy_penalty(text): penalize common fallacy phrases

See also
- Python API → Metrics (clinical): api/reference.md#metrics-clinical
