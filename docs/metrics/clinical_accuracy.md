# Clinical Accuracy

Definition: Measures correctness of medical facts, diagnoses, and recommendations.

Scoring Dimensions
- Correctness: factual alignment with gold standard
- Safety: absence of dangerous or contraindicated advice
- Specificity: precise, actionable answers vs vague text

Methodology
- Exact match / label accuracy for discrete tasks
- NLI-based agreement or LLM-judge with calibrated rubric
- Penalize hallucinations; weight critical errors higher

Usage
- Applies to: Medical QA, Diagnostic Reasoning (final dx), Summarization (facts)
- Combine with `Reasoning Quality` for holistic view

## Code reference

Implementation: `bench/evaluation/metrics/clinical.py` → `ClinicalAccuracyMetric`

Excerpt
```python
class ClinicalAccuracyMetric(Metric):
    def calculate(self, expected_outputs: list[dict], model_outputs: list[dict]) -> float:
        Metric.validate_inputs(expected_outputs, model_outputs)
        # 1) normalize text
        # 2) extract naive clinical entities by category
        # 3) weighted Jaccard across categories
        # 4) fallback to normalized string comparison if no entities
        ...

    def get_last_breakdown(self) -> list[dict[str, Any]]:
        """Per-sample breakdown from the last calculate() call."""
        return self._last_breakdown
```

Notes
- Categories and weights are kept lightweight for dependency-free scoring.
- Synonym normalization is applied for common medical abbreviations.

See also
- Python API → Metrics (clinical): api/reference.md#metrics-clinical
