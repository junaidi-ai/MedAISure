# Metrics Overview

This section summarizes metric categories and how scores are computed and aggregated.

## Categories
- Clinical Accuracy: correctness and safety of medical content
- Reasoning Quality: structure, plausibility, and evidence use in reasoning
- Domain-specific: task- or specialty-specific checks

## Methodology (flow)
1. Normalize inputs/outputs
   - Lowercasing, whitespace, common medical synonym normalization
2. Extract features
   - Clinical entities by category (e.g., dx, meds) or structural markers (steps, evidence)
3. Score per-sample
   - Clinical Accuracy: weighted Jaccard over extracted entities; fallback to normalized string compare
   - Reasoning Quality: composite of overlap F1, structure, evidence, factual consistency minus fallacy penalty
4. Aggregate
   - Per-task: mean of per-sample scores (with optional weighting per component)
   - Overall: average across tasks (reported in `BenchmarkReport`)

## Weighting & breakdowns
- Component weights are defined in metric implementations (see API links below).
- Per-sample breakdowns are accessible via metric APIs (e.g., `get_last_breakdown()`), and are surfaced in evaluation metadata when available.

For how individual metric keys map into high-level categories used by our combined score, see [Metric Categories](./metric_categories.md). To compute a weighted combined score from the CLI, see [Combined score via CLI](../api/cli.md#combined-score-via-cli-typer).

## Deep links (API)
- Python API → Metrics (clinical): api/reference.md#metrics-clinical
  - `ClinicalAccuracyMetric.calculate`
  - `ReasoningQualityMetric.calculate`

## References
- [metrics_guidelines.md](../metrics_guidelines.md)
- [metrics_testing.md](../metrics_testing.md)
