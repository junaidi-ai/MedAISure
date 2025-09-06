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

## Combined Score Formula
The combined score aggregates multiple category metrics into a single index. It is computed by `BenchmarkReport.add_combined_score()` using a weighted mean over the set of metrics present for each task.

Let `W = { (k, w_k) }` be the configured weights and `M_t = { (k, v_{t,k}) }` be the available metric values for task `t` (e.g., `diagnostics`, `safety`, `communication`, `summarization`). The per-task combined score is:

```
combined_t = sum_{k in present(t)} (w_k / Z_t) * v_{t,k}
```

Where:
- `present(t) = { k in W | k exists in task t's metrics }`
- `Z_t = sum_{k in present(t)} w_k` (renormalization factor)

Notes:
- If a task is missing some weighted categories, we renormalize the remaining weights so they still sum to 1.0 across the present categories (`renormalize_missing=True`).
- The overall combined score is the mean of `combined_t` across tasks where it can be computed.
- The metric name defaults to `combined_score`, but can be customized via CLI/config.

Appearance in reports:
- JSON: Included under `overall_scores[combined_metric_name]` and each `task_scores[task_id][combined_metric_name]`.
- Markdown/HTML: Prominently highlighted in the Overall section, along with the configured weights, followed by the full metric listings.

## Deep links (API)
- Python API → Metrics (clinical): api/reference.md#metrics-clinical
  - `ClinicalAccuracyMetric.calculate`
  - `ReasoningQualityMetric.calculate`

## References
- [metrics_guidelines.md](../metrics_guidelines.md)
- [metrics_testing.md](../metrics_testing.md)
