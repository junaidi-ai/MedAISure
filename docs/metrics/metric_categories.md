# Metric Categories (Canonical Mapping)

This page defines the canonical mapping of metric keys used in MedAISure to four high-level categories. These categories are used by the combined score feature to compute a single, weighted index across tasks.

- Diagnostics
- Safety
- Communication
- Summarization

The mapping focuses on consistency across tasks and reports. If a task omits a category, combined score computation can optionally re-normalize the remaining weights.

## Diagnostics
Core capability to arrive at correct clinical conclusions and decisions.

Recommended metrics:
- `clinical_accuracy`
- `diagnostic_accuracy`
- `final_answer_correct`
- `exact_match` (when the task’s primary target is correctness)

Notes:
- If a task uses `accuracy` for diagnostic reasoning, include it here.

## Safety
Measures related to patient safety, harmfulness avoidance, and guideline adherence.

Recommended metrics:
- `safety`
- `harm_avoidance`
- `toxicity_reduction`
- `factuality_safety` (safety-focused factual checks)

Notes:
- If a task provides a binary/graded safety assessment, map it here.

## Communication
Clarity, structure, and usefulness of the model’s communication to clinicians and patients.

Recommended metrics:
- `communication`
- `coherence`
- `helpfulness`
- `instruction_following` (when focused on communicative compliance)

Notes:
- Use for language quality and pragmatic helpfulness.

## Summarization
Quality of summarizing clinical content (notes, encounters, discharge letters).

Recommended metrics:
- `summarization`
- `summary_quality`
- `rouge_l`, `rouge1`, `rouge2` (if used as the task’s primary summary metric)
- `bertscore` (summary quality proxy)

Notes:
- If a task focuses on summarization, place its principal metric(s) here.

## Usage in Combined Score
The combined score accepts weights for any of these category keys. Each task contributes a per-task `combined_score` using the intersection of weights and available metrics.

- Missing categories: When a task lacks some categories, the remaining present weights can be re-normalized (default behavior) so the combined score remains comparable.
- Overlapping metrics: If a metric could fit two categories, choose the category that matches the task objective and documentation.

### CLI Usage Examples
You can enable the combined score directly from the CLI `evaluate` command (see `bench/cli_typer.py`). The `--combined-weights` flag accepts either JSON or comma-separated `key=value` pairs. The `--combined-metric-name` controls the output metric name (default: `combined_score`).

- JSON form:
  ```bash
  task-master evaluate <model-id> \
    --tasks <task-id> \
    --tasks-dir <path/to/tasks> \
    --model-type local \
    --output-dir out \
    --format json \
    --save-results \
    --combined-weights '{"diagnostics": 0.4, "safety": 0.3, "communication": 0.2, "summarization": 0.1}' \
    --combined-metric-name combined_score
  ```

- Comma-separated pairs:
  ```bash
  task-master evaluate <model-id> \
    --tasks <task-id> \
    --tasks-dir <path/to/tasks> \
    --model-type local \
    --output-dir out \
    --format json \
    --save-results \
    --combined-weights diagnostics=0.4,safety=0.3,communication=0.2,summarization=0.1 \
    --combined-metric-name combined_score
  ```

Notes:
- Weights must be non-negative and sum to 1.0 (±1e-6). Invalid inputs will be rejected.
- If a task is missing some weighted categories, remaining present weights are re-normalized by default (see `BenchmarkReport.add_combined_score(renormalize_missing=True)`).
- The combined score will appear in all generated report formats (JSON/CSV/Markdown/HTML).

## Tuned Default Mapping (as of current implementation)
The default category mapping used by `ResultAggregator.add_category_aggregates()` has been tuned to reflect the actual metrics available in `MetricCalculator`:

- Diagnostics
  - accuracy, diagnostic_accuracy, clinical_correctness, exact_match, final_answer_correct
- Safety
  - safety, harm_avoidance, toxicity_reduction, factuality_safety
- Communication
  - reasoning_quality, communication, coherence, helpfulness, instruction_following
- Summarization
  - summarization, summary_quality, rouge_l, rouge1, rouge2, bertscore, clinical_relevance, factual_consistency

Notes:
- Matching is case-insensitive. Non-numeric values are ignored.
- Per-task category scores are means over the present mapped metrics.
- Overall category scores are means over tasks where the category exists.

## Overriding the Category Map
You can supply a custom category map if your project uses different metric names.

- CLI (inline JSON):
  ```bash
  task-master evaluate <model-id> \
    --tasks <task-id> \
    --tasks-dir bench/tasks \
    --category-map '{"diagnostics":["accuracy","exact_match"],"summarization":["rouge_l"]}' \
    --combined-weights diagnostics=0.7,summarization=0.3
  ```

- CLI (JSON/YAML file):
  ```bash
  task-master evaluate <model-id> \
    --tasks <task-id> \
    --tasks-dir bench/tasks \
    --category-map-file .taskmaster/configs/category_map.yaml \
    --combined-weights diagnostics=0.7,summarization=0.3
  ```

- Config file (`BenchmarkConfig`):
  Add in your config (JSON or YAML):
  ```yaml
  category_map:
    diagnostics: [accuracy, exact_match]
    summarization: [rouge_l]
  combined_weights:
    diagnostics: 0.7
    summarization: 0.3
  combined_metric_name: combined_score
  ```

Under the hood, the harness calls `ResultAggregator.add_category_aggregates(run_id, category_map=...)` before computing the combined score, so your weights can reference categories even when tasks only expose raw metrics (e.g., `accuracy`, `rouge_l`).

## Adding New Metrics
When introducing a new metric:
- Select the category based on the task’s objective.
- Update this page if the metric becomes standard across multiple tasks.
- Keep names consistent across tasks for stable aggregation.
