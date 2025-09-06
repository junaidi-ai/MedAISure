# How to Submit to the Leaderboard

This guide explains how to generate a valid submission JSON for the MedAISure leaderboard and where to find the schema and examples.

## Overview

A leaderboard submission is a JSON file that contains, for each task:
- Items aligned with your evaluation inputs
- Each item includes an `input_id`, a `prediction` object, and an optional `reasoning` string

Validation is performed using a strict JSON Schema. See the full reference in [Submission Schema](../submission_schema.md).

## Generate submissions

You can generate a submission either:

- From a saved report using `generate-submission`, or
- Directly during evaluation using `--export-submission`.

### From a saved report

```bash
# Include reasoning traces (default)
python -m bench.cli_typer generate-submission \
  --run-id <run-id> \
  --results-dir ./results \
  --out submission.json \
  --include-reasoning

# Or explicitly disable reasoning traces
python -m bench.cli_typer generate-submission \
  --run-id <run-id> \
  --results-dir ./results \
  --out submission.json \
  --no-include-reasoning
```

The tool tries `./results/<run-id>.json` first, then scans `--results-dir` for a report whose `metadata.run_id` matches.

### During evaluation (in-memory)

```bash
python -m bench.cli_typer evaluate <model-id> \
  --tasks <task-id> \
  --tasks-dir bench/tasks \
  --output-dir results \
  --format json --save-results \
  --export-submission results/submission.json \
  --export-submission-include-reasoning
```

This writes a validated submission JSON in the specified path without reloading the saved report.

## Validate programmatically

```python
from bench.leaderboard.submission import build_and_validate_submission
from bench.models.benchmark_report import BenchmarkReport

report = BenchmarkReport.from_file("results/<run-id>.json")
payload = build_and_validate_submission(report, include_reasoning=True)
# If no exception is raised, the payload passes validation
```

## Uploading the file

Once generated, upload the `submission.json` file to the leaderboard portal following the competition or benchmark instructions.

- Ensure your file size and naming follow portal guidelines (if any).
- Keep a copy of the original report `<run-id>.json` for auditability.
- If the portal exposes a REST API, you can automate this step in CI with an authenticated `POST`.

## Troubleshooting

- If validation fails, the CLI prints a clear message. Fix the offending field indicated.
- If the portal rejects the file:
  - Re-validate locally (`python -m json.tool submission.json` and rerun the MedAISure validator via the CLI)
  - Confirm the portal’s submission window and allowed file names
  - Re-check that your predictions conform to the expected shape for the tasks you’ve evaluated
