# FAQ

## How do I generate a leaderboard submission?

- Use the Typer CLI:
  - During evaluation: `evaluate --export-submission ...`
  - From a saved report: `generate-submission --run-id <id> --out submission.json`
- See the [Usage Guide](usage.md#end-to-end-evaluate-and-export-leaderboard-submission) for full examples.

## Where can I find the submission schema?

- Read the docs page: [Submission Schema](submission_schema.md)
- Download the JSON Schema: [schema/submission.schema.json](schema/submission.schema.json)

## Why is my submission rejected?

- Run local validation using the exporter (it uses `jsonschema` when installed):
  - CLI validates automatically before writing
  - Programmatic: `build_and_validate_submission(report)` will raise with details
- See common issues in [CLI Troubleshooting](api/cli.md#troubleshooting).

## I get "Could not find report for run_id" errors

- Confirm `<run-id>.json` exists under `--results-dir` (default: `./results`)
- Or ensure a report with matching `metadata.run_id` exists (the CLI will scan the directory)

## How do I include/exclude reasoning traces?

- Flags:
  - `--include-reasoning` (default enabled)
  - `--no-include-reasoning` to disable

## What model types are supported?

- `huggingface`, `local`, and `api` via `ModelRunner`
- See [Models Guide](models/model_interface.md)

## How do I compute a combined score across categories?

- CLI flags during evaluate: `--combined-weights` and `--combined-metric-name`
- Config-based alternative available; see [CLI combined score](api/cli.md#combined-score-via-cli-typer)

## How can I preview datasets and registries?

- CLI commands:
  - `list-datasets` to list registry entries (optionally with composition)
  - `show-dataset <id>` to view one entry

## Where can I report issues or request features?

- Open an issue on GitHub: https://github.com/junaidi-ai/MedAISure/issues
