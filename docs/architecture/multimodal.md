# Multimodal Support (Scaffold)

Status: Planned (scaffold only, disabled by default)

This document outlines the planned scaffolding for multimodal (text+image) tasks in MedAISure. These changes are intentionally non-invasive and disabled by default, serving as a design reference for Phase 3.

Goals:
- Draft schemas for multimodal inputs and outputs
- Provide placeholder task definitions that are not auto-discovered by the loader
- Outline evaluation path and metric hooks for future enablement
- Keep runtime unchanged until explicitly enabled

## Proposed Task Shapes

1) Image-Text QA (VQA-style)
- Input schema (required): `question`, `image_path` (or `image_url`)
- Output schema (required): `answer`

Example dataset row:
```yaml
input:
  question: "What abnormality is shown in the X-ray?"
  image_path: "data/images/example_chest_xray.png"
output:
  answer: "pneumonia"
```

2) Clinical Report With Image Context
- Input schema (required): `document`, `image_path`
- Output schema (required): `summary`

Example dataset row:
```yaml
input:
  document: "Patient presents with cough and fever."
  image_path: "data/images/example_chest_xray.png"
output:
  summary: "Findings consistent with pneumonia."
```

## Placeholder Task Definitions

To avoid runtime activation, placeholder YAMLs live in `bench/tasks/_multimodal/` and are not auto-discovered by `TaskLoader.discover_tasks()` (which only scans the top-level `bench/tasks/`).

Files:
- `bench/tasks/_multimodal/image_text_qa.yaml`
- `bench/tasks/_multimodal/clinical_report_with_image.yaml`

Each file marks `disabled: true` and includes minimal examples and schemas.

## Schemas (Draft)

Python constants are provided in `bench/models/multimodal_schemas.py`:
- `IMAGE_TEXT_QA_INPUT_SCHEMA = {"required": ["question", "image_path"]}`
- `IMAGE_TEXT_QA_OUTPUT_SCHEMA = {"required": ["answer"]}`
- `REPORT_WITH_IMAGE_INPUT_SCHEMA = {"required": ["document", "image_path"]}`
- `REPORT_WITH_IMAGE_OUTPUT_SCHEMA = {"required": ["summary"]}`

These are not imported anywhere by default.

## Evaluation Path (Outline)

A disabled hooks module `bench/evaluation/multimodal_hooks.py` provides stub entry points:
- `prepare_inputs(records)` — expected to load images and tokenize text
- `run_model(model, prepared)` — runs a multimodal model
- `evaluate(golds, preds)` — computes multimodal metrics

The module is guarded by `DISABLED = True` and raises `NotImplementedError` to prevent accidental use.

## Metric Hooks (Outline)

A stub `bench/evaluation/multimodal_metrics.py` defines placeholders for planned metrics:
- `image_text_alignment` (e.g., CLIP-style similarity proxy)
- `answer_grounding` (alignment of answer tokens with salient image regions)
- `report_consistency` (image-aware factual consistency for summaries)

These metrics are not registered in the current registry; they are documentation-only until explicitly enabled.

## Activation Plan (Future Work)

- Extend `TaskType` and validators to include a `multimodal` variant (or reuse existing types with image-aware schemas)
- Add secure, optional image loading utilities (local path + URL with timeouts)
- Implement metric registration behind a feature flag
- Provide sample models (e.g., `transformers` vision-text models) with opt-in loading

Until then, all changes are docs and disabled scaffolds to keep runtime stable.
