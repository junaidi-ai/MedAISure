# Diagnostic Reasoning

Purpose: Assess stepwise clinical reasoning from presentation to differential and final diagnosis.

- Input: clinical vignette (structured or free text)
- Output: reasoning steps + differential + most likely diagnosis
- Format: chain-of-thought optional; require a final explicit diagnosis line

Implementation
- Task ID: diagnostic_reasoning
- Expected fields: {id, case_text, final_diagnosis, reasoning, metadata}
- Optional: require structured sections (History, Exam, Labs, Imaging)

Evaluation
- Reasoning Quality (structure, clinical plausibility)
- Clinical Accuracy (final diagnosis correctness)

Guidelines
- Avoid definitive claims when uncertainty remains
- Include safety considerations and red flags

See also: `docs/metrics/reasoning_quality.md`
