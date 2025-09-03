# Clinical Summarization

Purpose: Generate succinct, clinically relevant summaries from long-form notes or multi-modal inputs.

- Input: progress notes, discharge summaries, consult notes
- Output: targeted summary (e.g., Problem List, Assessment & Plan)
- Constraints: brevity, clinical salience, safety-critical details preserved

Implementation
- Task ID: clinical_summarization
- Expected fields: {id, source_text, target_summary, metadata}
- Style: bullet points preferred; avoid verbatim copy

Evaluation
- Content coverage and correctness (Clinical Accuracy)
- Structure and prioritization (Reasoning Quality)

See: `docs/metrics/clinical_accuracy.md`
