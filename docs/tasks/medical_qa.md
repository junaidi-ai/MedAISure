# Medical QA

Purpose: Evaluate short-form medical question answering across clinical knowledge domains.

- Input: question text, optional context
- Output: concise answer (text)
- Dataset examples: synthetic and curated clinical QA
- Scoring: see Metrics -> Clinical Accuracy, Reasoning Quality

Implementation
- Task ID: medical_qa
- Loader: see `bench/examples/` for templates
- Expected fields: {id, prompt, answer, metadata}

Evaluation Tips
- Encourage direct, unambiguous answers
- Prefer evidence-based statements with citations if available
- Penalize hallucinations and unsafe advice

References
- Docs: `docs/usage.md`, `docs/metrics/clinical_accuracy.md`
