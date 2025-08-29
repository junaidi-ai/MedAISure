from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import csv
import json

from .medical_task import MedicalTask, TaskType


def _read_json_or_csv(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    if p.suffix.lower() == ".csv":
        with p.open("r", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError("Unsupported data file format. Use .json or .csv")


class MedicalQATask(MedicalTask):
    """Simple Medical QA task with convenience loaders and basic metrics.

    Expected schemas:
    - input_schema.required: ["question"]
    - output_schema.required: ["answer"]
    Dataset rows may be one of:
    - {"input": {"question": ...}, "output": {"answer": ...}}
    - {"question": ..., "answer": ...}
    """

    def __init__(
        self, task_id: str, description: str = "", *, name: Optional[str] = None
    ):
        super().__init__(
            task_id=task_id,
            task_type=TaskType.QA,
            name=name or task_id,
            description=description,
            inputs=[{"question": ""}],
            expected_outputs=[{"answer": ""}],
            metrics=["accuracy", "clinical_correctness"],
            input_schema={"required": ["question"]},
            output_schema={"required": ["answer"]},
            dataset=[],
        )

    def load_data(self, data_path: str | Path) -> None:
        rows = _read_json_or_csv(data_path)
        dataset: List[Dict[str, Any]] = []
        for r in rows:
            if "input" in r or "output" in r:
                inp = r.get("input", {})
                out = r.get("output", {})
            else:
                inp = {"question": r.get("question", "")}
                out = {"answer": r.get("answer", "")}
            dataset.append({"input": inp, "output": out})
        self.dataset = dataset
        # Derive top-level inputs/expected_outputs examples (aligned)
        self.inputs = [row["input"] for row in dataset]
        self.expected_outputs = [row["output"] for row in dataset]

    def evaluate(self, model_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        # Exact-match accuracy (case-insensitive, stripped)
        gold = [o.get("answer", "") for o in (self.expected_outputs or [])]
        pred = [o.get("answer", "") for o in (model_outputs or [])]
        n = min(len(gold), len(pred))
        if n == 0:
            return {"accuracy": 0.0, "clinical_correctness": 0.0}
        correct = 0
        for g, p in zip(gold[:n], pred[:n]):
            if str(g).strip().lower() == str(p).strip().lower():
                correct += 1
        acc = correct / float(n)
        # For simplicity, mirror accuracy as clinical_correctness placeholder
        return {"accuracy": acc, "clinical_correctness": acc}


class DiagnosticReasoningTask(MedicalTask):
    """Diagnostic reasoning task.

    Expected schemas:
    - input_schema.required: ["case"]
    - output_schema.required: ["diagnosis"]
    """

    def __init__(
        self, task_id: str, description: str = "", *, name: Optional[str] = None
    ):
        super().__init__(
            task_id=task_id,
            task_type=TaskType.DIAGNOSTIC_REASONING,
            name=name or task_id,
            description=description,
            inputs=[{"case": ""}],
            expected_outputs=[{"diagnosis": ""}],
            metrics=["diagnostic_accuracy", "reasoning_quality"],
            input_schema={"required": ["case"]},
            output_schema={"required": ["diagnosis"]},
            dataset=[],
        )

    def load_data(self, data_path: str | Path) -> None:
        rows = _read_json_or_csv(data_path)
        dataset: List[Dict[str, Any]] = []
        for r in rows:
            if "input" in r or "output" in r:
                inp = r.get("input", {})
                out = r.get("output", {})
            else:
                inp = {"case": r.get("case", "")}
                out = {"diagnosis": r.get("diagnosis", "")}
            dataset.append({"input": inp, "output": out})
        self.dataset = dataset
        self.inputs = [row["input"] for row in dataset]
        self.expected_outputs = [row["output"] for row in dataset]

    def evaluate(self, model_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        gold = [o.get("diagnosis", "") for o in (self.expected_outputs or [])]
        pred = [o.get("diagnosis", "") for o in (model_outputs or [])]
        n = min(len(gold), len(pred))
        if n == 0:
            return {"diagnostic_accuracy": 0.0, "reasoning_quality": 0.0}
        correct = 0
        for g, p in zip(gold[:n], pred[:n]):
            if str(g).strip().lower() == str(p).strip().lower():
                correct += 1
        acc = correct / float(n)
        # Naive proxy for reasoning quality: proportion of outputs containing a justification cue
        cues = ["because", "due to", "based on", "given"]
        reason_hits = 0
        for o in (model_outputs or [])[:n]:
            txt = (o.get("explanation") or o.get("rationale") or "").lower()
            if any(c in txt for c in cues):
                reason_hits += 1
        rq = reason_hits / float(n)
        return {"diagnostic_accuracy": acc, "reasoning_quality": rq}


class ClinicalSummarizationTask(MedicalTask):
    """Clinical summarization task with lightweight metrics.

    Expected schemas:
    - input_schema.required: ["document"]
    - output_schema.required: ["summary"]
    """

    def __init__(
        self, task_id: str, description: str = "", *, name: Optional[str] = None
    ):
        super().__init__(
            task_id=task_id,
            task_type=TaskType.SUMMARIZATION,
            name=name or task_id,
            description=description,
            inputs=[{"document": ""}],
            expected_outputs=[{"summary": ""}],
            metrics=["rouge_l", "clinical_relevance", "factual_consistency"],
            input_schema={"required": ["document"]},
            output_schema={"required": ["summary"]},
            dataset=[],
        )

    def load_data(self, data_path: str | Path) -> None:
        rows = _read_json_or_csv(data_path)
        dataset: List[Dict[str, Any]] = []
        for r in rows:
            if "input" in r or "output" in r:
                inp = r.get("input", {})
                out = r.get("output", {})
            else:
                # Accept aliases: 'text' or 'note' map to 'document'
                doc = r.get("document") or r.get("text") or r.get("note") or ""
                inp = {"document": doc}
                out = {"summary": r.get("summary", "")}
            dataset.append({"input": inp, "output": out})
        self.dataset = dataset
        self.inputs = [row["input"] for row in dataset]
        self.expected_outputs = [row["output"] for row in dataset]

    @staticmethod
    def _unigrams(text: str) -> List[str]:
        return [t for t in (text or "").lower().split() if t]

    def evaluate(self, model_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
        refs = [o.get("summary", "") for o in (self.expected_outputs or [])]
        hyps = [o.get("summary", "") for o in (model_outputs or [])]
        n = min(len(refs), len(hyps))
        if n == 0:
            return {
                "rouge_l": 0.0,
                "clinical_relevance": 0.0,
                "factual_consistency": 0.0,
            }
        r1_total = 0.0
        rel_total = 0.0
        fact_total = 0.0
        med_keywords = {
            "patient",
            "diagnosis",
            "treatment",
            "history",
            "medication",
            "lab",
            "symptom",
        }
        for r, h in zip(refs[:n], hyps[:n]):
            r_uni = self._unigrams(r)
            h_uni = self._unigrams(h)
            if r_uni:
                overlap = sum(1 for t in set(r_uni) if t in set(h_uni))
                r1_total += overlap / float(len(set(r_uni)))
            # Clinical relevance: keyword overlap ratio
            rel_overlap = sum(1 for k in med_keywords if k in set(h_uni))
            rel_total += rel_overlap / max(1.0, float(len(med_keywords)))
            # Factual consistency placeholder: penalize hallucinated numbers not in ref
            ref_nums = {tok for tok in r_uni if tok.isdigit()}
            hyp_nums = {tok for tok in h_uni if tok.isdigit()}
            bad = len(hyp_nums - ref_nums)
            fact_total += max(
                0.0, 1.0 - (bad / max(1.0, float(len(hyp_nums))) if hyp_nums else 0.0)
            )
        return {
            "rouge_l": r1_total / n,
            "clinical_relevance": rel_total / n,
            "factual_consistency": fact_total / n,
        }
