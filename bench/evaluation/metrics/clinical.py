"""Concrete metric implementations for MedAISure metrics system.

Implements:
- ClinicalAccuracyMetric (aka clinical correctness on answers)
- ReasoningQualityMetric (token-overlap F1 on rationales)
- DiagnosticAccuracyMetric (label accuracy)
- ClinicalRelevanceMetric (token Jaccard between summary and note)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import Metric


def _normalize_text(s: Any) -> str:
    if s is None:
        return ""
    text = str(s).lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\.,!?;:\-\(\)\[\]\{\}\'\"]", "", text)
    return text


def _tokenize(s: Any) -> List[str]:
    n = _normalize_text(s)
    return n.split() if n else []


class ClinicalAccuracyMetric(Metric):
    @property
    def name(self) -> str:
        return "clinical_accuracy"

    def calculate(
        self, expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> float:
        Metric.validate_inputs(expected_outputs, model_outputs)
        scores: List[float] = []
        for ref, pred in zip(expected_outputs, model_outputs):
            t = (ref or {}).get("answer")
            p = (
                (pred or {}).get("answer")
                or (pred or {}).get("prediction")
                or (pred or {}).get("text")
            )
            t_norm = _normalize_text(t)
            p_norm = _normalize_text(p)
            if not t_norm and not p_norm:
                scores.append(1.0)
            elif not t_norm or not p_norm:
                scores.append(0.0)
            elif t_norm == p_norm or (t_norm in p_norm) or (p_norm in t_norm):
                scores.append(1.0)
            else:
                scores.append(0.0)
        return float(sum(scores) / len(scores)) if scores else float("nan")


class ReasoningQualityMetric(Metric):
    @property
    def name(self) -> str:
        return "reasoning_quality"

    def calculate(
        self, expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> float:
        Metric.validate_inputs(expected_outputs, model_outputs)

        def f1(a: Any, b: Any) -> float:
            a_tokens = _tokenize(a)
            b_tokens = _tokenize(b)
            if not a_tokens and not b_tokens:
                return 1.0
            if not a_tokens or not b_tokens:
                return 0.0
            common = 0
            b_count: Dict[str, int] = {}
            for tok in b_tokens:
                b_count[tok] = b_count.get(tok, 0) + 1
            for tok in a_tokens:
                if b_count.get(tok, 0) > 0:
                    common += 1
                    b_count[tok] -= 1
            precision = common / len(b_tokens)
            recall = common / len(a_tokens)
            return (
                0.0
                if (precision + recall) == 0
                else 2 * precision * recall / (precision + recall)
            )

        scores: List[float] = []
        for ref, pred in zip(expected_outputs, model_outputs):
            r = (ref or {}).get("rationale") or (ref or {}).get("explanation")
            p = (
                (pred or {}).get("rationale")
                or (pred or {}).get("explanation")
                or (pred or {}).get("prediction")
                or (pred or {}).get("text")
            )
            scores.append(f1(r, p))
        return float(sum(scores) / len(scores)) if scores else float("nan")


class DiagnosticAccuracyMetric(Metric):
    @property
    def name(self) -> str:
        return "diagnostic_accuracy"

    def calculate(
        self, expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> float:
        Metric.validate_inputs(expected_outputs, model_outputs)
        # Map labels to ints if needed
        y_true: List[Any] = []
        y_pred: List[Any] = []
        for ref, pred in zip(expected_outputs, model_outputs):
            y_true.append((ref or {}).get("label") or (ref or {}).get("diagnosis"))
            y_pred.append((pred or {}).get("label") or (pred or {}).get("prediction"))
        if not y_true or not y_pred:
            return float("nan")
        if isinstance(y_true[0], str) or isinstance(y_pred[0], str):
            labels = sorted(
                set(
                    [x for x in y_true if x is not None]
                    + [x for x in y_pred if x is not None]
                )
            )
            mapping = {lab: i for i, lab in enumerate(labels)}
            y_true = [mapping.get(v, -1) if v is not None else -1 for v in y_true]
            y_pred = [mapping.get(v, -1) if v is not None else -1 for v in y_pred]
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return float(correct / len(y_true)) if y_true else float("nan")


class ClinicalRelevanceMetric(Metric):
    @property
    def name(self) -> str:
        return "clinical_relevance"

    def calculate(
        self, expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> float:
        Metric.validate_inputs(expected_outputs, model_outputs)
        scores: List[float] = []
        for ref, pred in zip(expected_outputs, model_outputs):
            t = (ref or {}).get("note")
            p = (
                (pred or {}).get("summary")
                or (pred or {}).get("prediction")
                or (pred or {}).get("text")
            )
            set_t = set(_tokenize(t))
            set_p = set(_tokenize(p))
            if not set_t and not set_p:
                scores.append(1.0)
            elif not set_t or not set_p:
                scores.append(0.0)
            else:
                inter = len(set_t & set_p)
                union = len(set_t | set_p)
                scores.append(inter / union if union else 0.0)
        return float(sum(scores) / len(scores)) if scores else float("nan")
