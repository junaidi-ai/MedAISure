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

# ------------------------- Shared Synonyms -------------------------
# Central map for label synonyms used across metrics as needed.
LABEL_SYNONYMS: Dict[str, str] = {
    # cardiology
    "mi": "myocardial infarction",
    "hf": "heart failure",
    "chf": "heart failure",
    "cad": "coronary artery disease",
    "af": "atrial fibrillation",
    "htn": "hypertension",
    # neurology
    "cva": "stroke",
    "tia": "transient ischemic attack",
    # pulmonary
    "copd": "chronic obstructive pulmonary disease",
    "pna": "pneumonia",
    "pe": "pulmonary embolism",
    # vascular/hematology
    "dvt": "deep vein thrombosis",
    # renal
    "aki": "acute kidney injury",
    "ckd": "chronic kidney disease",
    # endocrine/infectious
    "dm": "diabetes mellitus",
    "uti": "urinary tract infection",
    # other common
    "hld": "hyperlipidemia",
    "arf": "acute renal failure",
    "pud": "peptic ulcer disease",
    "afib": "atrial fibrillation",
    "gerd": "gastroesophageal reflux disease",
}


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
    """Clinical correctness on answers using simple entity overlap.

    This implementation:
    - Extracts naive clinical entities from text into categories
      (diagnoses, treatments, medications, symptoms, labs) using
      lightweight lexicons and normalization.
    - Computes a weighted Jaccard similarity across categories.
    - Normalizes the final score to [0, 1].
    - Stores a per-pair breakdown retrievable via
      ``get_last_breakdown()`` without altering the base Metric API.

    Notes:
    - If no entities are found in both reference and prediction, we
      fall back to a conservative normalized string match (exact/substring)
      to avoid returning NaN and to keep intuitive behavior for free-form
      short answers.
    - This is intentionally simple and dependency-light. It can be
      upgraded later to use proper clinical NLP.
    """

    # Lightweight category weights; adjust as needed
    _CATEGORY_WEIGHTS = {
        "diagnoses": 3.0,
        "treatments": 2.0,
        "medications": 2.0,
        "symptoms": 1.0,
        "labs": 1.0,
    }

    # Minimal lexicon per category (lowercased, normalized)
    _LEXICON = {
        "diagnoses": {
            "pneumonia",
            "sepsis",
            "myocardial infarction",
            "mi",
            "stroke",
            "copd",
            "asthma",
            "heart failure",
            "diabetes",
        },
        "treatments": {
            "antibiotics",
            "oxygen",
            "ventilation",
            "iv fluids",
            "insulin",
            "surgery",
        },
        "medications": {
            "amoxicillin",
            "ceftriaxone",
            "azithromycin",
            "heparin",
            "aspirin",
            "insulin",
        },
        "symptoms": {
            "fever",
            "cough",
            "shortness of breath",
            "dyspnea",
            "tachycardia",
            "hypotension",
            "chest pain",
        },
        "labs": {
            "wbc",
            "white blood cell",
            "lactate",
            "troponin",
            "glucose",
        },
    }

    # Simple synonym/normalization map
    _NORMALIZE_MAP = {
        "mi": "myocardial infarction",
        "sob": "shortness of breath",
        "white blood cells": "white blood cell",
        "leukocytosis": "white blood cell",
    }

    def __init__(self) -> None:
        self._last_breakdown: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "clinical_accuracy"

    def calculate(
        self, expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> float:
        Metric.validate_inputs(expected_outputs, model_outputs)
        self._last_breakdown = []
        scores: List[float] = []

        for ref, pred in zip(expected_outputs, model_outputs):
            # Extract free-text answers
            t = (ref or {}).get("answer")
            p = (
                (pred or {}).get("answer")
                or (pred or {}).get("prediction")
                or (pred or {}).get("text")
            )
            t_norm = _normalize_text(t)
            p_norm = _normalize_text(p)

            # Entity extraction by category
            t_entities = self._extract_entities(t_norm)
            p_entities = self._extract_entities(p_norm)

            # Compute weighted Jaccard across categories
            cat_scores: Dict[str, float] = {}
            weighted_inter = 0.0
            weighted_union = 0.0
            for cat, weight in self._CATEGORY_WEIGHTS.items():
                t_set = t_entities.get(cat, set())
                p_set = p_entities.get(cat, set())
                inter = len(t_set & p_set)
                union = len(t_set | p_set)
                if union:
                    cat_scores[cat] = inter / union
                    weighted_inter += weight * inter
                    weighted_union += weight * union
                else:
                    cat_scores[cat] = 1.0 if not t_set and not p_set else 0.0

            if weighted_union > 0:
                score = weighted_inter / weighted_union
            else:
                # Fallback to simple normalized string comparison when
                # there are no recognized entities
                if not t_norm and not p_norm:
                    score = 1.0
                elif not t_norm or not p_norm:
                    score = 0.0
                elif t_norm == p_norm or (t_norm in p_norm) or (p_norm in t_norm):
                    score = 1.0
                else:
                    score = 0.0

            scores.append(score)
            self._last_breakdown.append(
                {
                    "reference": t_norm,
                    "prediction": p_norm,
                    "entities_ref": {k: sorted(v) for k, v in t_entities.items()},
                    "entities_pred": {k: sorted(v) for k, v in p_entities.items()},
                    "category_scores": cat_scores,
                    "pair_score": score,
                }
            )

        return float(sum(scores) / len(scores)) if scores else float("nan")

    def get_last_breakdown(self) -> List[Dict[str, Any]]:
        """Return per-sample breakdown from the most recent calculate() call."""
        return self._last_breakdown

    # ------------------------- helpers -------------------------
    def _normalize_term(self, term: str) -> str:
        t = term.strip()
        if not t:
            return t
        t = self._NORMALIZE_MAP.get(t, t)
        return t

    def _extract_entities(self, text: str) -> Dict[str, set]:
        entities: Dict[str, set] = {k: set() for k in self._CATEGORY_WEIGHTS.keys()}
        if not text:
            return entities

        # Detect multi-word terms first to avoid splitting
        remaining = text
        # Build a list of multi-word lexicon terms by category
        multi_terms: List[tuple[str, str]] = []  # (term, category)
        for cat, terms in self._LEXICON.items():
            for t in terms:
                if " " in t:
                    multi_terms.append((t, cat))

        for term, cat in multi_terms:
            if term in remaining:
                entities[cat].add(self._normalize_term(term))
                # naive removal to reduce double counting
                remaining = remaining.replace(term, " ")

        # Single-word detection from remaining tokens
        for tok in _tokenize(remaining):
            tok = self._normalize_term(tok)
            for cat, terms in self._LEXICON.items():
                if tok in terms:
                    entities[cat].add(tok)

        # Drop empty categories to keep breakdown concise
        return {k: v for k, v in entities.items() if v}


class ReasoningQualityMetric(Metric):
    """Token-overlap F1 plus lightweight reasoning heuristics.

    Components:
    - overlap_f1: token overlap between reference and predicted rationale
    - structure_score: presence of structure markers (because, therefore, steps)
    - evidence_score: mentions of evidence (labs, imaging, vitals, findings)
    - factual_consistency: simple rule checks against common clinical facts
    - fallacy_penalty: heuristic detection of common fallacy phrases

    Final score is a weighted combination clamped to [0,1].
    A per-sample breakdown is available via get_last_breakdown().
    """

    _W_WEIGHTS = {
        "overlap_f1": 0.4,
        "structure": 0.15,
        "evidence": 0.2,
        "factual": 0.25,
    }

    _STRUCTURE_MARKERS = {
        "because",
        "therefore",
        "thus",
        "hence",
        "so",
        "in conclusion",
        "first",
        "second",
        "third",
        "1.",
        "2.",
        "3.",
        "step",
    }

    _EVIDENCE_MARKERS = {
        "lab",
        "wbc",
        "lactate",
        "troponin",
        "imaging",
        "xray",
        "x-ray",
        "ct",
        "scan",
        "vitals",
        "fever",
        "tachycardia",
        "hypotension",
        "finding",
        "study",
        "source",
        "evidence",
    }

    _FALLACY_PHRASES = {
        "correlation implies causation",
        "post hoc",
        "begs the question",
        "circular reasoning",
        "non sequitur",
        "because i say",
        "everyone knows",
        "obviously true",
    }

    # minimal factual rules: diagnosis -> expected treatment/evidence keywords
    _FACTS = {
        "pneumonia": {"antibiotic", "antibiotics", "xray", "x-ray", "infiltrate"},
        "sepsis": {"iv fluids", "fluids", "lactate", "hypotension"},
        "myocardial infarction": {"troponin", "ekg", "aspirin", "heparin"},
        "mi": {"troponin", "ekg", "aspirin", "heparin"},
    }

    def __init__(self) -> None:
        self._last_breakdown: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "reasoning_quality"

    def calculate(
        self, expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> float:
        Metric.validate_inputs(expected_outputs, model_outputs)
        self._last_breakdown = []

        scores: List[float] = []
        for ref, pred in zip(expected_outputs, model_outputs):
            r = (ref or {}).get("rationale") or (ref or {}).get("explanation")
            p = (
                (pred or {}).get("rationale")
                or (pred or {}).get("explanation")
                or (pred or {}).get("prediction")
                or (pred or {}).get("text")
            )

            f1_score = self._f1(r, p)
            structure = self._structure_score(p)
            evidence = self._evidence_score(p)
            factual = self._factual_consistency_score(
                (ref or {}).get("answer") or (ref or {}).get("diagnosis"), p
            )
            fallacy_pen = self._fallacy_penalty(p)

            weighted = (
                self._W_WEIGHTS["overlap_f1"] * f1_score
                + self._W_WEIGHTS["structure"] * structure
                + self._W_WEIGHTS["evidence"] * evidence
                + self._W_WEIGHTS["factual"] * factual
            )
            score = max(0.0, min(1.0, weighted - fallacy_pen))
            scores.append(score)

            self._last_breakdown.append(
                {
                    "reference_rationale": _normalize_text(r),
                    "predicted_rationale": _normalize_text(p),
                    "overlap_f1": f1_score,
                    "structure_score": structure,
                    "evidence_score": evidence,
                    "factual_consistency": factual,
                    "fallacy_penalty": fallacy_pen,
                    "final_score": score,
                }
            )

        return float(sum(scores) / len(scores)) if scores else float("nan")

    def get_last_breakdown(self) -> List[Dict[str, Any]]:
        return self._last_breakdown

    # ----------------------- helpers -----------------------
    def _f1(self, a: Any, b: Any) -> float:
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

    def _structure_score(self, text: Any) -> float:
        t = _normalize_text(text)
        if not t:
            return 0.0
        hits = sum(1 for m in self._STRUCTURE_MARKERS if m in t)
        # cap at 3 for diminishing returns
        return min(1.0, hits / 3.0)

    def _evidence_score(self, text: Any) -> float:
        t = _normalize_text(text)
        if not t:
            return 0.0
        hits = sum(1 for m in self._EVIDENCE_MARKERS if m in t)
        return min(1.0, hits / 3.0)

    def _fallacy_penalty(self, text: Any) -> float:
        t = _normalize_text(text)
        if not t:
            return 0.0
        hits = sum(1 for m in self._FALLACY_PHRASES if m in t)
        # each hit penalizes modestly, capped
        return min(0.4, 0.15 * hits)

    def _factual_consistency_score(self, diagnosis: Any, text: Any) -> float:
        diag = _normalize_text(diagnosis)
        t = _normalize_text(text)
        if not t:
            return 0.0
        if not diag:
            # if unknown diagnosis, score based on generic evidence mentions
            return self._evidence_score(t)
        expected = set()
        for key, facts in self._FACTS.items():
            if key in diag:
                expected |= facts
        if not expected:
            return self._evidence_score(t)
        matches = sum(1 for kw in expected if kw in t)
        return min(1.0, matches / max(1, len(expected) // 2))


class DiagnosticAccuracyMetric(Metric):
    def _normalize_label(self, s: Any) -> str:
        """Normalize diagnosis/label text, including simple synonyms.

        Includes simple mappings for common abbreviations after base normalization.
        """
        t = _normalize_text(s)
        return LABEL_SYNONYMS.get(t, t)

    @property
    def name(self) -> str:
        return "diagnostic_accuracy"

    def calculate(
        self, expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> float:
        Metric.validate_inputs(expected_outputs, model_outputs)

        # Domain knowledge and specialty config
        self._last_breakdown: List[Dict[str, Any]] = []
        KNOWN_LABELS: Dict[str, set] = {
            "cardiology": {"myocardial infarction", "mi", "heart failure"},
            "infectious_disease": {"sepsis", "pneumonia", "flu"},
        }
        CRITICAL_LABELS: Dict[str, set] = {
            "cardiology": {"myocardial infarction", "mi"},
            "infectious_disease": {"sepsis"},
        }

        # Weighted average to emphasize critical errors
        total_weight = 0.0
        weighted_sum = 0.0
        for ref, pred in zip(expected_outputs, model_outputs):
            exp_label = (ref or {}).get("label") or (ref or {}).get("diagnosis")
            pred_label = (pred or {}).get("label") or (pred or {}).get("prediction")
            specialty = (ref or {}).get("specialty") or (pred or {}).get("specialty")

            # Normalize text labels for comparison (with synonym mapping)
            t_norm = self._normalize_label(exp_label)
            p_norm = self._normalize_label(pred_label)

            # Base correctness
            base = 1.0 if (t_norm and p_norm and t_norm == p_norm) else 0.0

            # Specialty-based weight and knowledge checks
            weight = 1.0
            known = True
            critical = False
            if specialty:
                s = _normalize_text(specialty)
                if s in KNOWN_LABELS:
                    known = (p_norm in KNOWN_LABELS[s]) if p_norm else False
                if s in CRITICAL_LABELS:
                    critical = t_norm in CRITICAL_LABELS[s]
                    if critical:
                        weight = 1.5  # emphasize critical diagnoses

            # Penalize unknown label predictions slightly
            if base == 0.0 and not known:
                weight *= 1.0  # keep simple; weight influences average

            total_weight += weight
            weighted_sum += weight * base

            self._last_breakdown.append(
                {
                    "expected": t_norm,
                    "predicted": p_norm,
                    "specialty": _normalize_text(specialty) if specialty else None,
                    "known_pred_label": known,
                    "critical_expected": critical,
                    "weight": weight,
                    "correct": base,
                }
            )

        return float(weighted_sum / total_weight) if total_weight > 0 else float("nan")

    def get_last_breakdown(self) -> List[Dict[str, Any]]:
        return getattr(self, "_last_breakdown", [])


class ClinicalRelevanceMetric(Metric):
    @property
    def name(self) -> str:
        return "clinical_relevance"

    def calculate(
        self, expected_outputs: List[Dict], model_outputs: List[Dict]
    ) -> float:
        Metric.validate_inputs(expected_outputs, model_outputs)
        self._last_breakdown: List[Dict[str, Any]] = []

        # Domain thresholds by specialty (soft adjustment)
        DOMAIN_THRESHOLDS: Dict[str, float] = {
            "cardiology": 0.35,
            "infectious_disease": 0.30,
        }

        scores: List[float] = []
        for ref, pred in zip(expected_outputs, model_outputs):
            t = (ref or {}).get("note")
            p = (
                (pred or {}).get("summary")
                or (pred or {}).get("prediction")
                or (pred or {}).get("text")
            )
            specialty = (ref or {}).get("specialty") or (pred or {}).get("specialty")

            set_t = set(_tokenize(t))
            set_p = set(_tokenize(p))

            # Base Jaccard
            if not set_t and not set_p:
                base = 1.0
            elif not set_t or not set_p:
                base = 0.0
            else:
                inter = len(set_t & set_p)
                union = len(set_t | set_p)
                base = inter / union if union else 0.0

            # Specialty knowledge boost: reward presence of key medical entities
            # using the ClinicalAccuracyMetric lexicon where relevant
            boost = 0.0
            acc_lex = ClinicalAccuracyMetric._LEXICON  # reuse
            # count shared clinically relevant tokens
            shared_keywords = 0
            for terms in acc_lex.values():
                for term in terms:
                    if " " in term:
                        continue
                    if term in set_t and term in set_p:
                        shared_keywords += 1
            if shared_keywords:
                boost = min(0.1, shared_keywords * 0.02)

            raw = min(1.0, base + boost)

            # Apply soft domain threshold adjustment
            thr = 0.25
            if specialty:
                s = _normalize_text(specialty)
                thr = DOMAIN_THRESHOLDS.get(s, thr)
            meets = raw >= thr
            adjusted = raw if meets else max(0.0, raw * 0.8)

            scores.append(adjusted)
            self._last_breakdown.append(
                {
                    "specialty": _normalize_text(specialty) if specialty else None,
                    "base_jaccard": base,
                    "knowledge_boost": boost,
                    "raw_score": raw,
                    "threshold": thr,
                    "meets_threshold": meets,
                    "final_score": adjusted,
                }
            )

        return float(sum(scores) / len(scores)) if scores else float("nan")

    def get_last_breakdown(self) -> List[Dict[str, Any]]:
        return getattr(self, "_last_breakdown", [])
