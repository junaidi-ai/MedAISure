"""Placeholder metrics for multimodal evaluation (scaffold only).

These functions are NOT registered in the current metric registry.
They serve as documentation for planned metrics.
"""

from __future__ import annotations

from typing import List

DISABLED: bool = True


def image_text_alignment(ref_texts: List[str], image_features: List[object]) -> float:
    """Proxy for CLIP-like alignment between text and image features."""
    if DISABLED:
        raise NotImplementedError("Multimodal metrics are disabled in this scaffold.")
    return 0.0


def answer_grounding(answers: List[str], saliency_maps: List[object]) -> float:
    """Measure alignment of answers with salient image regions (outline)."""
    if DISABLED:
        raise NotImplementedError("Multimodal metrics are disabled in this scaffold.")
    return 0.0


def report_consistency(ref_summaries: List[str], preds: List[str]) -> float:
    """Image-aware factual consistency for summaries (outline)."""
    if DISABLED:
        raise NotImplementedError("Multimodal metrics are disabled in this scaffold.")
    return 0.0
