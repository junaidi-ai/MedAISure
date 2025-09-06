"""Multimodal evaluation hooks (DISABLED scaffold).

These functions outline the intended flow for multimodal evaluation,
including input preparation, model execution, and evaluation. The module
is intentionally disabled; calling any function will raise to avoid
accidental activation in current runtime.
"""

from __future__ import annotations

from typing import Any, Dict, List

DISABLED: bool = True


def _guard() -> None:
    if DISABLED:
        raise NotImplementedError(
            "Multimodal hooks are disabled in this scaffold. Enable explicitly before use."
        )


def prepare_inputs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Load images and tokenize text for multimodal models (outline).

    Expected record shapes include keys like `question`/`document` and
    `image_path` or `image_url`.
    """
    _guard()
    return []


def run_model(model: Any, prepared: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run a multimodal model over prepared inputs (outline)."""
    _guard()
    return []


def evaluate(
    golds: List[Dict[str, Any]], preds: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute multimodal metrics (outline)."""
    _guard()
    return {}
