"""
Minimal local model example for MedAISure.
Exposes load_model(model_path, **kwargs) used by ModelRunner when model_type="local".
"""

from typing import Any, Dict, List


class MyLocalModel:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(
        self, batch: List[Dict[str, Any]], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        # Echoes a trivial prediction for each input. Replace with real logic.
        out: List[Dict[str, Any]] = []
        for _ in batch:
            out.append({"label": "entailment", "score": 0.9})
        return out


def load_model(model_path: str | None = None, **kwargs: Any) -> MyLocalModel:
    """Factory function discovered by ModelRunner for local models."""
    return MyLocalModel(**kwargs)
