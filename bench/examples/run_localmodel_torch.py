from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from bench.evaluation.model_interface import LocalModel


class TorchLike:
    """Callable that mimics a small torch model."""

    def __call__(self, batch: List[Dict[str, Any]], **kwargs: Any):
        # Pretend we run a forward pass and return a score
        return [
            {"score": float(i) / max(1, len(batch) - 1)} for i, _ in enumerate(batch)
        ]


def main() -> None:
    if torch is None:
        print("torch not installed; skipping example")
        return

    out_dir = Path("./.local_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / "echo.pt"

    # Save via torch (uses pickle under the hood for Python objects)
    torch.save(TorchLike(), str(model_file))

    # Load and run via LocalModel
    lm = LocalModel(model_path=str(model_file))
    inputs = [{"x": 0.1}, {"x": 0.9}, {"x": 1.7}]
    outputs = lm.predict(inputs)

    print("Outputs:", outputs)
    print("Metadata:", lm.metadata)


if __name__ == "__main__":
    main()
