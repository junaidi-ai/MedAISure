from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

from bench.evaluation.model_interface import LocalModel


class EchoCallable:
    """Trivial callable model: returns index for each input in a dict."""

    def __call__(self, batch: List[Dict[str, Any]], **kwargs: Any):
        return [{"idx": i, "ok": True} for i, _ in enumerate(batch)]


def main() -> None:
    tmp_path = Path("./.local_models")
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_file = tmp_path / "echo.pkl"

    # Save a small callable model using pickle
    with model_file.open("wb") as f:
        pickle.dump(EchoCallable(), f)

    # Load and run via LocalModel
    lm = LocalModel(model_path=str(model_file))
    inputs = [{"text": "a"}, {"text": "b"}]
    outputs = lm.predict(inputs)

    print("Outputs:", outputs)
    print("Metadata:", lm.metadata)


if __name__ == "__main__":
    main()
