from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore[assignment]

from bench.evaluation.model_interface import LocalModel


class EchoPredict:
    """Trivial model exposing a predict() method."""

    def predict(self, batch: List[Dict[str, Any]], **kwargs: Any):
        return [{"len": len(x), "ok": True} for x in batch]


def main() -> None:
    if joblib is None:
        print("joblib not installed; skipping example")
        return

    out_dir = Path("./.local_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / "echo.joblib"

    # Save via joblib
    joblib.dump(EchoPredict(), str(model_file))

    # Load and run via LocalModel
    lm = LocalModel(model_path=str(model_file))
    inputs = [{"a": 1}, {"bbb": 2}]
    outputs = lm.predict(inputs)

    print("Outputs:", outputs)
    print("Metadata:", lm.metadata)


if __name__ == "__main__":
    main()
