"""A very simple local model used for integration testing.

Implements a load_model(model_path, **kwargs) function that returns a callable
model. The callable accepts a list[dict] batch and returns a list[dict] with
keys: label, score.

Kwargs supported via load_model:
- default_label: str = "positive"  # label to emit if no rule matches
- label_map: dict[str,str] | None   # map raw_label->final label (applied by runner only for HF; here we keep raw)
- raise_on: list[str] | None        # if any input["text"] contains this substring, raise an error
- rules: list[tuple[str,str]] | None  # list of (substring, label)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class _Config:
    default_label: str = "positive"
    raise_on: Optional[List[str]] = None
    rules: Optional[List[Tuple[str, str]]] = None


class SimpleLocalModel:
    def __init__(self, config: _Config) -> None:
        self.config = config

    def __call__(
        self, batch: List[Dict[str, Any]], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for item in batch:
            text = str(item.get("text", ""))
            # Simulate failure for error propagation tests
            if self.config.raise_on:
                for needle in self.config.raise_on:
                    if needle in text:
                        raise RuntimeError(
                            f"Intentional error for text containing: {needle}"
                        )

            label = self.config.default_label
            if self.config.rules:
                for substring, rule_label in self.config.rules:
                    if substring in text:
                        label = rule_label
                        break
            outputs.append({"label": label, "score": 1.0})
        return outputs


def load_model(model_path: str, **kwargs: Any) -> SimpleLocalModel:
    # Build config from kwargs
    cfg = _Config(
        default_label=str(kwargs.get("default_label", "positive")),
        raise_on=list(kwargs.get("raise_on", []) or []) or None,
        rules=[tuple(x) for x in (kwargs.get("rules", []) or [])] or None,
    )
    return SimpleLocalModel(cfg)
