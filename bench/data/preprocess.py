from __future__ import annotations

from typing import Callable, List, Dict, Any


class DataPreprocessor:
    """
    Simple, composable preprocessing pipeline.

    Steps are callables that accept and return a dict.
    """

    def __init__(self) -> None:
        self.steps: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []

    def add_step(self, step: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self.steps.append(step)

    def process(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(data_item)
        for step in self.steps:
            result = step(result)
        return result

    def process_batch(self, data_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.process(item) for item in data_items]
