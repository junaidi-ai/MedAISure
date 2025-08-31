from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Iterable, List, Any, Optional


class DatasetError(Exception):
    """Generic dataset-related error."""


class ValidationError(DatasetError):
    """Raised when a data item fails validation."""


class DatasetConnector(ABC):
    """
    Abstract base class for dataset connectors.

    Subclasses must implement `load_data()` and `get_metadata()`.
    Provides utilities for validation and safe iteration.
    """

    required_keys: Optional[Iterable[str]] = None

    @abstractmethod
    def load_data(self) -> Iterator[Dict[str, Any]]:
        """Yield data items as dictionaries."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata (non-PII)."""
        raise NotImplementedError

    # Utility methods
    def validate_item(self, item: Dict[str, Any]) -> None:
        """Validate a single item against `required_keys` if provided."""
        if self.required_keys is None:
            return
        missing = [k for k in self.required_keys if k not in item]
        if missing:
            raise ValidationError(f"Missing required keys: {missing}")

    def iter_validated(self) -> Iterator[Dict[str, Any]]:
        """Iterate items, validating each one."""
        for item in self.load_data():
            self.validate_item(item)
            yield item

    def take(self, n: int) -> List[Dict[str, Any]]:
        """Collect first n items (validated)."""
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(self.iter_validated()):
            if i >= n:
                break
            out.append(item)
        return out

    def count(self, limit: Optional[int] = None) -> int:
        """Count items up to an optional limit to avoid OOM on huge datasets."""
        c = 0
        for _ in self.load_data():
            c += 1
            if limit is not None and c >= limit:
                break
        return c

    def iter_batches(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate validated items in batches of size `batch_size`.

        This utility helps downstream tasks to process large datasets efficiently.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        batch: List[Dict[str, Any]] = []
        for item in self.iter_validated():
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
