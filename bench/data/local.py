from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterator, Any, List, Optional

from .base import DatasetConnector
from .security import SecureDataHandler


class JSONDataset(DatasetConnector):
    """Load items from a JSON file containing a list of objects."""

    def __init__(self, file_path: Path | str, encryption_key: Optional[str] = None):
        self.file_path = Path(file_path)
        self.handler = SecureDataHandler(encryption_key)

    def _read_file(self) -> List[Dict[str, Any]]:
        with self.file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSONDataset expects a top-level list of objects")
        return data

    def load_data(self) -> Iterator[Dict[str, Any]]:
        for item in self._read_file():
            yield self.handler.decrypt_data(item)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source": str(self.file_path),
            "format": "json",
            "size": self.file_path.stat().st_size if self.file_path.exists() else 0,
        }


class CSVDataset(DatasetConnector):
    """Load items from a CSV file as dict rows."""

    def __init__(self, file_path: Path | str, encryption_key: Optional[str] = None):
        self.file_path = Path(file_path)
        self.handler = SecureDataHandler(encryption_key)

    def load_data(self) -> Iterator[Dict[str, Any]]:
        with self.file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # DictReader returns OrderedDict/Dict[str, str]; decrypt string cells
                yield self.handler.decrypt_data(dict(row))

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source": str(self.file_path),
            "format": "csv",
            "size": self.file_path.stat().st_size if self.file_path.exists() else 0,
        }
