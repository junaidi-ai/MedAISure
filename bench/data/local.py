from __future__ import annotations

import csv
import json
import gzip
import zipfile
from pathlib import Path
from typing import Dict, Iterator, Any, List, Optional, Iterable

from .base import DatasetConnector, DatasetError
from .security import SecureDataHandler


class JSONDataset(DatasetConnector):
    """Load items from a JSON file containing a list of objects."""

    def __init__(
        self,
        file_path: Path | str,
        encryption_key: Optional[str] = None,
        required_keys: Optional[Iterable[str]] = None,
    ):
        self.file_path = Path(file_path)
        self.handler = SecureDataHandler(encryption_key)
        self.required_keys = required_keys

    def _read_file(self) -> List[Dict[str, Any]]:
        try:
            if not self.file_path.exists():
                raise DatasetError(f"File not found: {self.file_path}")

            # Support gzip and zip archives
            if self.file_path.suffix == ".gz":
                with gzip.open(self.file_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            elif self.file_path.suffix == ".zip":
                with zipfile.ZipFile(self.file_path) as zf:
                    # pick the first .json file
                    name = next(
                        (n for n in zf.namelist() if n.lower().endswith(".json")), None
                    )
                    if not name:
                        raise DatasetError("No .json file found inside zip archive")
                    with zf.open(name) as fh:
                        data = json.loads(fh.read().decode("utf-8"))
            else:
                with self.file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

            if not isinstance(data, list):
                raise DatasetError("JSONDataset expects a top-level list of objects")
            return data
        except DatasetError:
            raise
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON: {e}") from e
        except zipfile.BadZipFile as e:
            raise DatasetError(f"Invalid zip file: {e}") from e
        except OSError as e:
            # Includes gzip.BadGzipFile and file IO errors
            raise DatasetError(f"File error: {e}") from e

    def load_data(self) -> Iterator[Dict[str, Any]]:
        for item in self._read_file():
            obj = self.handler.decrypt_data(item)
            self.validate_item(obj)
            yield obj

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source": str(self.file_path),
            "format": "json",
            "size": self.file_path.stat().st_size if self.file_path.exists() else 0,
        }


class CSVDataset(DatasetConnector):
    """Load items from a CSV file as dict rows."""

    def __init__(
        self,
        file_path: Path | str,
        encryption_key: Optional[str] = None,
        required_keys: Optional[Iterable[str]] = None,
    ):
        self.file_path = Path(file_path)
        self.handler = SecureDataHandler(encryption_key)
        self.required_keys = required_keys

    def load_data(self) -> Iterator[Dict[str, Any]]:
        if not self.file_path.exists():
            raise DatasetError(f"File not found: {self.file_path}")

        try:
            if self.file_path.suffix == ".gz":
                fctx = gzip.open(self.file_path, "rt", encoding="utf-8", newline="")
                close_needed = True
            elif self.file_path.suffix == ".zip":
                zf = zipfile.ZipFile(self.file_path)
                name = next(
                    (n for n in zf.namelist() if n.lower().endswith(".csv")), None
                )
                if not name:
                    zf.close()
                    raise DatasetError("No .csv file found inside zip archive")
                fctx = zf.open(name)
                # Text IO wrapper to decode
                import io

                fctx = io.TextIOWrapper(fctx, encoding="utf-8", newline="")
                close_needed = True
            else:
                fctx = self.file_path.open("r", encoding="utf-8", newline="")
                close_needed = True

            try:
                reader = csv.DictReader(fctx)
                for row in reader:
                    obj = self.handler.decrypt_data(dict(row))
                    # Normalize: treat empty required fields as missing
                    if self.required_keys:
                        for rk in self.required_keys:
                            if rk in obj:
                                v = obj[rk]
                                if v is None or (isinstance(v, str) and v == ""):
                                    del obj[rk]
                    self.validate_item(obj)
                    yield obj
            finally:
                if close_needed:
                    fctx.close()
                if "zf" in locals():
                    zf.close()
        except zipfile.BadZipFile as e:
            raise DatasetError(f"Invalid zip file: {e}") from e
        except OSError as e:
            raise DatasetError(f"File error: {e}") from e

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source": str(self.file_path),
            "format": "csv",
            "size": self.file_path.stat().st_size if self.file_path.exists() else 0,
        }
