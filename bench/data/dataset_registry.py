from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ValidationError, field_validator, model_validator


class DatasetMeta(BaseModel):
    id: str
    name: str
    description: str = ""
    size: Optional[int] = None
    task_categories: List[str] = []
    source_links: List[str] = []
    composition: Optional[Dict[str, int]] = None
    planned: bool = False

    @field_validator("id")
    @classmethod
    def _lower_id(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "-")

    @model_validator(mode="after")
    def _validate_composition(self) -> "DatasetMeta":
        # If both size and composition are provided, ensure they are consistent
        if self.size is not None and self.composition:
            try:
                total = sum(int(v) for v in self.composition.values())
                if int(self.size) != total:
                    raise ValueError(
                        f"composition totals {total} but size is {self.size}"
                    )
            except Exception as e:
                raise ValueError(f"Invalid composition values: {e}")
        return self


class DatasetRegistry:
    """JSON-file backed dataset registry with schema validation.

    - Validates entries with Pydantic schema
    - Provides CRUD operations
    - Exposes JSON Schema for documentation
    """

    def __init__(self, registry_path: Path | str | None = None) -> None:
        self.registry_path = Path(
            registry_path
            or Path(__file__).resolve().parent / "datasets" / "registry.json"
        )
        self._entries: Dict[str, DatasetMeta] = {}
        self._loaded = False

    # -----------------
    # Internal helpers
    # -----------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._entries = {}
        if self.registry_path.exists():
            try:
                raw = json.loads(self.registry_path.read_text())
                if isinstance(raw, list):
                    for item in raw:
                        try:
                            meta = DatasetMeta(**(item or {}))
                            self._entries[meta.id] = meta
                        except ValidationError:
                            # Skip invalid entries but continue loading others
                            continue
                elif isinstance(raw, dict):
                    for k, item in raw.items():
                        try:
                            if isinstance(item, dict):
                                item.setdefault("id", k)
                            meta = DatasetMeta(**(item or {}))
                            self._entries[meta.id] = meta
                        except ValidationError:
                            continue
            except Exception:
                # On parse errors, start with empty registry
                self._entries = {}
        self._loaded = True

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [json.loads(m.model_dump_json()) for m in self._entries.values()]
        self.registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # ---------
    # Public IO
    # ---------
    def json_schema(self) -> Dict[str, object]:
        return DatasetMeta.model_json_schema()

    # -----
    # CRUD
    # -----
    def list(self) -> List[DatasetMeta]:
        self._ensure_loaded()
        return sorted(self._entries.values(), key=lambda m: m.id)

    def get(self, dataset_id: str) -> DatasetMeta:
        self._ensure_loaded()
        key = dataset_id.strip().lower()
        if key not in self._entries:
            raise KeyError(f"Dataset not found: {dataset_id}")
        return self._entries[key]

    def add(
        self, meta: Dict[str, object] | DatasetMeta, *, save: bool = True
    ) -> DatasetMeta:
        self._ensure_loaded()
        if not isinstance(meta, DatasetMeta):
            meta = DatasetMeta(**meta)  # type: ignore[arg-type]
        self._entries[meta.id] = meta
        if save:
            self._save()
        return meta

    def update(
        self, dataset_id: str, patch: Dict[str, object], *, save: bool = True
    ) -> DatasetMeta:
        self._ensure_loaded()
        cur = self.get(dataset_id)
        data = cur.model_dump()
        data.update(patch or {})
        updated = DatasetMeta(**data)
        self._entries[updated.id] = updated
        if save:
            self._save()
        return updated

    def remove(self, dataset_id: str, *, save: bool = True) -> None:
        self._ensure_loaded()
        self._entries.pop(dataset_id, None)
        if save:
            self._save()


# Convenience: default registry accessor
_default_registry: Optional[DatasetRegistry] = None


def get_default_registry() -> DatasetRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = DatasetRegistry()
    return _default_registry
