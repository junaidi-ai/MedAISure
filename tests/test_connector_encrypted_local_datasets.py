from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pytest

from bench.data.local import JSONDataset, CSVDataset


@pytest.mark.parametrize(
    "factory_fixture, cls, required",
    [
        ("encrypted_json_file", JSONDataset, ["id", "name"]),
        ("encrypted_csv_file", CSVDataset, ["id", "name"]),
    ],
)
def test_encrypted_local_datasets_decrypt_end_to_end(
    request: pytest.FixtureRequest, factory_fixture: str, cls, required: List[str]
):
    # Files were encrypted with passphrase 'test-pass' in fixtures
    path: Path = request.getfixturevalue(factory_fixture)
    ds = cls(path, encryption_key="test-pass", required_keys=required)

    items: List[Dict[str, Any]] = list(ds.load_data())
    assert len(items) == 2
    # Ensure decrypted plaintext values are visible
    assert items[0]["name"] in {"Alice", "Bob"}
    assert items[1]["name"] in {"Alice", "Bob"}
    # Metadata is populated
    meta = ds.get_metadata()
    assert meta["format"] in {"json", "csv"}
    assert isinstance(meta["size"], int)
