from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import pytest

from bench.data.local import JSONDataset, CSVDataset
from bench.data.security import SecureDataHandler


@pytest.fixture()
def corrupted_encrypted_json(tmp_path: Path) -> Path:
    handler = SecureDataHandler("correct-pass")
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]
    enc = [handler.encrypt_data(x) for x in data]
    # Corrupt the ciphertext for the first item name field
    s = enc[0]["name"]
    # Flip a character safely (keep string type but invalidate token)
    enc[0]["name"] = ("X" + s[1:]) if isinstance(s, str) and len(s) > 1 else s + "X"
    f = tmp_path / "corrupted.json"
    f.write_text(json.dumps(enc), encoding="utf-8")
    return f


@pytest.fixture()
def wrong_key_encrypted_csv(tmp_path: Path) -> Path:
    from csv import DictWriter

    handler = SecureDataHandler("correct-pass")
    rows = [
        {"id": "1", "name": "Alice"},
        {"id": "2", "name": "Bob"},
    ]
    enc = [handler.encrypt_data(r) for r in rows]
    f = tmp_path / "enc_wrongkey.csv"
    with f.open("w", encoding="utf-8", newline="") as fh:
        w = DictWriter(fh, fieldnames=["id", "name"])
        w.writeheader()
        w.writerows(enc)
    return f


def test_json_dataset_with_corrupted_ciphertexts(corrupted_encrypted_json: Path):
    # Use the correct key, but first item has corrupted ciphertext -> handler should pass-through string
    ds = JSONDataset(
        corrupted_encrypted_json,
        encryption_key="correct-pass",
        required_keys=["id", "name"],
    )
    items: List[Dict[str, Any]] = list(ds.load_data())

    assert len(items) == 2
    # First item name remains an opaque string (not original plaintext)
    assert isinstance(items[0]["name"], str)
    assert items[0]["name"] != "Alice"
    # Second item should decrypt fine
    assert items[1]["name"] == "Bob"


def test_csv_dataset_with_wrong_key(wrong_key_encrypted_csv: Path):
    # Use wrong key so decryption fails; strings should pass through
    ds = CSVDataset(
        wrong_key_encrypted_csv,
        encryption_key="wrong-pass",
        required_keys=["id", "name"],
    )
    items: List[Dict[str, Any]] = list(ds.load_data())

    assert len(items) == 2
    # Names remain encrypted/opaque strings
    assert all(
        isinstance(it["name"], str) and it["name"] not in {"Alice", "Bob"}
        for it in items
    )
    # Required keys still present so validation passes


def test_mixed_partially_encrypted_rows_json(tmp_path: Path):
    """Mix plaintext and encrypted rows; handler should decrypt what it can and pass through others."""
    handler = SecureDataHandler("k1")
    plain = {"id": 1, "name": "PLAIN"}
    enc = handler.encrypt_data({"id": 2, "name": "SECRET"})
    # Mixed array order
    data = [plain, enc]
    f = tmp_path / "mixed.json"
    f.write_text(json.dumps(data), encoding="utf-8")

    # Use correct key: first stays plaintext, second decrypts
    ds = JSONDataset(f, encryption_key="k1", required_keys=["id", "name"])
    items: List[Dict[str, Any]] = list(ds.load_data())
    assert items[0]["name"] == "PLAIN"
    assert items[1]["name"] == "SECRET"


def test_truncated_base64_token_pass_through(tmp_path: Path):
    """Truncate an encrypted field to break base64; decryption should fail and return original string."""
    handler = SecureDataHandler("k2")
    enc = handler.encrypt_data({"id": 1, "name": "TOKEN"})
    # Truncate encrypted token to an invalid base64 (remove last 2 chars)
    s = enc["name"]
    enc["name"] = s[:-2] if isinstance(s, str) and len(s) > 2 else s + "X"
    f = tmp_path / "trunc.json"
    f.write_text(json.dumps([enc]), encoding="utf-8")

    ds = JSONDataset(f, encryption_key="k2", required_keys=["id", "name"])
    items: List[Dict[str, Any]] = list(ds.load_data())
    assert len(items) == 1
    # Name should be an opaque string, not original plaintext
    assert isinstance(items[0]["name"], str)
    assert items[0]["name"] != "TOKEN"
