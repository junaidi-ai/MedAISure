from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bench.data.security import SecureDataHandler


@pytest.mark.parametrize("algorithm", ["fernet", "aes-gcm", "chacha20"])
def test_algorithms_roundtrip(algorithm: str):
    handler = SecureDataHandler("secret-pass", algorithm=algorithm)
    src = {"a": "hello", "b": 5}
    enc = handler.encrypt_data(src)
    assert enc["b"] == 5 and isinstance(enc["a"], str) and enc["a"] != "hello"
    dec = handler.decrypt_data(enc)
    assert dec == src


def test_include_exclude_fields():
    handler = SecureDataHandler(
        "pass",
        algorithm="fernet",
        include_fields={"name"},
        exclude_fields={"skip"},
    )
    data = {"name": "Alice", "email": "a@example.com", "skip": "nope", "age": 33}
    enc = handler.encrypt_data(data)

    # Only 'name' should be encrypted (include_fields restricts)
    assert enc["name"] != "Alice"
    assert enc["email"] == "a@example.com"
    assert enc["skip"] == "nope"
    assert enc["age"] == 33

    dec = handler.decrypt_data(enc)
    assert dec == data


def test_anonymization_redacts_phi():
    handler = SecureDataHandler("pass", anonymize=True)
    text = "Contact me at john.doe@example.com or 555-123-4567 on 2024-01-31"
    data = {"msg": text, "other": 1}

    # Note: encrypt_data applies anonymization first when enabled
    enc = handler.encrypt_data(data)
    dec = handler.decrypt_data(enc)

    # Message should be anonymized; email/phone/date masked deterministically
    assert dec["other"] == 1
    assert dec["msg"] != text
    assert "example.com" not in dec["msg"]
    # Has redaction token format
    assert re.search(r"\[REDACTED:[A-Za-z0-9_-]{8}\]", dec["msg"]) is not None

    # Direct anonymization API should also redact
    anon = handler.anonymize_data({"msg": text})
    assert anon["msg"] != text


def test_audit_logging(tmp_path: Path):
    log_file = tmp_path / "audit.jsonl"
    handler = SecureDataHandler("pass", audit_log_path=str(log_file))

    d = {"x": "v"}
    _ = handler.encrypt_data(d)
    _ = handler.decrypt_data(d)
    _ = handler.anonymize_data(d)
    handler.rotate_key("newpass")

    assert log_file.exists()
    lines = [
        json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()
    ]
    actions = [rec.get("action") for rec in lines]
    # Ensure all actions were logged at least once
    for a in ("encrypt", "decrypt", "anonymize", "rotate_key"):
        assert a in actions


def test_key_rotation_and_backward_decrypt():
    handler = SecureDataHandler("pass", algorithm="aes-gcm")

    old_ct = handler.encrypt_data({"s": "one"})
    handler.rotate_key("pass-2")  # now current key differs
    # Should be able to decrypt ciphertext created with old key
    dec_old = handler.decrypt_data(old_ct)
    assert dec_old == {"s": "one"}

    # Also ensure new encryptions decrypt
    new_ct = handler.encrypt_data({"s": "two"})
    dec_new = handler.decrypt_data(new_ct)
    assert dec_new == {"s": "two"}


def test_batch_helpers():
    handler = SecureDataHandler("pass")
    batch = [{"a": "x"}, {"a": "y", "n": 1}]

    enc_batch = handler.encrypt_batch(batch)
    assert len(enc_batch) == 2
    assert enc_batch[0]["a"] != "x"

    dec_batch = handler.decrypt_batch(enc_batch)
    assert dec_batch == batch
