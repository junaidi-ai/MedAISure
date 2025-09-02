# Secure Data Handling

This document explains how to use `SecureDataHandler` (in `bench/data/security.py`) to protect sensitive data, including algorithm selection, PHI anonymization, field-level policies, audit logging, and key rotation.

## Quick Start

```python
from bench.data.security import SecureDataHandler

handler = SecureDataHandler("secret-key")  # Fernet by default
cipher = handler.encrypt_data({"name": "Alice", "age": 30})
plain = handler.decrypt_data(cipher)
```

- Only string values are encrypted/decrypted; non-strings pass through.
- If no key is provided, encrypt/decrypt are no-ops.

## Algorithm Selection

Supported algorithms:
- `fernet` (default) — simple, URL-safe tokens, includes integrity.
- `aes-gcm` — AEAD with GCM; encodes `nonce+ciphertext` in URL-safe base64.
- `chacha20` — AEAD ChaCha20-Poly1305; encodes `nonce+ciphertext` in URL-safe base64.

```python
SecureDataHandler("pass", algorithm="aes-gcm")
SecureDataHandler("pass", algorithm="chacha20")
```

## Field-Level Policies

Control which fields are processed (encrypted/decrypted):

```python
# Only encrypt these fields
handler = SecureDataHandler("pass", include_fields={"name", "email"})

# Encrypt all string fields except these
handler = SecureDataHandler("pass", exclude_fields={"public_note"})
```

`include_fields` takes precedence over `exclude_fields`.

## PHI Detection & Anonymization

Enable anonymization to automatically redact likely PHI before encryption:

```python
handler = SecureDataHandler("pass", anonymize=True)
redacted_then_encrypted = handler.encrypt_data({
    "msg": "Contact john.doe@example.com or 555-123-4567 on 2024-01-31",
})
```

- Emails, phone numbers, SSN-like patterns, MRN-like IDs, and `YYYY-MM-DD` dates are detected.
- Redaction is deterministic, producing tokens like `[REDACTED:ABCDEFGH]`.
- You can also call `anonymize_data()` directly.

## Compliance Audit Logging

Record audits to JSONL for encrypt/decrypt/anonymize/rotate:

```python
handler = SecureDataHandler("pass", audit_log_path="audit.jsonl")
handler.encrypt_data({"x": "1"})
handler.decrypt_data({"x": "..."})
handler.anonymize_data({"x": "..."})
handler.rotate_key("new-pass")
```

Each line contains a JSON object with:
- `ts` (ms timestamp)
- `action` (encrypt|decrypt|anonymize|rotate_key)
- `alg`, `kid` (current key id), and action `details`

If logging fails, operations continue (best-effort logging).

## Key Management & Rotation

Create with an initial key (derived from passphrase). Rotate to a new key while retaining old keys for decryption:

```python
handler = SecureDataHandler("pass", algorithm="aes-gcm")
old_ct = handler.encrypt_data({"s": "one"})
handler.rotate_key("new-pass")  # new key used for future encryptions
plain_old = handler.decrypt_data(old_ct)  # still decrypts
```

- Keys are referenced by `kid` (e.g., `k0`, `k1`, ...).
- Current key is preferred for decryption; fallbacks attempt older keys.

## Batch Helpers

For large datasets, use batch helpers to avoid per-item overhead:

```python
enc_list = handler.encrypt_batch(list_of_dicts)
plain_list = handler.decrypt_batch(enc_list)
```

## Operational Guidance

- Choose `fernet` for simplicity and compatibility; `aes-gcm` or `chacha20` for AEAD performance.
- Use `include_fields` to explicitly protect PII/PHI fields and avoid unexpected ciphertext expansion.
- Enable `anonymize=True` when ingesting free-text inputs that may contain PHI.
- Configure `audit_log_path` in environments requiring auditability. Persist logs to secure storage.
- Rotate keys periodically with `rotate_key()`. Keep audit logs of rotations and store passphrases in a secure secret manager.
- Test representative payloads. Some tokens (e.g., long base64) can expand storage size.
- Be cautious with double-encrypting fields. The handler attempts best-effort decryption and will pass through unknown/plaintext strings.

## References

- Implementation: `bench/data/security.py`
- Usage in data connectors: `bench/data/local.py`
- Tests: `tests/test_security_handler.py`
