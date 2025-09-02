from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305


@dataclass
class _KeyEntry:
    kid: str
    alg: str
    # Raw key bytes for AEAD algorithms or Fernet-ready 32-byte base64 key (urlsafe)
    key_bytes: bytes


class SecureDataHandler:
    """
    Secure data helper with optional encryption, PHI anonymization, and audit logging.

    Backward compatibility:
    - If constructed with only `encryption_key`, defaults to Fernet and encrypts all string
      values into opaque strings. Decryption best-effort returns original strings if not encrypted.

    Extended features:
    - Multiple algorithms: 'fernet' (default), 'aes-gcm', 'chacha20'
    - Field-level control via `include_fields` / `exclude_fields`
    - PHI detection and anonymization (email, phone, SSN, MRN-like, DOB-like patterns)
    - Compliance audit logging (JSONL) for encrypt/decrypt/anonymize/rotate
    - Key management and rotation; previous keys retained for decryption
    - Batch helpers for performance on large datasets
    """

    def __init__(
        self,
        encryption_key: Optional[str] = None,
        *,
        algorithm: str = "fernet",
        include_fields: Optional[Iterable[str]] = None,
        exclude_fields: Optional[Iterable[str]] = None,
        anonymize: bool = False,
        audit_log_path: Optional[str] = None,
    ) -> None:
        self.algorithm = algorithm.lower()
        self.anonymize_enabled = anonymize
        self.include_fields: Optional[Set[str]] = (
            set(include_fields) if include_fields else None
        )
        self.exclude_fields: Set[str] = set(exclude_fields) if exclude_fields else set()
        self.audit_log_path = audit_log_path

        # Key store by key id
        self._keys: Dict[str, _KeyEntry] = {}
        self._current_kid: Optional[str] = None

        if encryption_key:
            # Derive a stable 32-byte key from passphrase for symmetric algorithms
            derived = hashlib.sha256(encryption_key.encode()).digest()
            if self.algorithm == "fernet":
                fkey = base64.urlsafe_b64encode(derived)
                self._add_key("k0", fkey, alg="fernet")
            elif self.algorithm in {"aes-gcm", "chacha20"}:
                # For AESGCM and ChaCha20Poly1305 we use raw bytes (16/32 bytes ok)
                self._add_key("k0", derived, alg=self.algorithm)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # Precompile PHI regexes for speed
        self._phi_patterns: List[re.Pattern[str]] = [
            re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # email
            re.compile(
                r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)|\d{2,4})[-.\s]?\d{3,4}[-.\s]?\d{4}\b"
            ),  # phone
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
            re.compile(r"\b\d{8,10}\b"),  # MRN-like numeric ids
            re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),  # YYYY-MM-DD
        ]

    # --------------- Key management ---------------
    def _add_key(self, kid: str, key_bytes: bytes, *, alg: str) -> None:
        self._keys[kid] = _KeyEntry(kid=kid, alg=alg, key_bytes=key_bytes)
        self._current_kid = kid

    def rotate_key(self, new_passphrase: str, *, kid: Optional[str] = None) -> str:
        """Rotate to a new key derived from `new_passphrase`. Returns new kid.

        Previous keys are retained for decryption. Logs the rotation in audit log.
        """
        derived = hashlib.sha256(new_passphrase.encode()).digest()
        if self.algorithm == "fernet":
            key_bytes = base64.urlsafe_b64encode(derived)
        elif self.algorithm in {"aes-gcm", "chacha20"}:
            key_bytes = derived
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        kid = kid or f"k{len(self._keys)}"
        self._add_key(kid, key_bytes, alg=self.algorithm)
        self._audit("rotate_key", {"kid": kid, "alg": self.algorithm})
        return kid

    # --------------- Audit logging ---------------
    def _audit(self, action: str, details: Mapping[str, Any]) -> None:
        if not self.audit_log_path:
            return
        rec = {
            "ts": int(time.time() * 1000),
            "action": action,
            "alg": self.algorithm,
            "kid": self._current_kid,
            "details": dict(details),
        }
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # Do not break data paths due to logging failures
            pass

    # --------------- PHI anonymization ---------------
    def _anonymize_string(self, s: str) -> str:
        # Replace detected PHI substrings with deterministic tokens
        out = s
        for pat in self._phi_patterns:

            def repl(m: re.Match[str]) -> str:
                token = hashlib.sha256(m.group(0).encode()).digest()
                return "[REDACTED:" + base64.urlsafe_b64encode(token[:6]).decode() + "]"

            out = pat.sub(repl, out)
        return out

    def anonymize_data(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str):
                result[k] = self._anonymize_string(v)
            else:
                result[k] = v
        self._audit("anonymize", {"fields": list(data.keys())})
        return result

    # --------------- Encryption helpers ---------------
    def _should_process_field(self, field: str) -> bool:
        if self.include_fields is not None:
            return field in self.include_fields
        if field in self.exclude_fields:
            return False
        return True

    def _encrypt_str(self, txt: str) -> str:
        if self._current_kid is None:
            return txt
        entry = self._keys[self._current_kid]
        if entry.alg == "fernet":
            cipher = Fernet(entry.key_bytes)
            return cipher.encrypt(txt.encode()).decode()
        elif entry.alg == "aes-gcm":
            nonce = os.urandom(12)
            aead = AESGCM(entry.key_bytes)
            ct = aead.encrypt(nonce, txt.encode(), None)
            return base64.urlsafe_b64encode(nonce + ct).decode()
        elif entry.alg == "chacha20":
            nonce = os.urandom(12)
            aead = ChaCha20Poly1305(entry.key_bytes)
            ct = aead.encrypt(nonce, txt.encode(), None)
            return base64.urlsafe_b64encode(nonce + ct).decode()
        else:
            raise ValueError(f"Unsupported algorithm: {entry.alg}")

    def _try_decrypt_str_with_entry(self, s: str, entry: _KeyEntry) -> Optional[str]:
        try:
            if entry.alg == "fernet":
                cipher = Fernet(entry.key_bytes)
                return cipher.decrypt(s.encode()).decode()
            else:
                raw = base64.urlsafe_b64decode(s.encode())
                nonce, ct = raw[:12], raw[12:]
                if entry.alg == "aes-gcm":
                    aead = AESGCM(entry.key_bytes)
                    return aead.decrypt(nonce, ct, None).decode()
                elif entry.alg == "chacha20":
                    aead = ChaCha20Poly1305(entry.key_bytes)
                    return aead.decrypt(nonce, ct, None).decode()
                else:
                    return None
        except Exception:
            return None

    # --------------- Public API ---------------
    def encrypt_data(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        if self._current_kid is None:
            return dict(data)
        src = self.anonymize_data(data) if self.anonymize_enabled else data
        result: Dict[str, Any] = {}
        for k, v in src.items():
            if isinstance(v, str) and self._should_process_field(k):
                result[k] = self._encrypt_str(v)
            else:
                result[k] = v
        self._audit("encrypt", {"fields": list(src.keys())})
        return result

    def decrypt_data(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        if not self._keys:
            return dict(data)
        result: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str) and self._should_process_field(k):
                # Try all keys (current first) to support rotation
                tried: List[_KeyEntry] = []
                if self._current_kid is not None:
                    tried.append(self._keys[self._current_kid])
                for e in self._keys.values():
                    if not tried or e.kid != tried[0].kid:
                        tried.append(e)
                dec: Optional[str] = None
                for e in tried:
                    dec = self._try_decrypt_str_with_entry(v, e)
                    if dec is not None:
                        break
                result[k] = dec if dec is not None else v
            else:
                result[k] = v
        self._audit("decrypt", {"fields": list(data.keys())})
        return result

    # Batch helpers for performance
    def encrypt_batch(self, batch: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return [self.encrypt_data(x) for x in batch]

    def decrypt_batch(self, batch: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return [self.decrypt_data(x) for x in batch]
