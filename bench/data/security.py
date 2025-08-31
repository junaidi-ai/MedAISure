from __future__ import annotations

import base64
import hashlib
from typing import Dict, Any

from cryptography.fernet import Fernet


class SecureDataHandler:
    """
    Optional encryption/decryption helper using Fernet.

    Only string values are encrypted/decrypted; non-strings are passed through.
    """

    def __init__(self, encryption_key: str | None = None) -> None:
        self.encryption_key = encryption_key
        self.cipher: Fernet | None = None
        if encryption_key:
            # Derive a 32-byte key from the provided string and convert to a urlsafe base64 key
            key = hashlib.sha256(encryption_key.encode()).digest()
            fkey = base64.urlsafe_b64encode(key)
            self.cipher = Fernet(fkey)

    def encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cipher:
            return dict(data)
        result: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str):
                result[k] = self.cipher.encrypt(v.encode()).decode()
            else:
                result[k] = v
        return result

    def decrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cipher:
            return dict(data)
        result: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str):
                try:
                    result[k] = self.cipher.decrypt(v.encode()).decode()
                except Exception:
                    # Not encrypted or invalid token; return as-is
                    result[k] = v
            else:
                result[k] = v
        return result
