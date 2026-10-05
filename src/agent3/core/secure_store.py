"""Authenticated, encrypted JSON storage used for quotas and credentials.

Design notes
------------
* **AES-256-GCM** provides confidentiality *and* integrity, so a user cannot
  silently hand-edit the quota file to lift enterprise limits - any tampering
  makes decryption fail loudly (:class:`SecureStoreError`).
* The key is derived with **PBKDF2-HMAC-SHA256** (390k iterations) from either
  an administrator passphrase or, when no passphrase is configured, from a
  machine-bound default secret.  That keeps the "it just works" first-run
  experience while still allowing an admin to lock the policy down.
* The on-disk envelope is plain JSON so that the format stays debuggable and
  forward compatible (``v`` field).
* Writes are atomic (temp file + ``os.replace``) which matters because the main
  application and ``UsageLimitEditor.exe`` touch the same files concurrently.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ENVELOPE_VERSION = 1
KDF_ITERATIONS = 390_000
KEY_LENGTH = 32
NONCE_LENGTH = 12
# Static application salt - mixed into the machine-bound default passphrase.
_APP_SALT = b"agent3::secure-store::v1"


class SecureStoreError(RuntimeError):
    """Raised when a secure payload cannot be read, written or decrypted."""


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except Exception as exc:  # pragma: no cover - malformed file
        raise SecureStoreError(f"corrupt base64 payload: {exc}") from exc


def machine_default_passphrase() -> str:
    """Derive a stable, machine-bound passphrase for unattended first runs."""
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - exotic/no-login environments
        user = "unknown-user"
    material = "|".join(
        [
            platform.node() or "unknown-host",
            platform.machine() or "unknown-arch",
            platform.system() or "unknown-os",
            user,
        ]
    ).encode("utf-8")
    return hashlib.sha256(_APP_SALT + material).hexdigest()


def derive_key(passphrase: str, salt: bytes, iterations: int = KDF_ITERATIONS) -> bytes:
    """Derive a 256 bit AES key from *passphrase* and *salt*."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_LENGTH,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase.encode("utf-8"))


class SecureStore:
    """Encrypted JSON document stored at *path*.

    Example
    -------
    >>> store = SecureStore(Path(tmp) / "policy.enc")          # doctest: +SKIP
    >>> store.save({"max_requests_per_day": 500})              # doctest: +SKIP
    >>> store.load()["max_requests_per_day"]                   # doctest: +SKIP
    500
    """

    def __init__(self, path: Path, passphrase: Optional[str] = None) -> None:
        self._path = Path(path)
        self._passphrase = passphrase or machine_default_passphrase()
        self._custom_passphrase = passphrase is not None

    # ------------------------------------------------------------------ meta
    @property
    def path(self) -> Path:
        return self._path

    @property
    def uses_custom_passphrase(self) -> bool:
        return self._custom_passphrase

    def exists(self) -> bool:
        return self._path.is_file()

    def set_passphrase(self, passphrase: Optional[str]) -> None:
        """Switch the key material used for subsequent reads/writes."""
        self._passphrase = passphrase or machine_default_passphrase()
        self._custom_passphrase = passphrase is not None

    def rekey(self, new_passphrase: Optional[str]) -> None:
        """Re-encrypt the existing payload under a new passphrase."""
        payload = self.load() if self.exists() else {}
        self.set_passphrase(new_passphrase)
        self.save(payload)

    def is_passphrase_protected(self) -> bool:
        """True when the file cannot be opened with the machine default key."""
        if not self.exists():
            return False
        probe = SecureStore(self._path, passphrase=None)
        try:
            probe.load()
        except SecureStoreError:
            return True
        return False

    # ------------------------------------------------------------------- io
    def load(self, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Decrypt and return the stored mapping.

        ``default`` is returned when the file does not exist yet.  A corrupt or
        wrongly-keyed file always raises :class:`SecureStoreError` - silently
        falling back to defaults would be a quota bypass.
        """
        if not self._path.is_file():
            return dict(default) if default is not None else {}

        try:
            envelope = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SecureStoreError(f"cannot read {self._path.name}: {exc}") from exc

        if not isinstance(envelope, dict) or envelope.get("v") != ENVELOPE_VERSION:
            raise SecureStoreError(f"unsupported secure-store envelope in {self._path.name}")

        salt = _b64d(str(envelope.get("salt", "")))
        nonce = _b64d(str(envelope.get("nonce", "")))
        ciphertext = _b64d(str(envelope.get("ct", "")))
        iterations = int(envelope.get("iterations", KDF_ITERATIONS))

        key = derive_key(self._passphrase, salt, iterations)
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, _APP_SALT)
        except InvalidTag as exc:
            raise SecureStoreError(
                f"{self._path.name} could not be decrypted - wrong passphrase or tampered file"
            ) from exc

        try:
            data = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
            raise SecureStoreError(f"corrupt payload inside {self._path.name}: {exc}") from exc

        if not isinstance(data, dict):
            raise SecureStoreError(f"{self._path.name} does not contain a JSON object")
        return data

    def save(self, data: Dict[str, Any]) -> None:
        """Encrypt *data* and atomically replace the target file."""
        if not isinstance(data, dict):
            raise SecureStoreError("secure store payload must be a mapping")

        salt = os.urandom(16)
        nonce = os.urandom(NONCE_LENGTH)
        key = derive_key(self._passphrase, salt, KDF_ITERATIONS)
        plaintext = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, _APP_SALT)

        envelope = {
            "v": ENVELOPE_VERSION,
            "kdf": "pbkdf2-sha256",
            "iterations": KDF_ITERATIONS,
            "salt": _b64e(salt),
            "nonce": _b64e(nonce),
            "ct": _b64e(ciphertext),
        }
        self._atomic_write(json.dumps(envelope, indent=2))

    def _atomic_write(self, text: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self._path)
        except OSError as exc:
            raise SecureStoreError(f"cannot write {self._path.name}: {exc}") from exc
        finally:
            if os.path.exists(tmp_name):  # pragma: no cover - replace succeeded normally
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
