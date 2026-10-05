"""Quota policy definition and its encrypted persistence layer.

The policy is written by ``UsageLimitEditor.exe`` (the administrator utility)
and read *live* by the main application: the agent re-reads the file whenever
its modification time changes, so raising a quota takes effect without a
restart.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from agent3.core.logging_setup import get_logger
from agent3.core.paths import app_paths
from agent3.core.secure_store import SecureStore, SecureStoreError

logger = get_logger(__name__)

UNLIMITED = 0  # Sentinel: a limit of 0 means "no limit".


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class QuotaPolicy:
    """Administrator-controlled limits for one workstation.

    A value of ``0`` always means *unlimited* for that particular counter.
    """

    enabled: bool = True
    max_requests_per_day: int = 500
    max_tokens_per_session: int = 100_000
    max_tokens_per_day: int = 1_000_000
    max_runtime_seconds_per_day: int = 4 * 60 * 60
    max_agent_runs_per_day: int = 100
    max_tool_calls_per_run: int = 60
    max_tokens_per_request: int = 32_000
    min_seconds_between_requests: float = 0.0
    warn_threshold: float = 0.8
    developer_mode: bool = False
    #: When locked, the editor demands the administrator passphrase.
    locked: bool = False
    admin_password_hash: str = ""
    admin_password_salt: str = ""
    notes: str = ""
    updated_at: str = ""
    updated_by: str = ""

    # ---------------------------------------------------------- validation
    def normalised(self) -> "QuotaPolicy":
        """Return a copy with every field clamped into a sane range."""
        data = asdict(self)
        for key in (
            "max_requests_per_day",
            "max_tokens_per_session",
            "max_tokens_per_day",
            "max_runtime_seconds_per_day",
            "max_agent_runs_per_day",
            "max_tool_calls_per_run",
            "max_tokens_per_request",
        ):
            try:
                data[key] = max(0, int(data[key]))
            except (TypeError, ValueError):
                data[key] = 0
        try:
            data["min_seconds_between_requests"] = max(0.0, float(data["min_seconds_between_requests"]))
        except (TypeError, ValueError):
            data["min_seconds_between_requests"] = 0.0
        try:
            data["warn_threshold"] = min(1.0, max(0.1, float(data["warn_threshold"])))
        except (TypeError, ValueError):
            data["warn_threshold"] = 0.8
        for key in ("enabled", "developer_mode", "locked"):
            data[key] = bool(data[key])
        return QuotaPolicy(**data)

    # ------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "QuotaPolicy":
        if not isinstance(data, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        try:
            return cls(**payload).normalised()
        except TypeError:  # pragma: no cover - defensive
            return cls()

    # ------------------------------------------------------ admin password
    def set_admin_password(self, password: str) -> None:
        """Store a PBKDF2 hash of *password* (empty string clears it)."""
        if not password:
            self.admin_password_hash = ""
            self.admin_password_salt = ""
            self.locked = False
            return
        salt = secrets.token_hex(16)
        self.admin_password_salt = salt
        self.admin_password_hash = self._hash_password(password, salt)
        self.locked = True

    def verify_admin_password(self, password: str) -> bool:
        """Constant-time verification of the administrator password."""
        if not self.admin_password_hash:
            return True
        candidate = self._hash_password(password or "", self.admin_password_salt)
        return secrets.compare_digest(candidate, self.admin_password_hash)

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt or "00"), 200_000
        ).hex()

    # --------------------------------------------------------------- helpers
    def limit_for(self, name: str) -> int:
        return int(getattr(self, name, 0) or 0)

    def is_unlimited(self, name: str) -> bool:
        return self.limit_for(name) == UNLIMITED


class PolicyStore:
    """Encrypted, hot-reloadable storage for :class:`QuotaPolicy`."""

    def __init__(self, path: Optional[Path] = None, passphrase: Optional[str] = None) -> None:
        self._path = Path(path) if path else app_paths().policy_file
        self._store = SecureStore(self._path, passphrase=passphrase)
        self._cached: Optional[QuotaPolicy] = None
        self._mtime: float = -1.0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def secure_store(self) -> SecureStore:
        return self._store

    def exists(self) -> bool:
        return self._path.is_file()

    def _disk_mtime(self) -> float:
        try:
            return os.path.getmtime(self._path)
        except OSError:
            return -1.0

    def load(self, *, force: bool = False) -> QuotaPolicy:
        """Return the policy, re-reading the file when it changed on disk."""
        mtime = self._disk_mtime()
        if not force and self._cached is not None and mtime == self._mtime:
            return self._cached

        if not self._path.is_file():
            policy = QuotaPolicy(updated_at=_utc_now_iso(), updated_by="default")
            self.save(policy)
            return policy

        try:
            data = self._store.load()
            policy = QuotaPolicy.from_dict(data)
        except SecureStoreError as exc:
            # A tampered policy must *not* grant unlimited access: fail closed
            # onto the conservative defaults and shout about it in the log.
            logger.error("quota policy unreadable (%s) - falling back to defaults", exc)
            policy = QuotaPolicy()

        self._cached = policy
        self._mtime = mtime
        return policy

    def save(self, policy: QuotaPolicy, *, updated_by: str = "") -> QuotaPolicy:
        """Persist *policy* (normalised and timestamped)."""
        clean = policy.normalised()
        clean.updated_at = _utc_now_iso()
        if updated_by:
            clean.updated_by = updated_by
        elif not clean.updated_by:
            clean.updated_by = "agent3"
        self._store.save(clean.to_dict())
        self._cached = clean
        self._mtime = self._disk_mtime()
        return clean

    def reset_to_defaults(self, *, keep_password: bool = True) -> QuotaPolicy:
        """Restore factory quotas, optionally preserving the admin password."""
        current = self.load()
        fresh = QuotaPolicy()
        if keep_password and current.admin_password_hash:
            fresh.admin_password_hash = current.admin_password_hash
            fresh.admin_password_salt = current.admin_password_salt
            fresh.locked = current.locked
        return self.save(fresh, updated_by="reset")

    def changed_on_disk(self) -> bool:
        return self._disk_mtime() != self._mtime
