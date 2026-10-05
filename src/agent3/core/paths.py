"""Operating-system aware location resolution for Agent3 data files.

All mutable state (configuration, quota policy, usage database, logs, chat
sessions) lives in a single per-user application directory so that the frozen
executables never write next to themselves - a requirement when the ``.exe``
is installed into ``Program Files`` or launched from a read-only share.

The directory can be overridden with the ``AGENT3_HOME`` environment variable,
which is what the test-suite and the portable build use.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

APP_DIR_NAME = "Agent3"
ENV_HOME = "AGENT3_HOME"


def _platform_base_dir() -> Path:
    """Return the per-user base directory for the current platform."""
    if sys.platform.startswith("win"):
        root = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root) / APP_DIR_NAME
        return Path.home() / "AppData" / "Roaming" / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / APP_DIR_NAME.lower()
    return Path.home() / ".config" / APP_DIR_NAME.lower()


@dataclass(frozen=True)
class AppPaths:
    """Resolved absolute paths used across the application."""

    base: Path

    @property
    def config_file(self) -> Path:
        return self.base / "config.json"

    @property
    def policy_file(self) -> Path:
        """Encrypted quota policy shared with ``UsageLimitEditor.exe``."""
        return self.base / "limits.policy.enc"

    @property
    def credentials_file(self) -> Path:
        """Encrypted credential blob (GitHub token, etc.)."""
        return self.base / "credentials.enc"

    @property
    def usage_db(self) -> Path:
        return self.base / "usage.sqlite3"

    @property
    def sessions_db(self) -> Path:
        return self.base / "sessions.sqlite3"

    @property
    def log_dir(self) -> Path:
        return self.base / "logs"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "agent3.log"

    @property
    def crash_dir(self) -> Path:
        return self.base / "crashes"

    def ensure(self) -> "AppPaths":
        """Create every directory that the application expects to exist."""
        for directory in (self.base, self.log_dir, self.crash_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def app_paths() -> AppPaths:
    """Return the (cached) :class:`AppPaths` for this process."""
    override = os.environ.get(ENV_HOME)
    base = Path(override).expanduser() if override else _platform_base_dir()
    return AppPaths(base=base.resolve()).ensure()


def reset_app_paths_cache() -> None:
    """Clear the cached paths - used by tests that relocate ``AGENT3_HOME``."""
    app_paths.cache_clear()


def resource_dir() -> Path:
    """Return the directory holding bundled, read-only resources.

    Works both from source and from a PyInstaller one-file bundle (where the
    payload is unpacked into ``sys._MEIPASS``).
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "resources"
    return Path(__file__).resolve().parent.parent / "resources"


def is_frozen() -> bool:
    """True when running from a PyInstaller produced executable."""
    return bool(getattr(sys, "frozen", False))
