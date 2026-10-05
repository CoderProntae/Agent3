"""Typed application configuration with JSON persistence.

Non-sensitive settings live in ``config.json`` (human readable, easy to ship
with a team).  Secrets - currently the GitHub token - are kept in the
AES-GCM encrypted ``credentials.enc`` handled by :class:`SecureStore`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent3.core.logging_setup import get_logger
from agent3.core.paths import app_paths
from agent3.core.secure_store import SecureStore, SecureStoreError

logger = get_logger(__name__)

#: The product requirement: talk to Ollama on 11435, **not** the default 11434.
DEFAULT_OLLAMA_PORT = 11435
DEFAULT_OLLAMA_HOST = "localhost"
DEFAULT_MODEL = "qwen3.5-9b-abliterated"


def _coerce(cls: type, data: Any) -> Any:
    """Recursively build nested dataclasses from plain dictionaries."""
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _coerce(f.type, value)  # pragma: no cover - resolved below
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


@dataclass
class OllamaSettings:
    """Connection and sampling parameters for the local inference server."""

    scheme: str = "http"
    host: str = DEFAULT_OLLAMA_HOST
    port: int = DEFAULT_OLLAMA_PORT
    model: str = DEFAULT_MODEL
    fallback_models: List[str] = field(
        default_factory=lambda: ["qwen2.5-coder:7b", "qwen2.5:7b", "llama3.1:8b"]
    )
    #: Secondary endpoints tried when the primary refuses the connection.
    fallback_base_urls: List[str] = field(
        default_factory=lambda: ["http://127.0.0.1:11435", "http://localhost:11434"]
    )
    connect_timeout: float = 10.0
    read_timeout: float = 300.0
    max_retries: int = 4
    retry_backoff: float = 0.75
    keep_alive: str = "15m"
    stream: bool = True
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 16384
    num_predict: int = 4096
    stop: List[str] = field(default_factory=list)
    extra_headers: Dict[str, str] = field(default_factory=dict)
    #: Reasoning control sent as Ollama's ``think`` field. ``"auto"`` omits it
    #: (model default), ``"off"``/``"on"`` send booleans, and a level name such
    #: as ``"low"``/``"medium"``/``"high"`` is forwarded verbatim. The value is
    #: validated against ``/api/show`` before it reaches the wire, so an
    #: unsupported setting is silently dropped rather than failing the request.
    think: str = "auto"
    #: Show the reasoning trace in the chat transcript when the model emits one.
    show_thinking: bool = True

    @property
    def base_url(self) -> str:
        """Fully qualified base URL, e.g. ``http://localhost:11435``."""
        return f"{self.scheme}://{self.host}:{int(self.port)}"

    def endpoints(self) -> List[str]:
        """Primary endpoint followed by de-duplicated fallbacks."""
        ordered = [self.base_url, *self.fallback_base_urls]
        seen: set[str] = set()
        result: List[str] = []
        for url in ordered:
            clean = url.rstrip("/")
            if clean and clean not in seen:
                seen.add(clean)
                result.append(clean)
        return result

    def options(self) -> Dict[str, Any]:
        """Sampling options forwarded to Ollama's ``options`` field."""
        opts: Dict[str, Any] = {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "num_ctx": int(self.num_ctx),
            "num_predict": int(self.num_predict),
        }
        if self.stop:
            opts["stop"] = list(self.stop)
        return opts


@dataclass
class AgentSettings:
    """Behaviour of the autonomous loop."""

    max_iterations: int = 24
    self_correction_retries: int = 2
    auto_approve_tools: bool = True
    confirm_destructive: bool = True
    command_timeout: float = 240.0
    max_output_chars: int = 20_000
    max_file_read_bytes: int = 1_000_000
    history_window: int = 40
    #: Refuse `finish` while files changed in this run are still unverified.
    require_verification: bool = True
    #: How many times the loop pushes the model back to run its tests.
    verification_nudges: int = 1
    #: Refuse commands that obviously need a human, and kill the ones that
    #: print a question and then wait forever.
    detect_interactive_commands: bool = True
    #: Seconds of silence after a prompt-looking line before the command is
    #: treated as "waiting for input".
    interactive_idle_seconds: float = 15.0
    #: Parse/lint every file the agent writes and append the verdict to the
    #: tool result, so a syntax error is seen in the step that caused it.
    auto_syntax_check: bool = True
    #: Also run external linters (eslint/tsc/ruff) when they are installed.
    run_external_linters: bool = True
    #: Concurrently running background processes allowed per session.
    max_background_processes: int = 8
    #: Log lines retained per background process.
    process_log_lines: int = 2000
    #: How many pre-mutation file snapshots the undo buffer keeps.
    snapshot_history: int = 80
    #: Let the model keep a visible plan via ``manage_tasks``.
    enable_task_list: bool = True
    blocked_command_patterns: List[str] = field(
        default_factory=lambda: [
            r"rm\s+-rf\s+/",
            r":\(\)\s*\{.*\};:",
            r"mkfs(\.|\s)",
            r"shutdown\b",
            r"reboot\b",
            r"format\s+[a-zA-Z]:",
            r"del\s+/[sf]\s+/q\s+[a-zA-Z]:\\",
        ]
    )


@dataclass
class UiSettings:
    """Look & feel and remembered window state."""

    theme: str = "dark"
    accent: str = "#4f8cff"
    font_family: str = "Cascadia Code, Consolas, JetBrains Mono, monospace"
    font_size: int = 12
    ui_font_size: int = 10
    last_workspace: str = ""
    show_terminal: bool = True
    show_right_panel: bool = True
    window_state: str = ""
    window_geometry: str = ""


@dataclass
class GitHubSettings:
    """Repository synchronisation preferences (token lives encrypted)."""

    username: str = ""
    default_remote: str = "origin"
    default_branch: str = "main"
    auto_stage: bool = True
    auto_push: bool = False
    commit_prefix: str = "agent3:"


@dataclass
class AppConfig:
    """Root configuration document."""

    version: int = 1
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    github: GitHubSettings = field(default_factory=GitHubSettings)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            version=int(data.get("version", 1)),
            ollama=_coerce(OllamaSettings, data.get("ollama", {})) or OllamaSettings(),
            agent=_coerce(AgentSettings, data.get("agent", {})) or AgentSettings(),
            ui=_coerce(UiSettings, data.get("ui", {})) or UiSettings(),
            github=_coerce(GitHubSettings, data.get("github", {})) or GitHubSettings(),
        )


class ConfigManager:
    """Loads, validates and persists :class:`AppConfig`."""

    def __init__(self, path: Optional[Path] = None, credentials: Optional[SecureStore] = None):
        paths = app_paths()
        self._path = Path(path) if path else paths.config_file
        self._credentials = credentials or SecureStore(paths.credentials_file)
        self._config = self.load()

    # ----------------------------------------------------------------- data
    @property
    def path(self) -> Path:
        return self._path

    @property
    def config(self) -> AppConfig:
        return self._config

    def load(self) -> AppConfig:
        """Read ``config.json``; fall back to defaults on any problem."""
        if not self._path.is_file():
            self._config = AppConfig()
            self._apply_env_overrides()
            self.save()
            return self._config
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._config = AppConfig.from_dict(raw)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("config.json unreadable (%s) - using defaults", exc)
            self._config = AppConfig()
        self._apply_env_overrides()
        return self._config

    def save(self) -> None:
        """Atomically persist the current configuration."""
        payload = json.dumps(self._config.to_dict(), indent=2, ensure_ascii=False)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=str(self._path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp, self._path)
        finally:
            if os.path.exists(tmp):  # pragma: no cover
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def update(self, **sections: Any) -> AppConfig:
        """Replace whole configuration sections and persist immediately."""
        for name, value in sections.items():
            if not hasattr(self._config, name):
                raise KeyError(f"unknown configuration section: {name}")
            setattr(self._config, name, value)
        self.save()
        return self._config

    def _apply_env_overrides(self) -> None:
        """Allow CI / power users to override the endpoint via env vars."""
        host = os.environ.get("AGENT3_OLLAMA_HOST")
        port = os.environ.get("AGENT3_OLLAMA_PORT")
        model = os.environ.get("AGENT3_OLLAMA_MODEL")
        if host:
            self._config.ollama.host = host
        if port and port.isdigit():
            self._config.ollama.port = int(port)
        if model:
            self._config.ollama.model = model

    # ---------------------------------------------------------- credentials
    def get_secret(self, key: str, default: str = "") -> str:
        try:
            return str(self._credentials.load().get(key, default))
        except SecureStoreError as exc:
            logger.warning("credential store unreadable: %s", exc)
            return default

    def set_secret(self, key: str, value: str) -> None:
        try:
            data = self._credentials.load()
        except SecureStoreError:
            data = {}
        if value:
            data[key] = value
        else:
            data.pop(key, None)
        self._credentials.save(data)

    @property
    def github_token(self) -> str:
        return self.get_secret("github_token")

    @github_token.setter
    def github_token(self, value: str) -> None:
        self.set_secret("github_token", value)
