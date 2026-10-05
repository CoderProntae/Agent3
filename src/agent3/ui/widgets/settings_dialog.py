"""Settings dialog: Ollama endpoint, agent behaviour, GitHub and appearance."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from agent3.core.config import AppConfig, ConfigManager
from agent3.llm.ollama_client import OllamaClient
from agent3.ui.theme import COLORS
from agent3.ui.workers import HealthWorker


class SettingsDialog(QDialog):
    """Edits :class:`AppConfig` in place and persists it on *OK*."""

    def __init__(self, manager: ConfigManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._config: AppConfig = manager.config
        self._health_worker: Optional[HealthWorker] = None

        self.setWindowTitle("Agent3 Settings")
        self.setMinimumSize(620, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_connection_tab(), "Connection")
        self._tabs.addTab(self._build_agent_tab(), "Agent")
        self._tabs.addTab(self._build_github_tab(), "GitHub")
        self._tabs.addTab(self._build_appearance_tab(), "Appearance")
        layout.addWidget(self._tabs, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setObjectName("Primary")
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        layout.addWidget(buttons)

        self._load_values()

    # ------------------------------------------------------------- tabs
    def _build_connection_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setSpacing(12)

        endpoint_box = QGroupBox("Ollama endpoint")
        form = QFormLayout(endpoint_box)
        form.setSpacing(8)

        self._scheme = QComboBox()
        self._scheme.addItems(["http", "https"])
        form.addRow("Scheme", self._scheme)

        self._host = QLineEdit()
        form.addRow("Host", self._host)

        self._port = QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setToolTip("Agent3 targets port 11435 by default (not the stock 11434).")
        form.addRow("Port", self._port)

        model_row = QHBoxLayout()
        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.setMinimumWidth(260)
        model_row.addWidget(self._model, 1)
        self._refresh_models_button = QPushButton("Refresh")
        self._refresh_models_button.clicked.connect(self._test_connection)
        model_row.addWidget(self._refresh_models_button, 0)
        form.addRow("Model", model_row)

        self._fallback_urls = QLineEdit()
        self._fallback_urls.setPlaceholderText("http://127.0.0.1:11435, http://localhost:11434")
        self._fallback_urls.setToolTip("Comma separated endpoints tried when the primary is down.")
        form.addRow("Fallback URLs", self._fallback_urls)

        test_row = QHBoxLayout()
        self._test_button = QPushButton("Test connection")
        self._test_button.clicked.connect(self._test_connection)
        test_row.addWidget(self._test_button, 0)
        self._test_result = QLabel("")
        self._test_result.setWordWrap(True)
        self._test_result.setStyleSheet(f"color: {COLORS.text_faint};")
        test_row.addWidget(self._test_result, 1)
        form.addRow("", test_row)
        outer.addWidget(endpoint_box)

        transport_box = QGroupBox("Transport & sampling")
        transport = QFormLayout(transport_box)
        transport.setSpacing(8)

        self._connect_timeout = QDoubleSpinBox()
        self._connect_timeout.setRange(1.0, 120.0)
        self._connect_timeout.setSuffix(" s")
        transport.addRow("Connect timeout", self._connect_timeout)

        self._read_timeout = QDoubleSpinBox()
        self._read_timeout.setRange(5.0, 3600.0)
        self._read_timeout.setSuffix(" s")
        transport.addRow("Read timeout", self._read_timeout)

        self._max_retries = QSpinBox()
        self._max_retries.setRange(1, 10)
        transport.addRow("Max retries", self._max_retries)

        self._retry_backoff = QDoubleSpinBox()
        self._retry_backoff.setRange(0.1, 10.0)
        self._retry_backoff.setSingleStep(0.25)
        self._retry_backoff.setSuffix(" s")
        transport.addRow("Retry backoff base", self._retry_backoff)

        self._keep_alive = QLineEdit()
        self._keep_alive.setToolTip("How long Ollama keeps the model in memory, e.g. 15m or -1.")
        transport.addRow("Keep alive", self._keep_alive)

        self._stream = QCheckBox("Stream responses token by token")
        transport.addRow("", self._stream)

        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.0, 2.0)
        self._temperature.setSingleStep(0.05)
        transport.addRow("Temperature", self._temperature)

        self._num_ctx = QSpinBox()
        self._num_ctx.setRange(512, 1_048_576)
        self._num_ctx.setSingleStep(1024)
        transport.addRow("Context window", self._num_ctx)

        self._num_predict = QSpinBox()
        self._num_predict.setRange(64, 131_072)
        self._num_predict.setSingleStep(256)
        transport.addRow("Max new tokens", self._num_predict)

        outer.addWidget(transport_box)
        outer.addStretch(1)
        return page

    def _build_agent_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        box = QGroupBox("Autonomous loop")
        form = QFormLayout(box)
        form.setSpacing(8)

        self._max_iterations = QSpinBox()
        self._max_iterations.setRange(1, 200)
        form.addRow("Max steps per run", self._max_iterations)

        self._self_correction = QSpinBox()
        self._self_correction.setRange(0, 10)
        form.addRow("Self-correction retries", self._self_correction)

        self._history_window = QSpinBox()
        self._history_window.setRange(6, 200)
        form.addRow("History window (messages)", self._history_window)

        self._command_timeout = QDoubleSpinBox()
        self._command_timeout.setRange(5.0, 3600.0)
        self._command_timeout.setSuffix(" s")
        form.addRow("Command timeout", self._command_timeout)

        self._max_output = QSpinBox()
        self._max_output.setRange(1000, 200_000)
        self._max_output.setSingleStep(1000)
        form.addRow("Max tool output (chars)", self._max_output)

        self._auto_approve = QCheckBox("Execute tool calls without asking")
        form.addRow("", self._auto_approve)

        self._confirm_destructive = QCheckBox("Confirm destructive operations (delete, rename)")
        form.addRow("", self._confirm_destructive)

        outer.addWidget(box)
        outer.addStretch(1)
        return page

    def _build_github_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        box = QGroupBox("Repository synchronisation")
        form = QFormLayout(box)
        form.setSpacing(8)

        self._gh_username = QLineEdit()
        form.addRow("Username", self._gh_username)

        self._gh_token = QLineEdit()
        self._gh_token.setEchoMode(QLineEdit.EchoMode.Password)
        self._gh_token.setPlaceholderText("ghp_... (stored AES-256-GCM encrypted)")
        form.addRow("Personal access token", self._gh_token)

        self._gh_remote = QLineEdit()
        form.addRow("Default remote", self._gh_remote)

        self._gh_branch = QLineEdit()
        form.addRow("Default branch", self._gh_branch)

        self._gh_prefix = QLineEdit()
        form.addRow("Commit message prefix", self._gh_prefix)

        self._gh_auto_stage = QCheckBox("Stage all changes before committing")
        form.addRow("", self._gh_auto_stage)

        self._gh_auto_push = QCheckBox("Push automatically after a successful run")
        form.addRow("", self._gh_auto_push)

        note = QLabel(
            "The token never leaves this machine: it is encrypted at rest and only used for "
            "git push/pull from the workspace."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLORS.text_faint};")
        form.addRow("", note)

        outer.addWidget(box)
        outer.addStretch(1)
        return page

    def _build_appearance_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        box = QGroupBox("Look & feel")
        form = QFormLayout(box)
        form.setSpacing(8)

        self._accent = QLineEdit()
        self._accent.setPlaceholderText("#4f8cff")
        form.addRow("Accent colour", self._accent)

        self._ui_font_size = QSpinBox()
        self._ui_font_size.setRange(8, 18)
        form.addRow("Interface font size", self._ui_font_size)

        self._font_size = QSpinBox()
        self._font_size.setRange(8, 24)
        form.addRow("Editor font size", self._font_size)

        note = QLabel("Restart Agent3 to apply font and accent changes everywhere.")
        note.setStyleSheet(f"color: {COLORS.text_faint};")
        form.addRow("", note)

        outer.addWidget(box)
        outer.addStretch(1)
        return page

    # ------------------------------------------------------------ values
    def _load_values(self) -> None:
        config = self._config
        ollama = config.ollama
        self._scheme.setCurrentText(ollama.scheme)
        self._host.setText(ollama.host)
        self._port.setValue(int(ollama.port))
        self._model.setEditText(ollama.model)
        self._fallback_urls.setText(", ".join(ollama.fallback_base_urls))
        self._connect_timeout.setValue(float(ollama.connect_timeout))
        self._read_timeout.setValue(float(ollama.read_timeout))
        self._max_retries.setValue(int(ollama.max_retries))
        self._retry_backoff.setValue(float(ollama.retry_backoff))
        self._keep_alive.setText(ollama.keep_alive)
        self._stream.setChecked(bool(ollama.stream))
        self._temperature.setValue(float(ollama.temperature))
        self._num_ctx.setValue(int(ollama.num_ctx))
        self._num_predict.setValue(int(ollama.num_predict))

        agent = config.agent
        self._max_iterations.setValue(int(agent.max_iterations))
        self._self_correction.setValue(int(agent.self_correction_retries))
        self._history_window.setValue(int(agent.history_window))
        self._command_timeout.setValue(float(agent.command_timeout))
        self._max_output.setValue(int(agent.max_output_chars))
        self._auto_approve.setChecked(bool(agent.auto_approve_tools))
        self._confirm_destructive.setChecked(bool(agent.confirm_destructive))

        github = config.github
        self._gh_username.setText(github.username)
        self._gh_token.setText(self._manager.github_token)
        self._gh_remote.setText(github.default_remote)
        self._gh_branch.setText(github.default_branch)
        self._gh_prefix.setText(github.commit_prefix)
        self._gh_auto_stage.setChecked(bool(github.auto_stage))
        self._gh_auto_push.setChecked(bool(github.auto_push))

        ui = config.ui
        self._accent.setText(ui.accent)
        self._ui_font_size.setValue(int(ui.ui_font_size))
        self._font_size.setValue(int(ui.font_size))

    def _apply_values(self) -> None:
        config = self._config
        ollama = config.ollama
        ollama.scheme = self._scheme.currentText().strip() or "http"
        ollama.host = self._host.text().strip() or "localhost"
        ollama.port = int(self._port.value())
        ollama.model = self._model.currentText().strip() or ollama.model
        ollama.fallback_base_urls = [
            url.strip() for url in self._fallback_urls.text().split(",") if url.strip()
        ]
        ollama.connect_timeout = float(self._connect_timeout.value())
        ollama.read_timeout = float(self._read_timeout.value())
        ollama.max_retries = int(self._max_retries.value())
        ollama.retry_backoff = float(self._retry_backoff.value())
        ollama.keep_alive = self._keep_alive.text().strip() or "15m"
        ollama.stream = bool(self._stream.isChecked())
        ollama.temperature = float(self._temperature.value())
        ollama.num_ctx = int(self._num_ctx.value())
        ollama.num_predict = int(self._num_predict.value())

        agent = config.agent
        agent.max_iterations = int(self._max_iterations.value())
        agent.self_correction_retries = int(self._self_correction.value())
        agent.history_window = int(self._history_window.value())
        agent.command_timeout = float(self._command_timeout.value())
        agent.max_output_chars = int(self._max_output.value())
        agent.auto_approve_tools = bool(self._auto_approve.isChecked())
        agent.confirm_destructive = bool(self._confirm_destructive.isChecked())

        github = config.github
        github.username = self._gh_username.text().strip()
        github.default_remote = self._gh_remote.text().strip() or "origin"
        github.default_branch = self._gh_branch.text().strip() or "main"
        github.commit_prefix = self._gh_prefix.text().strip()
        github.auto_stage = bool(self._gh_auto_stage.isChecked())
        github.auto_push = bool(self._gh_auto_push.isChecked())

        ui = config.ui
        ui.accent = self._accent.text().strip() or "#4f8cff"
        ui.ui_font_size = int(self._ui_font_size.value())
        ui.font_size = int(self._font_size.value())

        self._manager.save()
        self._manager.github_token = self._gh_token.text().strip()

    def _restore_defaults(self) -> None:
        self._config = AppConfig()
        self._manager.update(
            ollama=self._config.ollama,
            agent=self._config.agent,
            ui=self._config.ui,
            github=self._config.github,
        )
        self._config = self._manager.config
        self._load_values()

    def _on_accept(self) -> None:
        self._apply_values()
        self.accept()

    # ------------------------------------------------------------ health
    def _test_connection(self) -> None:
        self._test_result.setText("Contacting server...")
        self._test_result.setStyleSheet(f"color: {COLORS.text_faint};")
        self._test_button.setEnabled(False)
        self._refresh_models_button.setEnabled(False)

        probe = AppConfig().ollama
        probe.scheme = self._scheme.currentText()
        probe.host = self._host.text().strip() or "localhost"
        probe.port = int(self._port.value())
        probe.connect_timeout = float(self._connect_timeout.value())
        probe.read_timeout = 20.0
        probe.max_retries = 1
        probe.fallback_base_urls = [
            url.strip() for url in self._fallback_urls.text().split(",") if url.strip()
        ]

        worker = HealthWorker(OllamaClient(probe), parent=self)
        worker.completed.connect(self._on_health)
        worker.finished.connect(lambda: self._set_test_enabled(True))
        self._health_worker = worker
        worker.start()

    def _set_test_enabled(self, enabled: bool) -> None:
        self._test_button.setEnabled(enabled)
        self._refresh_models_button.setEnabled(enabled)

    def _on_health(self, payload: dict) -> None:
        if payload.get("ok"):
            models = payload.get("models", [])
            current = self._model.currentText()
            self._model.clear()
            self._model.addItems(models)
            self._model.setEditText(current)
            version = payload.get("version") or "unknown"
            self._test_result.setText(
                f"Connected to {payload.get('endpoint')} (v{version}) - {len(models)} model(s)."
            )
            self._test_result.setStyleSheet(f"color: {COLORS.success};")
        else:
            self._test_result.setText(str(payload.get("error", "connection failed"))[:300])
            self._test_result.setStyleSheet(f"color: {COLORS.danger};")
