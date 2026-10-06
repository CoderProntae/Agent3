"""The Agent3 main window - the multi-pane workspace shell.

Layout
------
::

    +----------------------------------------------------------------+
    | toolbar: workspace | model | connection | Run / Stop            |
    +------------+--------------------------------+------------------+
    | Explorer   |  Chat + action cards           |  Editor / Diff   |
    | Sessions   |                                |                  |
    | Usage      +--------------------------------+                  |
    |            |  composer (Ctrl+Enter to send) |                  |
    +------------+--------------------------------+------------------+
    | Terminal console                                               |
    +----------------------------------------------------------------+
    | status bar: agent state | tokens | quota                       |
    +----------------------------------------------------------------+
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QByteArray, Qt, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence, QTextOption
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from agent3 import APP_NAME, __version__
from agent3.agent.loop import AgentLoop, AgentRunResult, AgentStopReason
from agent3.agent.session import SessionStore
from agent3.agent.tools import ToolCall, ToolContext, ToolRegistry, ToolResult
from agent3.core.config import ConfigManager
from agent3.core.logging_setup import get_logger
from agent3.core.paths import app_paths, is_frozen
from agent3.limits.manager import UsageManager
from agent3.llm.messages import Role
from agent3.llm.ollama_client import OllamaClient
from agent3.ui.theme import COLORS, mono_font
from agent3.ui.widgets.chat_view import ActionCard, ChatView
from agent3.ui.widgets.editor_panel import EditorPanel
from agent3.ui.widgets.file_tree import WorkspaceTree
from agent3.ui.widgets.session_list import SessionList
from agent3.ui.widgets.settings_dialog import SettingsDialog
from agent3.ui.widgets.terminal_panel import TerminalPanel
from agent3.ui.widgets.usage_panel import UsagePanel
from agent3.agent.tasks import TaskList
from agent3.workspace.processes import ProcessManager
from agent3.workspace.snapshots import SnapshotStore
from agent3.ui.widgets.composer_bar import ComposerBar
from agent3.ui.widgets.task_panel import TaskPanel
from agent3.ui.workers import AgentWorker, HealthWorker, ModelCapabilityWorker
from agent3.workspace.fs import WorkspaceFS
from agent3.workspace.git_ops import GitRepo
from agent3.workspace.terminal import CommandRunner

logger = get_logger(__name__)


class Composer(QPlainTextEdit):
    """Multi-line prompt box that sends on Ctrl+Enter."""

    def __init__(self, on_submit, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_submit = on_submit
        self.setPlaceholderText(
            "Describe the task... (Ctrl+Enter to run, Shift+Enter for a new line)"
        )
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setMaximumHeight(140)
        self.setMinimumHeight(70)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        ctrl = event.modifiers() & Qt.KeyboardModifier.ControlModifier
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and ctrl:
            self._on_submit()
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    """Top level window wiring every subsystem together."""

    def __init__(self, config_manager: Optional[ConfigManager] = None) -> None:
        super().__init__()
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.config

        self.client = OllamaClient(self.config.ollama)
        self.registry = ToolRegistry()
        self.usage = UsageManager()
        self.sessions = SessionStore()

        self.fs: Optional[WorkspaceFS] = None
        self.runner: Optional[CommandRunner] = None
        self.git: Optional[GitRepo] = None
        self.tool_context: Optional[ToolContext] = None
        self.processes: Optional[ProcessManager] = None
        self.loop: Optional[AgentLoop] = None
        self.worker: Optional[AgentWorker] = None
        self._health_worker: Optional[HealthWorker] = None
        self._capability_worker: Optional[ModelCapabilityWorker] = None
        self._active_cards: dict[int, ActionCard] = {}
        self._card_counter = 0
        #: Reasoning characters swallowed while the trace is hidden,
        #: surfaced in the status bar so a long silent think is visible.
        self._hidden_thinking_chars = 0
        self._current_session_id = ""

        self.setWindowTitle(f"{APP_NAME} - Autonomous Coding Workspace")
        self.resize(1680, 980)
        self.setMinimumSize(1100, 680)

        self._build_ui()
        self._build_actions()
        self._restore_window_state()

        self._usage_timer = QTimer(self)
        self._usage_timer.setInterval(2500)
        self._usage_timer.timeout.connect(self.refresh_usage)
        self._usage_timer.start()

        QTimer.singleShot(60, self._bootstrap)

    # =================================================================== UI
    def _build_ui(self) -> None:
        self.toolbar = QToolBar("Main")
        self.toolbar.setObjectName("MainToolBar")
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        self._workspace_label = QLabel("No workspace mounted")
        self._workspace_label.setStyleSheet(f"color: {COLORS.text_dim}; padding: 0 8px;")
        self.toolbar.addWidget(self._workspace_label)
        self.toolbar.addSeparator()

        # The model picker and the reasoning controls deliberately do NOT live
        # here: they belong next to the box you type in, so they are built as
        # part of the composer strip further down.

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)

        # ------------------------------------------------------- sidebar
        self.tree = WorkspaceTree()
        self.tree.file_opened.connect(self._open_file)

        self.session_list = SessionList()
        self.session_list.session_created.connect(self.new_session)
        self.session_list.session_selected.connect(self.load_session)
        self.session_list.session_deleted.connect(self._delete_session)
        self.session_list.session_renamed.connect(self._rename_session)

        self.usage_panel = UsagePanel()
        self.usage_panel.editor_button.clicked.connect(self.open_usage_editor)

        #: Live view of the plan the agent maintains with `manage_tasks`.
        self.task_panel = TaskPanel()

        explorer_header = QLabel("EXPLORER")
        explorer_header.setObjectName("SectionTitle")

        sidebar = QSplitter(Qt.Orientation.Vertical)
        explorer_box = QWidget()
        explorer_layout = QVBoxLayout(explorer_box)
        explorer_layout.setContentsMargins(0, 0, 0, 0)
        explorer_layout.setSpacing(0)
        explorer_layout.addWidget(explorer_header)
        explorer_layout.addWidget(self.tree, 1)
        sidebar.addWidget(explorer_box)
        sidebar.addWidget(self.task_panel)
        sidebar.addWidget(self.session_list)
        sidebar.addWidget(self.usage_panel)
        sidebar.setSizes([330, 250, 170, 290])
        sidebar.setMinimumWidth(240)

        # ---------------------------------------------------------- chat
        self.quota_banner = QFrame()
        self.quota_banner.setVisible(False)
        self.quota_banner.setStyleSheet(
            f"background-color: #2a2113; border: 1px solid {COLORS.warning}; border-radius: 6px;"
        )
        banner_layout = QHBoxLayout(self.quota_banner)
        banner_layout.setContentsMargins(12, 7, 12, 7)
        self._banner_label = QLabel("")
        self._banner_label.setWordWrap(True)
        self._banner_label.setStyleSheet(f"color: {COLORS.warning};")
        banner_layout.addWidget(self._banner_label, 1)
        banner_button = QPushButton("Manage limits")
        banner_button.setObjectName("Ghost")
        banner_button.clicked.connect(self.open_usage_editor)
        banner_layout.addWidget(banner_button, 0)

        self.chat = ChatView()

        self.composer = Composer(self.send_message)
        self.composer.setFont(mono_font(10))

        # Everything you tune per message sits directly under the input box.
        self.composer_bar = ComposerBar()
        self.composer_bar.set_model(self.config.ollama.model)
        self.composer_bar.set_think_spec(self.config.ollama.think)
        self.composer_bar.model_changed.connect(self._on_model_changed)
        self.composer_bar.think_changed.connect(self._on_think_changed)

        composer_row = QHBoxLayout()
        composer_row.setSpacing(8)
        self._run_button = QPushButton("Run agent  (Ctrl+Enter)")
        self._run_button.setObjectName("Primary")
        self._run_button.clicked.connect(self.send_message)
        self._stop_button = QPushButton("Stop")
        self._stop_button.setObjectName("Danger")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop_agent)
        self._status_chip = QLabel("idle")
        self._status_chip.setObjectName("StatusBadge")
        composer_row.addWidget(self._status_chip, 0)
        composer_row.addStretch(1)
        composer_row.addWidget(self._stop_button, 0)
        composer_row.addWidget(self._run_button, 0)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)
        center_layout.addWidget(self.quota_banner)
        center_layout.addWidget(self.chat, 1)
        center_layout.addWidget(self.composer, 0)
        center_layout.addWidget(self.composer_bar, 0)
        center_layout.addLayout(composer_row)

        # -------------------------------------------------------- editor
        self.editor = EditorPanel()

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(sidebar)
        self.main_splitter.addWidget(center)
        self.main_splitter.addWidget(self.editor)
        # The conversation is the primary surface, so it gets the largest
        # share and grows fastest; the editor/diff pane is a companion view.
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 5)
        self.main_splitter.setStretchFactor(2, 3)
        self.main_splitter.setSizes([280, 830, 570])

        # ------------------------------------------------------ terminal
        self.terminal = TerminalPanel()

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.addWidget(self.main_splitter)
        self.vertical_splitter.addWidget(self.terminal)
        self.vertical_splitter.setStretchFactor(0, 4)
        self.vertical_splitter.setStretchFactor(1, 1)
        self.vertical_splitter.setSizes([720, 220])
        self.setCentralWidget(self.vertical_splitter)

        # ----------------------------------------------------- statusbar
        status = QStatusBar()
        self.setStatusBar(status)
        self._status_text = QLabel("Ready")
        self._status_tokens = QLabel("tokens: 0")
        self._status_quota = QLabel("quota: -")
        for widget in (self._status_text,):
            status.addWidget(widget, 1)
        for widget in (self._status_tokens, self._status_quota):
            widget.setStyleSheet(f"color: {COLORS.text_faint}; padding: 0 10px;")
            status.addPermanentWidget(widget, 0)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open workspace folder...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.choose_workspace)
        file_menu.addAction(open_action)

        save_action = QAction("Save file", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.editor.save_current)
        file_menu.addAction(save_action)

        save_all_action = QAction("Save all", self)
        save_all_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_all_action.triggered.connect(self.editor.save_all)
        file_menu.addAction(save_all_action)

        file_menu.addSeparator()
        settings_action = QAction("Settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()
        quit_action = QAction("Exit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        agent_menu = self.menuBar().addMenu("&Agent")
        run_action = QAction("Run agent", self)
        run_action.setShortcut(QKeySequence("Ctrl+Return"))
        run_action.triggered.connect(self.send_message)
        agent_menu.addAction(run_action)

        stop_action = QAction("Stop agent", self)
        stop_action.setShortcut(QKeySequence("Esc"))
        stop_action.triggered.connect(self.stop_agent)
        agent_menu.addAction(stop_action)

        agent_menu.addSeparator()
        new_session_action = QAction("New session", self)
        new_session_action.setShortcut(QKeySequence("Ctrl+N"))
        new_session_action.triggered.connect(self.new_session)
        agent_menu.addAction(new_session_action)

        export_action = QAction("Export session to markdown...", self)
        export_action.triggered.connect(self.export_session)
        agent_menu.addAction(export_action)

        git_menu = self.menuBar().addMenu("&Git")
        for label, handler in (
            ("Initialise repository", self._git_init),
            ("Status", self._git_status),
            ("Commit all changes...", self._git_commit),
            ("Push", self._git_push),
        ):
            action = QAction(label, self)
            action.triggered.connect(handler)
            git_menu.addAction(action)

        view_menu = self.menuBar().addMenu("&View")
        self._toggle_terminal_action = QAction("Terminal", self, checkable=True, checked=True)
        self._toggle_terminal_action.setShortcut(QKeySequence("Ctrl+`"))
        self._toggle_terminal_action.toggled.connect(self.terminal.setVisible)
        view_menu.addAction(self._toggle_terminal_action)

        self._toggle_editor_action = QAction("Editor panel", self, checkable=True, checked=True)
        self._toggle_editor_action.toggled.connect(self.editor.setVisible)
        view_menu.addAction(self._toggle_editor_action)

        tools_menu = self.menuBar().addMenu("&Tools")
        usage_action = QAction("Usage Limit Editor", self)
        usage_action.triggered.connect(self.open_usage_editor)
        tools_menu.addAction(usage_action)

        logs_action = QAction("Open log folder", self)
        logs_action.triggered.connect(self._open_logs)
        tools_menu.addAction(logs_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("About Agent3", self)
        about_action.triggered.connect(self._about)
        help_menu.addAction(about_action)

    # ============================================================ startup
    def _bootstrap(self) -> None:
        last = self.config.ui.last_workspace
        if last and Path(last).is_dir():
            self.open_workspace(last)
        else:
            self.chat.add_notice(
                "Open a workspace folder (Ctrl+O) to let the agent read and write code.",
                "system",
            )
        self._refresh_sessions()
        if not self._current_session_id:
            self.new_session(silent=True)
        self.check_connection()
        self.refresh_usage()

    def check_connection(self) -> None:
        """Ping Ollama in the background and update the toolbar badge."""
        self.composer_bar.set_connection(False, "connecting…")
        self.composer_bar.connection_label.setStyleSheet(
            f"color: {COLORS.warning}; font-size: 11px;"
        )
        worker = HealthWorker(self.client, parent=self)
        worker.completed.connect(self._on_health)
        self._health_worker = worker
        worker.start()

    @Slot(dict)
    def _on_health(self, payload: dict) -> None:
        if payload.get("ok"):
            models = payload.get("models", [])
            current = self.composer_bar.current_model() or self.config.ollama.model
            self.composer_bar.set_models(models, current)
            endpoint = payload.get("endpoint", self.config.ollama.base_url)
            version = payload.get("version", "")
            self.composer_bar.set_connection(
                True,
                f"online · {endpoint}",
                f"Ollama {version} at {endpoint}" if version else str(endpoint),
            )
            self.refresh_model_capabilities(current)
            if current and models and current not in models:
                self.chat.add_notice(
                    f"Model '{current}' is not installed on the server. "
                    f"Available: {', '.join(models[:6])}",
                    "error",
                )
        else:
            self.composer_bar.set_connection(
                False,
                f"offline · {self.config.ollama.base_url}",
                str(payload.get("error", "")),
            )
            self.chat.add_notice(
                f"Cannot reach Ollama at {self.config.ollama.base_url}. "
                "Start it with: OLLAMA_HOST=127.0.0.1:11435 ollama serve",
                "error",
            )

    def refresh_model_capabilities(self, model: str = "") -> None:
        """Ask the server what the selected model supports, then rebuild the
        reasoning controls from the answer - never from a guess."""
        name = (model or self.composer_bar.current_model()).strip()
        if not name:
            return
        if self._capability_worker is not None and self._capability_worker.isRunning():
            return
        worker = ModelCapabilityWorker(self.client, name, parent=self)
        worker.completed.connect(self._on_capabilities)
        self._capability_worker = worker
        worker.start()

    @Slot(str, object)
    def _on_capabilities(self, model: str, support) -> None:
        if model.strip() != self.composer_bar.current_model().strip():
            return  # the user moved on to another model while we were asking
        self.composer_bar.set_thinking_support(support, self.config.ollama.think)
        spec = self.composer_bar.think_spec()
        if spec != self.config.ollama.think:
            self.config.ollama.think = spec
            self.config_manager.save()
        if self.loop is not None:
            self.loop.think = spec

    def _on_think_changed(self, spec: str) -> None:
        """Persist the reasoning setting and apply it to the running loop."""
        self.config.ollama.think = spec
        self.client.settings.think = spec
        self.config_manager.save()
        if self.loop is not None:
            self.loop.think = spec
        self.statusBar().showMessage(self.composer_bar.thinking_summary(), 2500)

    # ========================================================== workspace
    def choose_workspace(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select workspace folder", self.config.ui.last_workspace or str(Path.home())
        )
        if directory:
            self.open_workspace(directory)

    def open_workspace(self, path: str) -> None:
        """Mount *path* as the agent's sandbox and rebuild every panel."""
        try:
            fs = WorkspaceFS(path, max_read_bytes=self.config.agent.max_file_read_bytes)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Workspace", f"Cannot open {path}:\n{exc}")
            return

        self.fs = fs
        self.runner = CommandRunner(
            fs.root,
            blocked_patterns=self.config.agent.blocked_command_patterns,
            default_timeout=self.config.agent.command_timeout,
            detect_interactive=self.config.agent.detect_interactive_commands,
            interactive_idle_seconds=self.config.agent.interactive_idle_seconds,
        )
        self.git = GitRepo(fs.root)
        self.processes = ProcessManager(
            self.runner,
            max_processes=self.config.agent.max_background_processes,
            max_log_lines=self.config.agent.process_log_lines,
        )
        self.tool_context = ToolContext(
            fs=fs,
            runner=self.runner,
            git=self.git,
            command_timeout=self.config.agent.command_timeout,
            max_output_chars=self.config.agent.max_output_chars,
            processes=self.processes,
            snapshots=SnapshotStore(max_entries=self.config.agent.snapshot_history),
            tasks=TaskList(),
            auto_syntax_check=self.config.agent.auto_syntax_check,
            run_external_linters=self.config.agent.run_external_linters,
        )
        self.loop = AgentLoop(
            self.client,
            self.registry,
            self.tool_context,
            self.usage,
            self.config.agent,
            model=self.composer_bar.current_model() or self.config.ollama.model,
        )
        self.loop.think = self.composer_bar.think_spec()

        self.tree.set_workspace(fs)
        self.editor.set_workspace(fs)
        self.terminal.set_runner(self.runner)
        self.task_panel.clear()
        self._update_process_badge()
        self._workspace_label.setText(f"  {fs.root}")
        self.setWindowTitle(f"{APP_NAME} - {fs.root.name}")

        self.config.ui.last_workspace = str(fs.root)
        self.config_manager.save()

        branch = self.git.current_branch() if self.git.is_repo() else ""
        suffix = f" (git: {branch})" if branch else " (not a git repository)"
        self.chat.add_notice(f"Workspace mounted: {fs.root}{suffix}", "success")
        self.statusBar().showMessage(f"Workspace: {fs.root}", 5000)

    def _open_file(self, relative: str) -> None:
        self.editor.open_file(relative)

    # ============================================================ sessions
    def _refresh_sessions(self) -> None:
        self.session_list.set_sessions(self.sessions.list(), self._current_session_id)

    def new_session(self, silent: bool = False) -> None:
        session = self.sessions.create(
            workspace=str(self.fs.root) if self.fs else "",
            model=self.composer_bar.current_model(),
        )
        self._current_session_id = session.id
        self.usage.start_session(session.id)
        if self.loop is not None:
            self.loop.reset()
        self.chat.clear()
        self._refresh_sessions()
        if not silent:
            self.chat.add_notice("New session started - token budget reset.", "system")
        self.refresh_usage()

    def load_session(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        self._current_session_id = session_id
        self.usage.start_session(session_id)
        self.chat.clear()
        messages = self.sessions.messages(session_id)
        for message in messages:
            if message.role is Role.USER:
                self.chat.add_message("user", message.content)
            elif message.role is Role.ASSISTANT:
                self.chat.add_message("assistant", message.content)
            elif message.role is Role.TOOL:
                card = self.chat.add_action_card(message.metadata.get("tool", "tool"))
                card.set_status(message.metadata.get("status", "success"))
                card.set_details(message.content)
        if self.loop is not None:
            self.loop.load_history(messages)
        self._refresh_sessions()
        self.refresh_usage()

    def _delete_session(self, session_id: str) -> None:
        self.sessions.delete(session_id)
        if session_id == self._current_session_id:
            self.new_session(silent=True)
        else:
            self._refresh_sessions()

    def _rename_session(self, session_id: str, title: str) -> None:
        self.sessions.rename(session_id, title)
        self._refresh_sessions()

    def export_session(self) -> None:
        if not self._current_session_id:
            return
        target, _ = QFileDialog.getSaveFileName(
            self, "Export session", str(Path.home() / "agent3-session.md"), "Markdown (*.md)"
        )
        if not target:
            return
        try:
            Path(target).write_text(
                self.sessions.export_markdown(self._current_session_id), encoding="utf-8"
            )
            self.statusBar().showMessage(f"Exported to {target}", 5000)
        except OSError as exc:
            QMessageBox.critical(self, "Export", str(exc))

    # =============================================================== agent
    def send_message(self) -> None:
        """Validate preconditions and start an agent run."""
        text = self.composer.toPlainText().strip()
        if not text:
            return
        if self.fs is None or self.loop is None:
            QMessageBox.information(
                self, "No workspace", "Open a workspace folder first (Ctrl+O)."
            )
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Agent busy", "The agent is already running.")
            return

        decision = self.usage.check_request(0)
        if not decision.allowed and decision.code != "cooldown":
            self._show_banner(decision.reason, blocking=True)
            QMessageBox.warning(self, "Usage limit reached", decision.reason)
            return

        self.composer.clear()
        self.chat.add_message("user", text)
        if self._current_session_id:
            self.sessions.add_message(self._current_session_id, Role.USER, text)
            session = self.sessions.get(self._current_session_id)
            if session and session.title == "New session":
                self.sessions.autoname(self._current_session_id, text)
                self._refresh_sessions()

        self.loop.model = self.composer_bar.current_model() or self.config.ollama.model
        self.loop.settings = self.config.agent
        self._active_cards.clear()
        self._hidden_thinking_chars = 0
        self._set_running(True)

        worker = AgentWorker(self.loop, text, parent=self)
        worker.status_changed.connect(self._on_status)
        worker.assistant_delta.connect(self.chat.append_assistant_delta)
        worker.thinking_delta.connect(self._on_thinking_delta)
        worker.tasks_changed.connect(self._on_tasks_changed)
        worker.processes_changed.connect(self._update_process_badge)
        worker.assistant_message.connect(self._on_assistant_message)
        worker.tool_started.connect(self._on_tool_started)
        worker.tool_finished.connect(self._on_tool_finished)
        worker.command_output.connect(self.terminal.append_stream)
        worker.file_changed.connect(self._on_file_changed)
        worker.usage_updated.connect(self._on_usage_updated)
        worker.error_raised.connect(self._on_error)
        worker.finished_run.connect(self._on_run_finished)
        self.worker = worker
        worker.start()

    def stop_agent(self) -> None:
        if self.processes is not None and self.processes.running_count:
            stopped = self.processes.stop_all()
            logger.info("stopped %d background process(es) on exit", len(stopped))

        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self._on_status("stopping...")

    def _set_running(self, running: bool) -> None:
        self.composer_bar.set_busy(running)
        self._run_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self.composer.setReadOnly(running)
        self._status_chip.setText("running" if running else "idle")
        self._status_chip.setStyleSheet(
            f"color: {COLORS.warning if running else COLORS.text_dim};"
            f" background-color: {COLORS.panel_alt}; border: 1px solid {COLORS.border};"
            " border-radius: 9px; padding: 2px 8px;"
        )

    # ------------------------------------------------------- agent slots
    @Slot(str)
    def _on_status(self, text: str) -> None:
        self._status_text.setText(text)

    @Slot(str)
    def _on_thinking_delta(self, delta: str) -> None:
        """Stream the reasoning trace into its own collapsible block.

        When the trace is hidden the tokens must not vanish silently: a model
        can reason for a minute before its first word of answer, and an
        application that shows nothing for that minute looks frozen. The
        character counter in the status bar is the proof of life.
        """
        if not self.config.ollama.show_thinking:
            self._hidden_thinking_chars += len(delta)
            self._status_text.setText(
                f"Reasoning... ({self._hidden_thinking_chars:,} characters so far)"
            )
            return
        self.chat.append_thinking_delta(delta)

    @Slot(object)
    def _on_tasks_changed(self, tasks) -> None:
        """Mirror the agent's plan into the sidebar."""
        try:
            items = list(tasks)
        except TypeError:  # pragma: no cover - defensive
            return
        self.task_panel.set_tasks(items)
        summary = tasks.summary_line() if hasattr(tasks, "summary_line") else ""
        if summary:
            self.statusBar().showMessage(f"Plan: {summary}", 4000)

    @Slot()
    def _update_process_badge(self) -> None:
        """Show how many background processes the agent left running."""
        count = self.processes.running_count if self.processes is not None else 0
        self.terminal.set_process_count(count)

    @Slot(str)
    def _on_assistant_message(self, text: str) -> None:
        self.chat.end_assistant_stream(text)
        if self._current_session_id and text.strip():
            self.sessions.add_message(self._current_session_id, Role.ASSISTANT, text)

    @Slot(object)
    def _on_tool_started(self, call: ToolCall) -> None:
        self.chat.end_assistant_stream()
        subtitle = self._describe_call(call)
        card = self.chat.add_action_card(f"[AGENT] {call.name}", subtitle)
        self._card_counter += 1
        self._active_cards[id(call)] = card
        if call.name == "run_command":
            self.terminal.append_command(str(call.args.get("command", "")))

    @Slot(object, object)
    def _on_tool_finished(self, call: ToolCall, result: ToolResult) -> None:
        card = self._active_cards.pop(id(call), None)
        if card is None:
            card = self.chat.add_action_card(f"[AGENT] {call.name}", self._describe_call(call))
        blocked = bool(result.data.get("blocked") or result.data.get("interactive"))
        status = "success" if result.ok else ("blocked" if blocked else "error")
        card.set_status(status, f"{result.duration_ms} ms")
        card.set_subtitle(result.output.splitlines()[0][:160] if result.ok and result.output else
                          (result.error.splitlines()[0][:160] if result.error else ""))
        details = result.output if result.ok else (result.error or "")
        if result.diff:
            # A file change gets the review view: only the changed lines, with
            # old/new line numbers, green additions and red deletions.
            card.set_diff(result.diff)
            details = f"{details}\n\n{result.diff}" if details else result.diff
        else:
            card.set_details(details, kind="output")

        if self._current_session_id:
            self.sessions.add_message(
                self._current_session_id,
                Role.TOOL,
                details[:20000],
                {"tool": call.name, "status": "success" if result.ok else "error"},
            )
        if result.diff:
            path = str(result.data.get("path", call.args.get("path", "")))
            self.editor.show_diff(path, result.diff)
        self.chat.scroll_to_bottom()

    @Slot(str, str)
    def _on_file_changed(self, path: str, diff: str) -> None:
        self.tree.refresh()
        self.editor.reload_file(path)
        if diff:
            self.editor.show_diff(path, diff)

    @Slot(dict)
    def _on_usage_updated(self, payload: dict) -> None:
        self._status_tokens.setText(f"tokens: {payload.get('total_tokens', 0):,}")
        self.refresh_usage()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.chat.end_assistant_stream()
        self.chat.add_message("error", message)

    @Slot(object)
    def _on_run_finished(self, result: Optional[AgentRunResult]) -> None:
        self._set_running(False)
        self.chat.end_assistant_stream()
        self.tree.refresh()
        self.refresh_usage()
        if result is None:
            return
        if result.final_message:
            kind = "assistant" if result.ok else "error"
            self.chat.add_message(kind, result.final_message)
            if self._current_session_id:
                self.sessions.add_message(
                    self._current_session_id, Role.ASSISTANT, result.final_message
                )
        self.chat.add_notice(
            f"{result.stop_reason.value} · {result.iterations} steps · "
            f"{result.tool_calls} tool calls · {result.total_tokens:,} tokens · "
            f"{result.duration_ms / 1000:.1f}s",
            "success" if result.ok else "error",
        )
        self._status_text.setText(f"Finished: {result.stop_reason.value}")
        if (
            result.stop_reason is AgentStopReason.FINISHED
            and self.config.github.auto_push
            and self.git is not None
            and self.git.is_repo()
        ):
            self._git_push()

    @staticmethod
    def _describe_call(call: ToolCall) -> str:
        args = call.args
        if call.name == "run_command":
            return f"$ {args.get('command', '')}"
        for key in ("path", "source", "query", "action", "summary"):
            if key in args:
                return f"{key}: {str(args[key])[:160]}"
        return ""

    # =============================================================== usage
    def refresh_usage(self) -> None:
        """Pull a fresh snapshot and update the sidebar / banner / status."""
        try:
            snapshot = self.usage.snapshot()
        except Exception as exc:  # pragma: no cover - telemetry must never crash the UI
            logger.debug("usage snapshot failed: %s", exc)
            return
        self.usage_panel.update_snapshot(snapshot)
        day = snapshot.day_totals
        self._status_tokens.setText(f"tokens today: {day.total_tokens:,}")
        policy = snapshot.policy
        if policy.max_requests_per_day:
            self._status_quota.setText(
                f"requests: {day.requests}/{policy.max_requests_per_day}"
            )
        else:
            self._status_quota.setText(f"requests: {day.requests}")

        if snapshot.blocked:
            self._show_banner(snapshot.block_reason, blocking=True)
        elif snapshot.warning:
            self._show_banner(snapshot.warning, blocking=False)
        else:
            self.quota_banner.setVisible(False)

    def _show_banner(self, text: str, *, blocking: bool) -> None:
        color = COLORS.danger if blocking else COLORS.warning
        background = "#2a1517" if blocking else "#2a2113"
        self.quota_banner.setStyleSheet(
            f"background-color: {background}; border: 1px solid {color}; border-radius: 6px;"
        )
        self._banner_label.setStyleSheet(f"color: {color};")
        self._banner_label.setText(text)
        self.quota_banner.setVisible(bool(text))
        self._run_button.setEnabled(not blocking and not (self.worker and self.worker.isRunning()))

    def open_usage_editor(self) -> None:
        """Launch ``UsageLimitEditor`` as a separate process."""
        try:
            if is_frozen():
                candidates = [
                    Path(sys.executable).parent / "UsageLimitEditor.exe",
                    Path(sys.executable).parent / "UsageLimitEditor",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        subprocess.Popen([str(candidate)])
                        return
                QMessageBox.warning(
                    self,
                    "Usage Limit Editor",
                    "UsageLimitEditor executable was not found next to Agent3.",
                )
                return
            subprocess.Popen(
                [sys.executable, "-m", "usage_limit_editor"],
                cwd=str(Path(__file__).resolve().parents[3]),
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
            )
        except OSError as exc:
            QMessageBox.critical(self, "Usage Limit Editor", str(exc))

    # ================================================================= git
    def _require_git(self) -> bool:
        if self.git is None:
            QMessageBox.information(self, "Git", "Open a workspace folder first.")
            return False
        if not GitRepo.git_available():
            QMessageBox.warning(self, "Git", "git was not found in PATH.")
            return False
        return True

    def _git_init(self) -> None:
        if not self._require_git():
            return
        result = self.git.init(default_branch=self.config.github.default_branch)
        self.terminal.append(f"\n$ git init\n{result.output}\n", COLORS.text_dim)
        self.chat.add_notice(result.output or "repository initialised", "success")
        self.tree.refresh()

    def _git_status(self) -> None:
        if not self._require_git():
            return
        self.terminal.append(f"\n$ git status\n{self.git.status_text()}\n", COLORS.text_dim)

    def _git_commit(self) -> None:
        if not self._require_git():
            return
        message, ok = QInputDialog.getText(
            self, "Commit", "Commit message:", text=f"{self.config.github.commit_prefix} "
        )
        if not ok or not message.strip():
            return
        result = self.git.commit(message.strip(), stage_all=self.config.github.auto_stage)
        self.terminal.append(f"\n$ git commit\n{result.output}\n", COLORS.text_dim)
        self.chat.add_notice(result.output[:500], "success" if result.ok else "error")

    def _git_push(self) -> None:
        if not self._require_git():
            return
        result = self.git.push(self.config.github.default_remote, set_upstream=True)
        self.terminal.append(f"\n$ git push\n{result.output}\n", COLORS.text_dim)
        self.chat.add_notice(result.output[:500] or "pushed", "success" if result.ok else "error")

    # ============================================================== misc
    def open_settings(self) -> None:
        dialog = SettingsDialog(self.config_manager, self)
        if dialog.exec():
            self.config = self.config_manager.config
            self.client.update_settings(self.config.ollama)
            self.composer_bar.set_model(self.config.ollama.model)
            self.composer_bar.set_think_spec(self.config.ollama.think)
            if self.runner is not None:
                self.runner.default_timeout = self.config.agent.command_timeout
                self.runner.set_blocked_patterns(self.config.agent.blocked_command_patterns)
            if self.tool_context is not None:
                self.tool_context.command_timeout = self.config.agent.command_timeout
                self.tool_context.max_output_chars = self.config.agent.max_output_chars
                self.tool_context.auto_syntax_check = self.config.agent.auto_syntax_check
                self.tool_context.run_external_linters = self.config.agent.run_external_linters
            if self.processes is not None:
                self.processes.max_processes = self.config.agent.max_background_processes
            if self.loop is not None:
                self.loop.settings = self.config.agent
            self.check_connection()

    def _on_model_changed(self, model: str) -> None:
        if not model.strip():
            return
        self.config.ollama.model = model.strip()
        self.config_manager.save()
        if self.loop is not None:
            self.loop.model = model.strip()
        if self._current_session_id:
            self.sessions.set_model(self._current_session_id, model.strip())
        # Reasoning levels are per model: re-ask the server what this one takes.
        self.refresh_model_capabilities(model.strip())

    def _open_logs(self) -> None:
        folder = app_paths().log_dir
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:  # pragma: no cover
            QMessageBox.information(self, "Logs", f"Log folder: {folder}\n({exc})")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About Agent3",
            f"<h3>{APP_NAME} {__version__}</h3>"
            "<p>Local autonomous AI coding agent and workspace.</p>"
            f"<p>Inference endpoint: <code>{self.config.ollama.base_url}</code><br>"
            f"Model: <code>{self.config.ollama.model}</code><br>"
            f"Data folder: <code>{app_paths().base}</code></p>"
            "<p>Everything runs on this machine - no cloud calls.</p>",
        )

    # --------------------------------------------------------- lifecycle
    def _restore_window_state(self) -> None:
        geometry = self.config.ui.window_geometry
        state = self.config.ui.window_state
        try:
            if geometry:
                self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
            if state:
                self.restoreState(QByteArray.fromBase64(state.encode("ascii")))
        except Exception:  # pragma: no cover - corrupt saved state
            logger.debug("could not restore window state", exc_info=True)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self.editor.has_unsaved():
            answer = QMessageBox.question(
                self,
                "Unsaved changes",
                "Save modified files before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save:
                self.editor.save_all()

        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(4000)
        if self._health_worker is not None and self._health_worker.isRunning():
            self._health_worker.requestInterruption()
            self._health_worker.wait(3000)
        if self.terminal.is_busy():
            self.terminal.stop()

        try:
            self.config.ui.window_geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
            self.config.ui.window_state = bytes(self.saveState().toBase64()).decode("ascii")
            self.config_manager.save()
        except Exception:  # pragma: no cover
            logger.debug("could not persist window state", exc_info=True)

        self._usage_timer.stop()
        self.client.close()
        self.usage.close()
        self.sessions.close()
        event.accept()
