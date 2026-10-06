"""Embedded terminal console: live agent output + manual command entry."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from agent3.ui.theme import COLORS, mono_font
from agent3.ui.workers import CommandWorker
from agent3.workspace.terminal import CommandResult, CommandRunner

MAX_BLOCKS = 5000


class _CommandLine(QLineEdit):
    """Input line with bash-like history navigation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._history: List[str] = []
        self._index = 0
        self.setPlaceholderText("Type a command and press Enter (e.g. pytest -q)")

    def push_history(self, command: str) -> None:
        if command and (not self._history or self._history[-1] != command):
            self._history.append(command)
        self._index = len(self._history)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt API)
        if event.key() == Qt.Key.Key_Up and self._history:
            self._index = max(0, self._index - 1)
            self.setText(self._history[self._index])
            return
        if event.key() == Qt.Key.Key_Down and self._history:
            self._index = min(len(self._history), self._index + 1)
            self.setText("" if self._index >= len(self._history) else self._history[self._index])
            return
        super().keyPressEvent(event)


class TerminalPanel(QWidget):
    """Bottom dock console wired to a :class:`CommandRunner`."""

    command_finished = Signal(object)  # CommandResult

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._runner: Optional[CommandRunner] = None
        self._worker: Optional[CommandWorker] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame {{ background-color: {COLORS.bg_alt};"
            f" border-bottom: 1px solid {COLORS.border_soft}; }}"
            f"QFrame QLabel {{ background: transparent; }}"
        )
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(14, 6, 10, 6)
        bar_layout.setSpacing(9)

        title = QLabel("TERMINAL")
        title.setStyleSheet(
            f"color: {COLORS.text_faint}; font-weight: 700; font-size: 8pt;"
            f" letter-spacing: 1.4px;"
        )
        bar_layout.addWidget(title, 0)

        self._cwd_label = QLabel("")
        self._cwd_label.setStyleSheet(f"color: {COLORS.text_faint};")
        bar_layout.addWidget(self._cwd_label, 1)

        #: Count of background processes the agent started and never stopped.
        self._process_badge = QLabel("")
        self._process_badge.setVisible(False)
        self._process_badge.setStyleSheet(
            f"color: {COLORS.warning}; background-color: {COLORS.panel_alt};"
            f" border: 1px solid {COLORS.border}; border-radius: 9px; padding: 1px 8px;"
        )
        bar_layout.addWidget(self._process_badge, 0)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setObjectName("Danger")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        bar_layout.addWidget(self._stop_button, 0)

        clear_button = QPushButton("Clear")
        clear_button.setObjectName("Ghost")
        clear_button.clicked.connect(self.clear)
        bar_layout.addWidget(clear_button, 0)
        layout.addWidget(bar)

        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(mono_font(10))
        self._output.setMaximumBlockCount(MAX_BLOCKS)
        self._output.setFrameShape(QFrame.Shape.NoFrame)
        self._output.setStyleSheet(
            f"QPlainTextEdit {{ background-color: {COLORS.bg}; color: {COLORS.text_dim};"
            f" border: none; padding: 8px 12px; }}"
        )
        # An empty console is a large black rectangle with nothing to say for
        # itself; the placeholder explains what will appear there.
        self._output.setPlaceholderText(
            "Command output appears here - both the agent's and your own."
        )
        layout.addWidget(self._output, 1)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(8, 6, 8, 8)
        input_row.setSpacing(8)
        prompt = QLabel("$")
        prompt.setStyleSheet(f"color: {COLORS.success}; font-weight: 700;")
        input_row.addWidget(prompt, 0)

        self._input = _CommandLine()
        self._input.setFont(mono_font(10))
        self._input.returnPressed.connect(self._on_submit)
        input_row.addWidget(self._input, 1)

        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self._on_submit)
        input_row.addWidget(self._run_button, 0)
        layout.addLayout(input_row)

    # -------------------------------------------------------------- setup
    def set_process_count(self, count: int) -> None:
        """Show ``N running`` in the title bar for background processes."""
        count = max(0, int(count))
        self._process_badge.setVisible(count > 0)
        if count:
            self._process_badge.setText(f"● {count} background")
            self._process_badge.setToolTip(
                f"{count} background process(es) started by the agent are still running"
            )

    def process_count_text(self) -> str:
        return self._process_badge.text() if self._process_badge.isVisible() else ""

    def set_runner(self, runner: Optional[CommandRunner]) -> None:
        self._runner = runner
        self._cwd_label.setText(str(runner.cwd) if runner else "no workspace mounted")
        self._input.setEnabled(runner is not None)
        self._run_button.setEnabled(runner is not None)

    # ------------------------------------------------------------- output
    def append(self, text: str, color: Optional[str] = None) -> None:
        """Append *text* to the console with an optional colour."""
        if not text:
            return
        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color or COLORS.text_dim))
        cursor.insertText(text, fmt)
        self._output.setTextCursor(cursor)
        self._output.ensureCursorVisible()

    def append_stream(self, stream: str, text: str) -> None:
        self.append(text, COLORS.danger if stream == "stderr" else COLORS.text_dim)

    def append_command(self, command: str) -> None:
        self.append(f"\n$ {command}\n", COLORS.accent)

    def append_result(self, result: CommandResult) -> None:
        color = COLORS.success if result.ok else COLORS.danger
        status = (
            "blocked"
            if result.blocked
            else "timed out"
            if result.timed_out
            else f"exit {result.exit_code}"
        )
        self.append(f"[{status} in {result.duration_ms} ms]\n", color)

    def clear(self) -> None:
        self._output.clear()

    # ----------------------------------------------------------- running
    def run_command(self, command: str) -> None:
        """Execute *command* asynchronously."""
        if self._runner is None:
            self.append("No workspace mounted.\n", COLORS.danger)
            return
        if self._worker is not None and self._worker.isRunning():
            self.append("A command is already running.\n", COLORS.warning)
            return
        self.append_command(command)
        self._input.push_history(command)
        self._set_busy(True)
        worker = CommandWorker(self._runner, command, parent=self)
        worker.output.connect(self.append_stream)
        worker.completed.connect(self._on_completed)
        worker.finished.connect(lambda: self._set_busy(False))
        self._worker = worker
        worker.start()

    def stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.append("\n^C terminated\n", COLORS.warning)

    def is_busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _on_submit(self) -> None:
        command = self._input.text().strip()
        if not command:
            return
        self._input.clear()
        if command in {"clear", "cls"}:
            self.clear()
            return
        self.run_command(command)

    def _on_completed(self, result: CommandResult) -> None:
        self.append_result(result)
        self.command_finished.emit(result)

    def _set_busy(self, busy: bool) -> None:
        self._run_button.setEnabled(not busy and self._runner is not None)
        self._stop_button.setEnabled(busy)
