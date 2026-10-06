"""Sidebar panel rendering the agent's live plan.

The model maintains this list with the ``manage_tasks`` tool. Showing it in
the sidebar turns a long autonomous run from a wall of scrolling text into
something the user can supervise at a glance: what is done, what is being
worked on right now, what is still queued.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from agent3.agent.tasks import Task, TaskList
from agent3.ui.theme import COLORS

_GLYPH = {
    "pending": "○",
    "in_progress": "◐",
    "completed": "✓",
    "cancelled": "✕",
}

_COLOR = {
    "pending": COLORS.text_dim,
    "in_progress": COLORS.accent,
    "completed": COLORS.success,
    "cancelled": COLORS.text_faint,
}


class TaskPanel(QWidget):
    """``PLAN`` section of the sidebar."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("SectionHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 8, 12, 8)
        header_layout.setSpacing(6)
        title = QLabel("PLAN")
        title.setObjectName("SectionTitle")
        self._counter = QLabel("")
        self._counter.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._counter.setStyleSheet(f"color: {COLORS.text_faint}; font-size: 11px;")
        header_layout.addWidget(title, 1)
        header_layout.addWidget(self._counter, 0)
        layout.addWidget(header)

        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(4)
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setStyleSheet(
            f"""
            QProgressBar {{ background-color: {COLORS.bg_alt}; border: none; }}
            QProgressBar::chunk {{ background-color: {COLORS.accent}; }}
            """
        )
        layout.addWidget(self._bar)

        self.list = QListWidget()
        self.list.setObjectName("TaskList")
        self.list.setWordWrap(True)
        self.list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.list.setStyleSheet(
            f"""
            QListWidget#TaskList {{
                background-color: {COLORS.panel};
                border: none;
                padding: 2px 4px;
            }}
            QListWidget#TaskList::item {{ padding: 3px 4px; border: none; }}
            """
        )
        layout.addWidget(self.list, 1)

        self._empty = QLabel("  The agent has not written a plan yet.")
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet(f"color: {COLORS.text_faint}; padding: 8px 10px;")
        layout.addWidget(self._empty)

        self.set_tasks([])

    # ------------------------------------------------------------- render
    def set_tasks(self, tasks: Sequence[Task] | TaskList | Iterable[Task]) -> None:
        """Redraw the panel from a :class:`TaskList` or a list of tasks."""
        items: List[Task] = list(tasks) if tasks is not None else []
        self.list.clear()

        if not items:
            self._counter.setText("")
            self._bar.setValue(0)
            self._empty.setVisible(True)
            self.list.setVisible(False)
            return

        self._empty.setVisible(False)
        self.list.setVisible(True)
        done = 0
        for task in items:
            status = getattr(task, "status", "pending")
            if status in ("completed", "cancelled"):
                done += 1
            entry = QListWidgetItem(f"{_GLYPH.get(status, '○')}  {task.description}")
            entry.setForeground(Qt.GlobalColor.white)
            font = QFont()
            if status == "in_progress":
                font.setBold(True)
            elif status in ("completed", "cancelled"):
                font.setStrikeOut(status == "completed")
            entry.setFont(font)
            entry.setToolTip(f"{task.id} - {status.replace('_', ' ')}")
            entry.setData(Qt.ItemDataRole.UserRole, task.id)
            self.list.addItem(entry)
            widget_color = _COLOR.get(status, COLORS.text)
            entry.setForeground(_qt_color(widget_color))

        total = len(items)
        self._counter.setText(f"{done}/{total}")
        self._bar.setValue(int(round(done / total * 100)) if total else 0)

    def clear(self) -> None:
        self.set_tasks([])

    def task_ids(self) -> List[str]:
        return [
            str(self.list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.list.count())
        ]


def _qt_color(value: str):
    from PySide6.QtGui import QColor

    return QColor(value)
