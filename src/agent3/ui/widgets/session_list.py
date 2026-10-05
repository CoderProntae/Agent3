"""Chat/session switcher for the left sidebar."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from agent3.agent.session import ChatSession
from agent3.ui.theme import COLORS

SESSION_ROLE = Qt.ItemDataRole.UserRole + 1


class SessionList(QWidget):
    """List of saved conversations with new/rename/delete actions."""

    session_selected = Signal(str)
    session_created = Signal()
    session_deleted = Signal(str)
    session_renamed = Signal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(10, 6, 6, 0)
        title = QLabel("SESSIONS")
        title.setObjectName("SectionTitle")
        title.setContentsMargins(0, 0, 0, 0)
        header.addWidget(title, 1)
        self._new_button = QPushButton("+ New")
        self._new_button.setObjectName("Ghost")
        self._new_button.clicked.connect(self.session_created.emit)
        header.addWidget(self._new_button, 0)
        layout.addLayout(header)

        self._list = QListWidget()
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._context_menu)
        self._list.itemClicked.connect(self._on_clicked)
        layout.addWidget(self._list, 1)

    # -------------------------------------------------------------- data
    def set_sessions(self, sessions: List[ChatSession], current_id: str = "") -> None:
        self._list.clear()
        for session in sessions:
            item = QListWidgetItem(session.title or "Untitled")
            item.setData(SESSION_ROLE, session.id)
            item.setToolTip(f"{session.title}\nupdated: {session.updated_at}\nmodel: {session.model}")
            self._list.addItem(item)
            if session.id == current_id:
                self._list.setCurrentItem(item)
                item.setForeground(Qt.GlobalColor.white)

    def current_session_id(self) -> str:
        item = self._list.currentItem()
        return item.data(SESSION_ROLE) if item else ""

    # ----------------------------------------------------------- actions
    def _on_clicked(self, item: QListWidgetItem) -> None:
        session_id = item.data(SESSION_ROLE)
        if session_id:
            self.session_selected.emit(session_id)

    def _context_menu(self, position) -> None:
        item = self._list.itemAt(position)
        menu = QMenu(self)

        new_action = QAction("New session", self)
        new_action.triggered.connect(self.session_created.emit)
        menu.addAction(new_action)

        if item is not None:
            session_id = item.data(SESSION_ROLE)
            menu.addSeparator()
            rename = QAction("Rename...", self)
            rename.triggered.connect(lambda: self._rename(session_id, item.text()))
            menu.addAction(rename)

            delete = QAction("Delete", self)
            delete.triggered.connect(lambda: self._delete(session_id, item.text()))
            menu.addAction(delete)

        menu.exec(self._list.viewport().mapToGlobal(position))

    def _rename(self, session_id: str, current: str) -> None:
        title, ok = QInputDialog.getText(self, "Rename session", "Title:", text=current)
        if ok and title.strip():
            self.session_renamed.emit(session_id, title.strip())

    def _delete(self, session_id: str, title: str) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete session",
            f"Delete session '{title}' and all of its messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.session_deleted.emit(session_id)
