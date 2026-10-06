"""Workspace file explorer (lazy loading tree with a context menu)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from agent3.core.logging_setup import get_logger
from agent3.workspace.fs import WorkspaceFS

logger = get_logger(__name__)

PATH_ROLE = Qt.ItemDataRole.UserRole + 1
LOADED_ROLE = Qt.ItemDataRole.UserRole + 2

FILE_GLYPHS = {
    ".py": "PY", ".js": "JS", ".ts": "TS", ".tsx": "TS", ".jsx": "JS",
    ".json": "{}", ".md": "MD", ".yml": "YML", ".yaml": "YML", ".toml": "TOML",
    ".html": "<>", ".css": "CSS", ".sh": "SH", ".txt": "TXT", ".sql": "SQL",
}


class WorkspaceTree(QTreeWidget):
    """Directory tree bound to a :class:`WorkspaceFS` sandbox."""

    file_opened = Signal(str)          # workspace-relative path
    file_system_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fs: Optional[WorkspaceFS] = None
        self.setHeaderLabels(["Explorer"])
        self.setColumnCount(1)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setExpandsOnDoubleClick(False)
        self.setUniformRowHeights(True)
        self.setAnimated(True)
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(True)

        self.itemExpanded.connect(self._on_expanded)
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.customContextMenuRequested.connect(self._show_context_menu)

    # ------------------------------------------------------------- public
    @property
    def fs(self) -> Optional[WorkspaceFS]:
        return self._fs

    def set_workspace(self, fs: Optional[WorkspaceFS]) -> None:
        """Mount a new workspace (or clear the view when ``None``)."""
        self._fs = fs
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the tree, preserving expansion state where possible."""
        expanded = self._expanded_paths()
        selected = self.current_path()
        self.clear()
        if self._fs is None:
            self.setHeaderLabels(["Explorer"])
            return
        self.setHeaderLabels([self._fs.root.name or str(self._fs.root)])
        self._populate(None, "")
        self._restore_expanded(expanded)
        if selected:
            self.select_path(selected)

    def current_path(self) -> str:
        item = self.currentItem()
        return item.data(0, PATH_ROLE) if item else ""

    def select_path(self, relative: str) -> None:
        """Expand to and select *relative* if it is present in the tree."""
        parts = [p for p in relative.replace("\\", "/").split("/") if p]
        parent: Optional[QTreeWidgetItem] = None
        for depth, part in enumerate(parts):
            container = self.invisibleRootItem() if parent is None else parent
            match = None
            for index in range(container.childCount()):
                child = container.child(index)
                if child.text(0) == part or str(child.data(0, PATH_ROLE) or "").endswith(part):
                    match = child
                    break
            if match is None:
                return
            if depth < len(parts) - 1:
                match.setExpanded(True)
                parent = match
            else:
                self.setCurrentItem(match)

    # ------------------------------------------------------------ private
    def _populate(self, parent: Optional[QTreeWidgetItem], relative: str) -> None:
        if self._fs is None:
            return
        try:
            entries = self._fs.list_dir(relative)
        except (OSError, PermissionError) as exc:
            logger.debug("cannot list %s: %s", relative, exc)
            return
        for entry in entries:
            item = QTreeWidgetItem([entry.name])
            item.setData(0, PATH_ROLE, entry.path)
            item.setData(0, LOADED_ROLE, False)
            if entry.is_dir:
                item.setIcon(0, QIcon())
                item.setToolTip(0, entry.path)
                item.addChild(QTreeWidgetItem(["loading..."]))
            else:
                suffix = Path(entry.name).suffix.lower()
                item.setToolTip(0, f"{entry.path}  ({entry.size:,} bytes)")
                item.setText(0, entry.name)
                if suffix in FILE_GLYPHS:
                    item.setStatusTip(0, FILE_GLYPHS[suffix])
            if parent is None:
                self.addTopLevelItem(item)
            else:
                parent.addChild(item)

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        if item.data(0, LOADED_ROLE):
            return
        item.takeChildren()
        item.setData(0, LOADED_ROLE, True)
        self._populate(item, item.data(0, PATH_ROLE) or "")

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        path = item.data(0, PATH_ROLE)
        if not path or self._fs is None:
            return
        if self._fs.is_dir(path):
            item.setExpanded(not item.isExpanded())
        else:
            self.file_opened.emit(path)

    def _expanded_paths(self) -> set[str]:
        found: set[str] = set()

        def walk(item: QTreeWidgetItem) -> None:
            for index in range(item.childCount()):
                child = item.child(index)
                if child.isExpanded():
                    path = child.data(0, PATH_ROLE)
                    if path:
                        found.add(path)
                    walk(child)

        walk(self.invisibleRootItem())
        return found

    def _restore_expanded(self, paths: set[str]) -> None:
        if not paths:
            return

        def walk(item: QTreeWidgetItem) -> None:
            for index in range(item.childCount()):
                child = item.child(index)
                path = child.data(0, PATH_ROLE)
                if path in paths:
                    child.setExpanded(True)
                    walk(child)

        walk(self.invisibleRootItem())

    # ------------------------------------------------------- context menu
    def _show_context_menu(self, position) -> None:
        if self._fs is None:
            return
        item = self.itemAt(position)
        target = item.data(0, PATH_ROLE) if item else ""
        is_dir = self._fs.is_dir(target) if target else True
        directory = target if is_dir else str(Path(target).parent) if target else ""
        if directory == ".":
            directory = ""

        menu = QMenu(self)
        if item and not is_dir:
            open_action = QAction("Open", self)
            open_action.triggered.connect(lambda: self.file_opened.emit(target))
            menu.addAction(open_action)
            menu.addSeparator()

        new_file = QAction("New file...", self)
        new_file.triggered.connect(lambda: self._create_file(directory))
        menu.addAction(new_file)

        new_folder = QAction("New folder...", self)
        new_folder.triggered.connect(lambda: self._create_folder(directory))
        menu.addAction(new_folder)

        if item:
            menu.addSeparator()
            rename = QAction("Rename...", self)
            rename.triggered.connect(lambda: self._rename(target))
            menu.addAction(rename)

            delete = QAction("Delete", self)
            delete.triggered.connect(lambda: self._delete(target))
            menu.addAction(delete)

        menu.addSeparator()
        refresh = QAction("Refresh", self)
        refresh.triggered.connect(self.refresh)
        menu.addAction(refresh)
        menu.exec(self.viewport().mapToGlobal(position))

    def _create_file(self, directory: str) -> None:
        name, ok = QInputDialog.getText(self, "New file", "File name:")
        if not ok or not name.strip() or self._fs is None:
            return
        relative = f"{directory}/{name}".strip("/") if directory else name
        try:
            if self._fs.exists(relative):
                QMessageBox.warning(self, "New file", f"{relative} already exists.")
                return
            self._fs.write_text(relative, "")
        except (OSError, PermissionError) as exc:
            QMessageBox.critical(self, "New file", str(exc))
            return
        self.refresh()
        self.file_system_changed.emit()
        self.file_opened.emit(relative)

    def _create_folder(self, directory: str) -> None:
        name, ok = QInputDialog.getText(self, "New folder", "Folder name:")
        if not ok or not name.strip() or self._fs is None:
            return
        relative = f"{directory}/{name}".strip("/") if directory else name
        try:
            self._fs.mkdir(relative)
        except (OSError, PermissionError) as exc:
            QMessageBox.critical(self, "New folder", str(exc))
            return
        self.refresh()
        self.file_system_changed.emit()

    def _rename(self, target: str) -> None:
        if self._fs is None or not target:
            return
        current = Path(target).name
        name, ok = QInputDialog.getText(self, "Rename", "New name:", text=current)
        if not ok or not name.strip() or name == current:
            return
        parent = str(Path(target).parent)
        destination = f"{parent}/{name}".strip("/") if parent not in (".", "") else name
        try:
            self._fs.rename(target, destination)
        except (OSError, PermissionError, ValueError) as exc:
            QMessageBox.critical(self, "Rename", str(exc))
            return
        self.refresh()
        self.file_system_changed.emit()

    def _delete(self, target: str) -> None:
        if self._fs is None or not target:
            return
        confirm = QMessageBox.question(
            self,
            "Delete",
            f"Delete '{target}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._fs.delete(target, recursive=True)
        except (OSError, PermissionError, ValueError) as exc:
            QMessageBox.critical(self, "Delete", str(exc))
            return
        self.refresh()
        self.file_system_changed.emit()
