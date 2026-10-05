"""Persistent chat sessions (SQLite) so conversations survive restarts."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent3.core.logging_setup import get_logger
from agent3.core.paths import app_paths
from agent3.llm.messages import ChatMessage, Role

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT 'New session',
    workspace  TEXT NOT NULL DEFAULT '',
    model      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    meta       TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ChatSession:
    """Metadata of one conversation."""

    id: str
    title: str = "New session"
    workspace: str = ""
    model: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "workspace": self.workspace,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived": self.archived,
        }


class SessionStore:
    """CRUD for sessions and their messages."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else app_paths().sessions_db
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=15.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass

    # ------------------------------------------------------------ sessions
    def create(self, *, title: str = "New session", workspace: str = "", model: str = "") -> ChatSession:
        session = ChatSession(id=uuid.uuid4().hex[:16], title=title, workspace=workspace, model=model)
        with self._lock:
            self._conn.execute(
                """INSERT INTO sessions(id, title, workspace, model, created_at, updated_at, archived)
                   VALUES(?,?,?,?,?,?,0)""",
                (
                    session.id,
                    session.title,
                    session.workspace,
                    session.model,
                    session.created_at,
                    session.updated_at,
                ),
            )
            self._conn.commit()
        return session

    def get(self, session_id: str) -> Optional[ChatSession]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def list(self, *, include_archived: bool = False, limit: int = 100) -> List[ChatSession]:
        sql = "SELECT * FROM sessions"
        if not include_archived:
            sql += " WHERE archived = 0"
        sql += " ORDER BY datetime(updated_at) DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (int(limit),)).fetchall()
        return [self._row_to_session(row) for row in rows]

    def rename(self, session_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title[:120], _now(), session_id),
            )
            self._conn.commit()

    def set_model(self, session_id: str, model: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET model = ?, updated_at = ? WHERE id = ?", (model, _now(), session_id)
            )
            self._conn.commit()

    def archive(self, session_id: str, archived: bool = True) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET archived = ?, updated_at = ? WHERE id = ?",
                (1 if archived else 0, _now(), session_id),
            )
            self._conn.commit()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()

    # ------------------------------------------------------------ messages
    def add_message(
        self, session_id: str, role: Role | str, content: str, meta: Optional[Dict[str, Any]] = None
    ) -> int:
        role_value = role.value if isinstance(role, Role) else str(role)
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO messages(session_id, role, content, meta, created_at) VALUES(?,?,?,?,?)",
                (session_id, role_value, content, json.dumps(meta or {}, ensure_ascii=False), _now()),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id)
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def messages(self, session_id: str, *, limit: int = 1000) -> List[ChatMessage]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, meta FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, int(limit)),
            ).fetchall()
        result: List[ChatMessage] = []
        for row in rows:
            try:
                role = Role(row["role"])
            except ValueError:  # pragma: no cover
                role = Role.USER
            try:
                meta = json.loads(row["meta"] or "{}")
            except json.JSONDecodeError:  # pragma: no cover
                meta = {}
            result.append(ChatMessage(role=role, content=row["content"], metadata=meta))
        return result

    def raw_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def clear_messages(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.commit()

    def autoname(self, session_id: str, first_user_message: str) -> str:
        """Derive a readable title from the first instruction."""
        title = " ".join(first_user_message.strip().split())[:60] or "New session"
        self.rename(session_id, title)
        return title

    def export_markdown(self, session_id: str) -> str:
        """Render a whole conversation as markdown (used by 'Export chat')."""
        session = self.get(session_id)
        lines = [f"# {session.title if session else session_id}", ""]
        for message in self.messages(session_id):
            lines.append(f"## {message.role.value}")
            lines.append(message.content)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            id=row["id"],
            title=row["title"],
            workspace=row["workspace"],
            model=row["model"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            archived=bool(row["archived"]),
        )
