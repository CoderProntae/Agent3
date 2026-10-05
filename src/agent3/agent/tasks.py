"""The agent's visible plan - a to-do list it maintains while it works.

A long autonomous run drifts: the model forgets step four, repeats step two,
or declares success with half the request unfinished. :class:`TaskList` is the
cure. The model writes its plan down with ``manage_tasks``, flips items to
``in_progress`` / ``completed`` as it goes, and the UI renders the same list
live in the sidebar so the user can watch progress instead of guessing.

The structure is intentionally tiny - id, description, status - because a plan
the model has to fight with is a plan it stops updating.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

#: Allowed lifecycle states, in display order.
STATUSES = ("pending", "in_progress", "completed", "cancelled")

_STATUS_ALIASES = {
    "todo": "pending",
    "open": "pending",
    "not_started": "pending",
    "doing": "in_progress",
    "active": "in_progress",
    "in-progress": "in_progress",
    "inprogress": "in_progress",
    "wip": "in_progress",
    "done": "completed",
    "complete": "completed",
    "finished": "completed",
    "cancel": "cancelled",
    "canceled": "cancelled",
    "skipped": "cancelled",
}

_GLYPHS = {
    "pending": "○",
    "in_progress": "◐",
    "completed": "✓",
    "cancelled": "✕",
}

_CHECKBOX = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
    "cancelled": "[-]",
}


class TaskError(ValueError):
    """Raised for an unknown task id or an invalid status."""


def normalise_status(value: str) -> str:
    """Map a loose status string onto one of :data:`STATUSES`."""
    key = str(value or "").strip().lower().replace(" ", "_")
    key = _STATUS_ALIASES.get(key, key)
    if key not in STATUSES:
        raise TaskError(f"unknown status {value!r} - use one of: {', '.join(STATUSES)}")
    return key


def slugify(text: str, fallback: str = "task") -> str:
    """Derive a stable, readable id from a description."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    slug = "-".join(slug.split("-")[:4])
    return slug or fallback


@dataclass
class Task:
    """One planned step."""

    id: str
    description: str
    status: str = "pending"
    note: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.status = normalise_status(self.status)
        self.description = str(self.description).strip()
        self.id = str(self.id).strip() or slugify(self.description)

    @property
    def done(self) -> bool:
        return self.status in ("completed", "cancelled")

    @property
    def glyph(self) -> str:
        return _GLYPHS[self.status]

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "id": self.id,
            "description": self.description,
            "status": self.status,
        }
        if self.note:
            data["note"] = self.note
        return data

    def to_line(self) -> str:
        suffix = f" - {self.note}" if self.note else ""
        return f"{_CHECKBOX[self.status]} {self.id}: {self.description}{suffix}"


class TaskList:
    """An ordered, id-addressed collection of :class:`Task` objects."""

    def __init__(self, tasks: Optional[Iterable[Task]] = None) -> None:
        self._tasks: List[Task] = list(tasks or [])
        self.revision = 0

    # ------------------------------------------------------------- basics
    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)

    def __bool__(self) -> bool:
        return bool(self._tasks)

    @property
    def tasks(self) -> List[Task]:
        return list(self._tasks)

    def get(self, task_id: str) -> Optional[Task]:
        wanted = str(task_id).strip()
        for task in self._tasks:
            if task.id == wanted:
                return task
        return None

    def require(self, task_id: str) -> Task:
        task = self.get(task_id)
        if task is None:
            known = ", ".join(t.id for t in self._tasks) or "none"
            raise TaskError(f"unknown task id {task_id!r} (known ids: {known})")
        return task

    # -------------------------------------------------------------- edits
    def _unique_id(self, candidate: str) -> str:
        base = candidate or "task"
        if self.get(base) is None:
            return base
        index = 2
        while self.get(f"{base}-{index}") is not None:
            index += 1
        return f"{base}-{index}"

    def add(self, description: str, *, task_id: str = "", status: str = "pending", note: str = "") -> Task:
        text = str(description or "").strip()
        if not text:
            raise TaskError("a task needs a description")
        task = Task(
            id=self._unique_id(str(task_id).strip() or slugify(text, f"task-{len(self._tasks) + 1}")),
            description=text,
            status=status,
            note=note,
        )
        self._tasks.append(task)
        self.revision += 1
        return task

    def update(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        description: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Task:
        task = self.require(task_id)
        if status is not None:
            task.status = normalise_status(status)
        if description:
            task.description = str(description).strip()
        if note is not None:
            task.note = str(note).strip()
        task.updated_at = time.time()
        self.revision += 1
        return task

    def upsert(self, payload: Dict[str, object]) -> Task:
        """Add or update from a raw ``{"id":..., "description":..., "status":...}``."""
        task_id = str(payload.get("id") or "").strip()
        description = str(payload.get("description") or payload.get("task") or "").strip()
        status = payload.get("status")
        note = payload.get("note")
        existing = self.get(task_id) if task_id else None
        if existing is not None:
            return self.update(
                existing.id,
                status=str(status) if status is not None else None,
                description=description or None,
                note=str(note) if note is not None else None,
            )
        return self.add(
            description or task_id,
            task_id=task_id,
            status=str(status) if status is not None else "pending",
            note=str(note or ""),
        )

    def replace(self, payloads: Sequence[Dict[str, object]]) -> List[Task]:
        """Replace the whole plan (``action="set"``)."""
        self._tasks = []
        created = [self.upsert(dict(payload)) for payload in payloads]
        self.revision += 1
        return created

    def remove(self, task_id: str) -> Task:
        task = self.require(task_id)
        self._tasks.remove(task)
        self.revision += 1
        return task

    def clear(self) -> None:
        self._tasks = []
        self.revision += 1

    # ------------------------------------------------------------ queries
    def counts(self) -> Dict[str, int]:
        counts = {status: 0 for status in STATUSES}
        for task in self._tasks:
            counts[task.status] += 1
        return counts

    @property
    def current(self) -> Optional[Task]:
        """The task the agent says it is working on right now."""
        for task in self._tasks:
            if task.status == "in_progress":
                return task
        for task in self._tasks:
            if task.status == "pending":
                return task
        return None

    @property
    def open_tasks(self) -> List[Task]:
        return [task for task in self._tasks if not task.done]

    @property
    def all_done(self) -> bool:
        return bool(self._tasks) and not self.open_tasks

    def progress(self) -> tuple[int, int]:
        """``(completed_or_cancelled, total)``."""
        return sum(1 for task in self._tasks if task.done), len(self._tasks)

    # ------------------------------------------------------------ render
    def to_list(self) -> List[Dict[str, object]]:
        return [task.to_dict() for task in self._tasks]

    def render(self) -> str:
        """Plain-text plan for the model's context and for tool output."""
        if not self._tasks:
            return "(the task list is empty)"
        done, total = self.progress()
        header = f"Plan - {done}/{total} done"
        return "\n".join([header] + [task.to_line() for task in self._tasks])

    def render_markdown(self) -> str:
        if not self._tasks:
            return "_No tasks yet._"
        lines = []
        for task in self._tasks:
            text = task.description
            if task.status == "completed":
                text = f"~~{text}~~"
            elif task.status == "in_progress":
                text = f"**{text}**"
            suffix = f" — _{task.note}_" if task.note else ""
            lines.append(f"- {_CHECKBOX[task.status]} {text}{suffix}")
        return "\n".join(lines)

    def summary_line(self) -> str:
        """Compact status used in the window title / status bar."""
        if not self._tasks:
            return ""
        done, total = self.progress()
        current = self.current
        head = f"{done}/{total}"
        return f"{head} · {current.description}" if current else head
