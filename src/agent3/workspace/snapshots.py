"""In-memory undo history for every agent file mutation.

When ``edit_file`` or ``patch_file`` lands badly the model tends to make things
worse while trying to repair the damage by hand. :class:`SnapshotStore` records
the *previous* content of a file immediately before every mutating tool runs,
so a single ``undo_file_change`` call restores the last known good state.

Design notes
------------
* Memory is bounded twice - by entry count and by total bytes - because a
  long run touching large files must never grow without limit.
* A file that did not exist is recorded with ``content=None``; undoing such a
  snapshot deletes the file again, which is what "restore the previous state"
  means for a freshly created file.
* The store is deliberately in memory only: it is an undo buffer, not a backup
  system, and writing shadow copies into the user's workspace would pollute
  their git status.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent3.core.logging_setup import get_logger

logger = get_logger(__name__)

#: How many snapshots are kept before the oldest is dropped.
DEFAULT_MAX_ENTRIES = 80

#: Total payload budget; large files are still recorded, old ones evicted.
DEFAULT_MAX_BYTES = 16 * 1024 * 1024


@dataclass
class Snapshot:
    """The state of one file immediately before a mutation."""

    path: str
    #: ``None`` means "the file did not exist".
    content: Optional[str]
    tool: str
    at: float = field(default_factory=time.time)

    @property
    def existed(self) -> bool:
        return self.content is not None

    @property
    def size(self) -> int:
        return len(self.content or "")

    def describe(self) -> str:
        age = max(0, int(time.time() - self.at))
        state = f"{len(self.content.splitlines())} lines" if self.content is not None else "absent"
        return f"{self.path} ({state}, before {self.tool}, {age}s ago)"

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "existed": self.existed,
            "tool": self.tool,
            "bytes": self.size,
            "age_seconds": round(time.time() - self.at, 1),
        }


class SnapshotStore:
    """A bounded LIFO stack of :class:`Snapshot` objects."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._stack: List[Snapshot] = []

    # ------------------------------------------------------------ record
    def record(self, path: str, content: Optional[str], tool: str) -> Snapshot:
        """Push the pre-mutation state of *path*."""
        snapshot = Snapshot(path=path, content=content, tool=tool)
        self._stack.append(snapshot)
        self._evict()
        return snapshot

    def _evict(self) -> None:
        while len(self._stack) > self.max_entries:
            self._stack.pop(0)
        total = sum(entry.size for entry in self._stack)
        while total > self.max_bytes and len(self._stack) > 1:
            dropped = self._stack.pop(0)
            total -= dropped.size

    # ------------------------------------------------------------- query
    def __len__(self) -> int:
        return len(self._stack)

    @property
    def entries(self) -> List[Snapshot]:
        return list(self._stack)

    def peek(self, path: Optional[str] = None) -> Optional[Snapshot]:
        """Newest snapshot overall, or the newest one for *path*."""
        for entry in reversed(self._stack):
            if path is None or entry.path == path:
                return entry
        return None

    def history(self, path: Optional[str] = None, limit: int = 20) -> List[Snapshot]:
        rows = [e for e in reversed(self._stack) if path is None or e.path == path]
        return rows[:limit]

    def paths(self) -> List[str]:
        seen: List[str] = []
        for entry in reversed(self._stack):
            if entry.path not in seen:
                seen.append(entry.path)
        return seen

    # ---------------------------------------------------------------- pop
    def pop(self, path: Optional[str] = None) -> Optional[Snapshot]:
        """Remove and return the snapshot that ``undo`` should restore."""
        for index in range(len(self._stack) - 1, -1, -1):
            if path is None or self._stack[index].path == path:
                return self._stack.pop(index)
        return None

    def clear(self) -> None:
        self._stack.clear()
