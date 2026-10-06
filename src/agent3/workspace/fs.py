"""Filesystem access that is hard-sandboxed to the mounted workspace root.

Every path coming from the model is funnelled through :meth:`WorkspaceFS.resolve`,
which rejects absolute paths outside the root, ``..`` traversal and symlinks
that point out of the tree.  Nothing else in the codebase is allowed to touch
user files directly.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from agent3.core.logging_setup import get_logger

logger = get_logger(__name__)

#: Directories never shown in the tree and never walked during search.
DEFAULT_IGNORES: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    ".idea",
    ".vscode",
    ".tox",
    "htmlcov",
    ".DS_Store",
)

TEXT_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".md", ".rst", ".txt", ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".java", ".kt", ".go", ".rs", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".sql", ".xml", ".svg", ".env",
    ".gitignore", ".dockerignore", ".editorconfig", ".lock", ".properties", ".gradle",
}

BINARY_SNIFF_BYTES = 8192


class WorkspaceSecurityError(PermissionError):
    """Raised when a path would escape the workspace sandbox."""


class WorkspaceError(RuntimeError):
    """Generic, user-presentable filesystem failure."""


@dataclass
class FileEntry:
    """One node of the workspace tree."""

    path: str          # POSIX-style, relative to the workspace root
    name: str
    is_dir: bool
    size: int = 0
    modified: str = ""
    children: List["FileEntry"] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "name": self.name,
            "is_dir": self.is_dir,
            "size": self.size,
            "modified": self.modified,
            "children": [c.to_dict() for c in self.children],
        }


class WorkspaceFS:
    """Sandboxed file operations rooted at a single directory."""

    def __init__(
        self,
        root: Path | str,
        *,
        ignores: Sequence[str] = DEFAULT_IGNORES,
        max_read_bytes: int = 1_000_000,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        if not self._root.exists():
            self._root.mkdir(parents=True, exist_ok=True)
        if not self._root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self._root}")
        self._ignores = tuple(ignores)
        self.max_read_bytes = int(max_read_bytes)

    # ---------------------------------------------------------------- paths
    @property
    def root(self) -> Path:
        return self._root

    @property
    def ignores(self) -> tuple[str, ...]:
        return self._ignores

    def resolve(self, relative: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve *relative* inside the sandbox or raise.

        Accepts both ``src/main.py`` and an absolute path that already lives
        under the root (models love emitting absolute paths).
        """
        raw = str(relative).strip().replace("\\", "/")
        if not raw or raw in {".", "./"}:
            candidate = self._root
        else:
            path = Path(raw)
            if path.is_absolute():
                candidate = path
            else:
                candidate = self._root / path

        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:  # pragma: no cover - exotic FS errors
            raise WorkspaceSecurityError(f"cannot resolve path {relative!r}: {exc}") from exc

        if resolved != self._root and self._root not in resolved.parents:
            raise WorkspaceSecurityError(
                f"path escapes the workspace sandbox: {relative!r} -> {resolved}"
            )
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"no such file or directory: {self.relative(resolved)}")
        return resolved

    def relative(self, path: Path | str) -> str:
        """Return the POSIX path of *path* relative to the workspace root."""
        target = Path(path).resolve()
        try:
            return PurePosixPath(target.relative_to(self._root)).as_posix()
        except ValueError:
            return target.as_posix()

    def is_ignored(self, name: str) -> bool:
        return any(fnmatch.fnmatch(name, pattern) for pattern in self._ignores)

    # ----------------------------------------------------------- inspection
    def exists(self, relative: str) -> bool:
        try:
            return self.resolve(relative).exists()
        except WorkspaceSecurityError:
            return False

    def is_dir(self, relative: str) -> bool:
        try:
            return self.resolve(relative).is_dir()
        except WorkspaceSecurityError:
            return False

    def stat(self, relative: str) -> Dict[str, object]:
        target = self.resolve(relative, must_exist=True)
        info = target.stat()
        return {
            "path": self.relative(target),
            "size": info.st_size,
            "is_dir": target.is_dir(),
            "modified": datetime.fromtimestamp(info.st_mtime, tz=timezone.utc).isoformat(
                timespec="seconds"
            ),
            "binary": self.is_binary(relative) if target.is_file() else False,
        }

    def is_binary(self, relative: str) -> bool:
        """Heuristic binary sniffing (NUL byte or undecodable UTF-8)."""
        target = self.resolve(relative)
        if not target.is_file():
            return False
        if target.suffix.lower() in TEXT_EXTENSIONS:
            return False
        try:
            chunk = target.open("rb").read(BINARY_SNIFF_BYTES)
        except OSError:  # pragma: no cover
            return True
        if b"\x00" in chunk:
            return True
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError:
            return True
        return False

    def list_dir(self, relative: str = "") -> List[FileEntry]:
        """Directory listing (directories first, then files, alphabetical)."""
        target = self.resolve(relative, must_exist=True)
        if not target.is_dir():
            raise WorkspaceError(f"not a directory: {self.relative(target)}")
        entries: List[FileEntry] = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if self.is_ignored(child.name):
                continue
            try:
                info = child.stat()
                size = 0 if child.is_dir() else info.st_size
                modified = datetime.fromtimestamp(info.st_mtime, tz=timezone.utc).isoformat(
                    timespec="seconds"
                )
            except OSError:  # pragma: no cover - race with external deletion
                size, modified = 0, ""
            entries.append(
                FileEntry(
                    path=self.relative(child),
                    name=child.name,
                    is_dir=child.is_dir(),
                    size=size,
                    modified=modified,
                )
            )
        return entries

    def tree(self, relative: str = "", *, max_depth: int = 4, max_entries: int = 2000) -> FileEntry:
        """Recursive tree snapshot, depth and size bounded."""
        root_path = self.resolve(relative, must_exist=True)
        counter = {"n": 0}

        def build(path: Path, depth: int) -> FileEntry:
            entry = FileEntry(
                path=self.relative(path) or ".",
                name=path.name or self._root.name,
                is_dir=path.is_dir(),
            )
            if not path.is_dir() or depth >= max_depth:
                if path.is_file():
                    try:
                        entry.size = path.stat().st_size
                    except OSError:  # pragma: no cover
                        entry.size = 0
                return entry
            try:
                children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:  # pragma: no cover - permission denied
                return entry
            for child in children:
                if counter["n"] >= max_entries:
                    break
                if self.is_ignored(child.name):
                    continue
                counter["n"] += 1
                entry.children.append(build(child, depth + 1))
            return entry

        return build(root_path, 0)

    def tree_text(self, relative: str = "", *, max_depth: int = 3, max_entries: int = 400) -> str:
        """ASCII rendering of :meth:`tree` - injected into the system prompt."""
        root = self.tree(relative, max_depth=max_depth, max_entries=max_entries)
        lines: List[str] = [f"{root.name}/"]

        def walk(entry: FileEntry, prefix: str) -> None:
            count = len(entry.children)
            for index, child in enumerate(entry.children):
                last = index == count - 1
                connector = "`-- " if last else "|-- "
                suffix = "/" if child.is_dir else ""
                lines.append(f"{prefix}{connector}{child.name}{suffix}")
                if child.is_dir and child.children:
                    walk(child, prefix + ("    " if last else "|   "))

        walk(root, "")
        return "\n".join(lines)

    def walk_files(self, relative: str = "") -> Iterator[Path]:
        """Yield every non-ignored file below *relative*."""
        start = self.resolve(relative, must_exist=True)
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = [d for d in dirnames if not self.is_ignored(d)]
            for filename in filenames:
                if self.is_ignored(filename):
                    continue
                yield Path(dirpath) / filename

    # ----------------------------------------------------------------- read
    def read_text(self, relative: str, *, max_bytes: Optional[int] = None) -> str:
        """Read a UTF-8 text file (with graceful fallbacks)."""
        target = self.resolve(relative, must_exist=True)
        if target.is_dir():
            raise WorkspaceError(f"{self.relative(target)} is a directory, not a file")
        limit = int(max_bytes or self.max_read_bytes)
        size = target.stat().st_size
        data = target.read_bytes()[:limit]
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - latin-1 never fails
            raise WorkspaceError(f"cannot decode {self.relative(target)}")
        if size > limit:
            text += f"\n\n... [truncated: showing {limit} of {size} bytes] ..."
        return text

    def read_lines(self, relative: str, start: int = 1, end: Optional[int] = None) -> str:
        """Read a 1-based inclusive line range, prefixed with line numbers."""
        content = self.read_text(relative)
        lines = content.splitlines()
        first = max(1, int(start))
        last = min(len(lines), int(end) if end else len(lines))
        chunk = lines[first - 1 : last]
        width = len(str(last))
        return "\n".join(f"{first + i:>{width}} | {line}" for i, line in enumerate(chunk))

    # ---------------------------------------------------------------- write
    def write_text(self, relative: str, content: str, *, create_dirs: bool = True) -> Path:
        """Create or overwrite a text file (atomic replace)."""
        target = self.resolve(relative)
        if create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".agent3.tmp")
        tmp.write_text(content, encoding="utf-8", newline="")
        os.replace(tmp, target)
        logger.info("wrote %s (%d bytes)", self.relative(target), len(content.encode("utf-8")))
        return target

    def append_text(self, relative: str, content: str) -> Path:
        target = self.resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="") as handle:
            handle.write(content)
        return target

    def mkdir(self, relative: str) -> Path:
        target = self.resolve(relative)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def delete(self, relative: str, *, recursive: bool = False) -> bool:
        """Delete a file or (optionally) a whole directory tree."""
        target = self.resolve(relative, must_exist=True)
        if target == self._root:
            raise WorkspaceSecurityError("refusing to delete the workspace root")
        if target.is_dir():
            if not recursive:
                raise WorkspaceError(
                    f"{self.relative(target)} is a directory - pass recursive=True to delete it"
                )
            shutil.rmtree(target)
        else:
            target.unlink()
        logger.info("deleted %s", self.relative(target))
        return True

    def rename(self, source: str, destination: str, *, overwrite: bool = False) -> Path:
        src = self.resolve(source, must_exist=True)
        dst = self.resolve(destination)
        if dst.exists() and not overwrite:
            raise WorkspaceError(f"destination already exists: {self.relative(dst)}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        logger.info("renamed %s -> %s", self.relative(src), self.relative(dst))
        return dst

    def copy(self, source: str, destination: str) -> Path:
        src = self.resolve(source, must_exist=True)
        dst = self.resolve(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return dst

    # --------------------------------------------------------------- search
    def search_text(
        self,
        pattern: str,
        *,
        glob: str = "*",
        max_results: int = 200,
        case_sensitive: bool = False,
        regex: bool = False,
    ) -> List[Dict[str, object]]:
        """Grep-like search returning ``{path, line, text}`` dictionaries."""
        import re as _re

        if regex:
            flags = 0 if case_sensitive else _re.IGNORECASE
            try:
                matcher = _re.compile(pattern, flags)
            except _re.error as exc:
                raise WorkspaceError(f"invalid regular expression: {exc}") from exc

            def matches(line: str) -> bool:
                return bool(matcher.search(line))
        else:
            needle = pattern if case_sensitive else pattern.lower()

            def matches(line: str) -> bool:
                return needle in (line if case_sensitive else line.lower())

        results: List[Dict[str, object]] = []
        for file_path in self.walk_files():
            if len(results) >= max_results:
                break
            if glob != "*" and not fnmatch.fnmatch(file_path.name, glob):
                continue
            rel = self.relative(file_path)
            try:
                if self.is_binary(rel):
                    continue
                with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for number, line in enumerate(handle, start=1):
                        if matches(line):
                            results.append(
                                {"path": rel, "line": number, "text": line.rstrip("\n")[:400]}
                            )
                            if len(results) >= max_results:
                                break
            except OSError:  # pragma: no cover - unreadable file
                continue
        return results

    def find_files(self, glob: str = "*", *, max_results: int = 500) -> List[str]:
        """Return workspace-relative paths matching *glob*."""
        found: List[str] = []
        for file_path in self.walk_files():
            rel = self.relative(file_path)
            if fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(file_path.name, glob):
                found.append(rel)
                if len(found) >= max_results:
                    break
        return sorted(found)

    def summarize(self, *, max_entries: int = 300) -> Dict[str, object]:
        """Lightweight project overview used to prime the agent's context."""
        counts: Dict[str, int] = {}
        total_files = 0
        total_bytes = 0
        for file_path in self.walk_files():
            total_files += 1
            try:
                total_bytes += file_path.stat().st_size
            except OSError:  # pragma: no cover
                pass
            counts[file_path.suffix.lower() or "<none>"] = (
                counts.get(file_path.suffix.lower() or "<none>", 0) + 1
            )
            if total_files >= max_entries * 10:
                break
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
        return {
            "root": str(self._root),
            "files": total_files,
            "bytes": total_bytes,
            "extensions": dict(top),
        }
