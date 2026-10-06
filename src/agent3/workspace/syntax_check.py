"""Language-aware syntax and lint checking for freshly written files.

The agent's most expensive failure mode is writing a file with a broken
paren, declaring victory, and only discovering the breakage three tool calls
later. Every mutating tool therefore runs :func:`check_source` on its result
and appends the verdict to the tool output, so the model sees the error in the
very step that caused it.

Two tiers of checking:

``parse``
    Always available, always fast, no subprocess: Python via :func:`compile`,
    JSON via :mod:`json`, TOML via :mod:`tomllib`, YAML/XML/INI via the
    standard library. Pure in-process parsing - safe in a frozen ``.exe``.

``lint``
    Optional external tools (``node --check``, ``tsc --noEmit``, ``eslint``,
    ``ruff``) that run only when the binary is actually installed and only
    with a short timeout. A missing linter is reported as "skipped", never as
    a failure - the agent must not be blocked by the user's toolchain.
"""

from __future__ import annotations

import ast
import configparser
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agent3.core.logging_setup import get_logger
from agent3.workspace.outline import detect_language

logger = get_logger(__name__)

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    tomllib = None  # type: ignore[assignment]

#: Seconds an external linter may run before it is abandoned.
LINT_TIMEOUT = 20.0

#: Extensions handled by the in-process parsers.
_DATA_LANGUAGES = {
    ".json": "json",
    ".jsonc": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".ini": "ini",
    ".cfg": "ini",
}


@dataclass
class Diagnostic:
    """One problem found in a file."""

    severity: str  # error | warning
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    source: str = "parser"

    def format(self) -> str:
        where = ""
        if self.line:
            where = f"line {self.line}"
            if self.column:
                where += f", col {self.column}"
            where = f" ({where})"
        return f"{self.severity}{where}: {self.message} [{self.source}]"


@dataclass
class CheckResult:
    """Outcome of checking a single file."""

    path: str
    language: str
    checked: bool
    diagnostics: List[Diagnostic] = field(default_factory=list)
    skipped_reason: str = ""
    tools_used: List[str] = field(default_factory=list)

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.severity != "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "language": self.language,
            "checked": self.checked,
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "tools": list(self.tools_used),
        }

    def summary(self) -> str:
        """One-line verdict used as a suffix on tool output."""
        if not self.checked:
            return f"syntax check skipped ({self.skipped_reason})" if self.skipped_reason else ""
        if self.ok and not self.warnings:
            tools = "+".join(self.tools_used) or self.language or "parser"
            return f"syntax OK ({tools})"
        if self.ok:
            return f"syntax OK, {len(self.warnings)} warning(s)"
        return f"SYNTAX ERROR x{len(self.errors)}"

    def report(self, max_items: int = 10) -> str:
        """Full human/model readable report."""
        if not self.checked:
            return self.summary()
        if self.ok and not self.warnings:
            return self.summary()
        lines = [self.summary() + f" in {self.path}:"]
        for diagnostic in (self.errors + self.warnings)[:max_items]:
            lines.append("  " + diagnostic.format())
        remaining = len(self.diagnostics) - max_items
        if remaining > 0:
            lines.append(f"  ... and {remaining} more")
        return "\n".join(lines)


# --------------------------------------------------------------- parsers
def _check_python(path: str, source: str) -> List[Diagnostic]:
    try:
        compile(source, path, "exec", ast.PyCF_ONLY_AST)
    except SyntaxError as exc:
        return [
            Diagnostic(
                severity="error",
                message=exc.msg or "invalid syntax",
                line=exc.lineno,
                column=exc.offset,
                source="python-ast",
            )
        ]
    except ValueError as exc:  # null bytes and friends
        return [Diagnostic("error", str(exc), source="python-ast")]
    return []


def _check_json(path: str, source: str) -> List[Diagnostic]:
    if not source.strip():
        return [Diagnostic("error", "empty JSON document", line=1, source="json")]
    try:
        json.loads(source)
    except json.JSONDecodeError as exc:
        return [Diagnostic("error", exc.msg, line=exc.lineno, column=exc.colno, source="json")]
    return []


def _check_toml(path: str, source: str) -> List[Diagnostic]:
    if tomllib is None:  # pragma: no cover - 3.11+ guaranteed by packaging
        return []
    try:
        tomllib.loads(source)
    except Exception as exc:  # tomllib.TOMLDecodeError
        return [Diagnostic("error", str(exc), source="toml")]
    return []


def _check_yaml(path: str, source: str) -> List[Diagnostic]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _check_yaml_indent(source)
    try:
        list(yaml.safe_load_all(source))
    except Exception as exc:  # yaml.YAMLError
        mark = getattr(exc, "problem_mark", None)
        line = (mark.line + 1) if mark is not None else None
        column = (mark.column + 1) if mark is not None else None
        message = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        return [Diagnostic("error", str(message), line=line, column=column, source="yaml")]
    return []


def _check_yaml_indent(source: str) -> List[Diagnostic]:
    """Tab detection - the single most common YAML breakage without PyYAML."""
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.lstrip(" ")
        if stripped.startswith("\t"):
            return [
                Diagnostic(
                    "error",
                    "YAML forbids tab characters for indentation",
                    line=number,
                    source="yaml-lite",
                )
            ]
    return []


def _check_xml(path: str, source: str) -> List[Diagnostic]:
    try:
        ET.fromstring(source)
    except ET.ParseError as exc:
        line, column = getattr(exc, "position", (None, None))
        return [Diagnostic("error", str(exc), line=line, column=column, source="xml")]
    return []


def _check_ini(path: str, source: str) -> List[Diagnostic]:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        parser.read_string(source, source=path)
    except configparser.Error as exc:
        return [Diagnostic("error", str(exc).splitlines()[0], source="ini")]
    return []


# --------------------------------------------------------- external tools
def _which(binary: str) -> Optional[str]:
    """Locate *binary*, including the local ``node_modules/.bin`` directory."""
    found = shutil.which(binary)
    if found:
        return found
    if sys.platform.startswith("win"):  # pragma: no cover - windows only
        for suffix in (".cmd", ".exe", ".bat"):
            found = shutil.which(binary + suffix)
            if found:
                return found
    return None


def _local_bin(root: Path, binary: str) -> Optional[str]:
    candidate = root / "node_modules" / ".bin" / binary
    if sys.platform.startswith("win"):  # pragma: no cover - windows only
        for suffix in (".cmd", ".exe", ""):
            if (candidate.parent / (binary + suffix)).exists():
                return str(candidate.parent / (binary + suffix))
        return None
    return str(candidate) if candidate.exists() else None


def _run(command: Sequence[str], cwd: Path, timeout: float = LINT_TIMEOUT):
    env = dict(os.environ)
    env.setdefault("CI", "1")
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_node_check(output: str, filename: str) -> List[Diagnostic]:
    """Turn ``node --check`` stderr into a diagnostic."""
    diagnostics: List[Diagnostic] = []
    line_no: Optional[int] = None
    message = ""
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line and line.split(":")[0].endswith(Path(filename).name):
            tail = line.rsplit(":", 1)[-1]
            if tail.isdigit():
                line_no = int(tail)
        if line.startswith("SyntaxError:") or line.endswith("SyntaxError"):
            message = line
    if message:
        diagnostics.append(Diagnostic("error", message, line=line_no, source="node"))
    return diagnostics


def _check_javascript(abs_path: Path, root: Path) -> tuple[List[Diagnostic], List[str], str]:
    """``node --check`` for JS; it is the fastest real parser available."""
    node = _which("node")
    if node is None:
        return [], [], "node is not installed"
    try:
        completed = _run([node, "--check", str(abs_path)], root)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return [], [], f"node --check failed to run ({exc})"
    if completed.returncode == 0:
        return [], ["node"], ""
    diagnostics = _parse_node_check(completed.stderr or completed.stdout, abs_path.name)
    if not diagnostics:
        tail = (completed.stderr or completed.stdout).strip().splitlines()
        diagnostics = [Diagnostic("error", tail[-1] if tail else "syntax error", source="node")]
    return diagnostics, ["node"], ""


_TS_DIAGNOSTIC = "): error TS"


def _parse_tsc(output: str, wanted: str) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    for raw in output.splitlines():
        line = raw.strip()
        if _TS_DIAGNOSTIC not in line:
            continue
        head, _, message = line.partition(": ")
        location = head[head.rfind("(") + 1 : head.rfind(")")] if "(" in head else ""
        file_part = head.split("(")[0]
        if wanted and Path(file_part).name != Path(wanted).name:
            continue
        numbers = [part for part in location.split(",") if part.strip().isdigit()]
        diagnostics.append(
            Diagnostic(
                "error",
                message or line,
                line=int(numbers[0]) if numbers else None,
                column=int(numbers[1]) if len(numbers) > 1 else None,
                source="tsc",
            )
        )
    return diagnostics


def _check_typescript(abs_path: Path, root: Path) -> tuple[List[Diagnostic], List[str], str]:
    tsc = _local_bin(root, "tsc") or _which("tsc")
    if tsc is None:
        return [], [], "tsc is not installed"
    command = [tsc, "--noEmit", "--skipLibCheck"]
    if (root / "tsconfig.json").exists():
        command += ["--project", str(root / "tsconfig.json")]
    else:
        command += ["--allowJs", "--target", "es2020", "--moduleResolution", "node", str(abs_path)]
    try:
        completed = _run(command, root, timeout=LINT_TIMEOUT * 2)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return [], [], f"tsc did not finish ({exc})"
    if completed.returncode == 0:
        return [], ["tsc"], ""
    diagnostics = _parse_tsc(completed.stdout + completed.stderr, str(abs_path))
    if not diagnostics:
        # The project build failed elsewhere - do not blame this file.
        return [], ["tsc"], "tsc reported errors in other files"
    return diagnostics, ["tsc"], ""


def _parse_eslint_json(payload: str) -> List[Diagnostic]:
    try:
        reports = json.loads(payload)
    except json.JSONDecodeError:
        return []
    diagnostics: List[Diagnostic] = []
    for report in reports if isinstance(reports, list) else []:
        for item in report.get("messages", []):
            diagnostics.append(
                Diagnostic(
                    severity="error" if item.get("severity") == 2 else "warning",
                    message=str(item.get("message", "")).strip()
                    + (f" ({item['ruleId']})" if item.get("ruleId") else ""),
                    line=item.get("line"),
                    column=item.get("column"),
                    source="eslint",
                )
            )
    return diagnostics


def _check_eslint(abs_path: Path, root: Path) -> tuple[List[Diagnostic], List[str], str]:
    eslint = _local_bin(root, "eslint") or _which("eslint")
    if eslint is None:
        return [], [], ""
    has_config = any(
        (root / name).exists()
        for name in (
            "eslint.config.js",
            "eslint.config.mjs",
            "eslint.config.cjs",
            ".eslintrc",
            ".eslintrc.js",
            ".eslintrc.cjs",
            ".eslintrc.json",
            ".eslintrc.yml",
            ".eslintrc.yaml",
        )
    )
    if not has_config:
        return [], [], ""
    try:
        completed = _run([eslint, "--format", "json", str(abs_path)], root)
    except (subprocess.TimeoutExpired, OSError):
        return [], [], ""
    return _parse_eslint_json(completed.stdout), ["eslint"], ""


def _check_ruff(abs_path: Path, root: Path) -> tuple[List[Diagnostic], List[str], str]:
    ruff = _which("ruff")
    if ruff is None:
        return [], [], ""
    try:
        completed = _run([ruff, "check", "--output-format", "json", str(abs_path)], root)
    except (subprocess.TimeoutExpired, OSError):
        return [], [], ""
    if completed.returncode not in (0, 1):
        return [], [], ""
    try:
        items = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return [], [], ""
    diagnostics = []
    for item in items if isinstance(items, list) else []:
        location = item.get("location") or {}
        code = item.get("code") or ""
        diagnostics.append(
            Diagnostic(
                severity="warning",
                message=f"{item.get('message', '')}" + (f" ({code})" if code else ""),
                line=location.get("row"),
                column=location.get("column"),
                source="ruff",
            )
        )
    return diagnostics, ["ruff"], ""


# ------------------------------------------------------------- front door
def supported_languages() -> List[str]:
    """Languages the checker understands without any external tool."""
    return ["python", "json", "toml", "yaml", "xml", "ini", "javascript", "typescript"]


def check_source(
    relative_path: str,
    source: str,
    *,
    root: Optional[Path] = None,
    absolute_path: Optional[Path] = None,
    run_linters: bool = True,
) -> CheckResult:
    """Check *source* as if it were saved at *relative_path*.

    External linters only run when *absolute_path* exists on disk, because
    ``eslint`` and ``tsc`` need a real file. In-process parsers always run.
    """
    suffix = Path(relative_path).suffix.lower()
    language = detect_language(relative_path) or _DATA_LANGUAGES.get(suffix, "")
    result = CheckResult(path=relative_path, language=language or suffix.lstrip("."), checked=True)

    if "\x00" in source:
        result.diagnostics.append(Diagnostic("error", "file contains NUL bytes", source="binary"))
        return result

    data_kind = _DATA_LANGUAGES.get(suffix)
    if language == "python":
        result.diagnostics.extend(_check_python(relative_path, source))
        result.tools_used.append("python-ast")
    elif data_kind == "json":
        result.diagnostics.extend(_check_json(relative_path, source))
        result.tools_used.append("json")
    elif data_kind == "toml":
        result.diagnostics.extend(_check_toml(relative_path, source))
        result.tools_used.append("toml")
    elif data_kind == "yaml":
        result.diagnostics.extend(_check_yaml(relative_path, source))
        result.tools_used.append("yaml")
    elif data_kind == "xml":
        result.diagnostics.extend(_check_xml(relative_path, source))
        result.tools_used.append("xml")
    elif data_kind == "ini":
        result.diagnostics.extend(_check_ini(relative_path, source))
        result.tools_used.append("ini")
    elif language not in ("javascript", "typescript"):
        result.checked = False
        result.skipped_reason = f"no checker for {language or suffix or 'this file type'}"
        return result

    # ---- external tools ------------------------------------------------
    if not run_linters or absolute_path is None or root is None or not absolute_path.exists():
        if language in ("javascript", "typescript") and not result.tools_used:
            result.checked = False
            result.skipped_reason = "file not on disk yet"
        return result

    if result.errors:
        # A hard parse error makes linting noise; report the real cause only.
        return result

    checkers = []
    if language == "javascript":
        checkers = [_check_javascript, _check_eslint]
    elif language == "typescript":
        checkers = [_check_typescript, _check_eslint]
    elif language == "python":
        checkers = [_check_ruff]

    skipped: List[str] = []
    for checker in checkers:
        diagnostics, tools, reason = checker(absolute_path, root)
        result.diagnostics.extend(diagnostics)
        result.tools_used.extend(tools)
        if reason:
            skipped.append(reason)
        if any(d.severity == "error" for d in diagnostics):
            break

    if not result.tools_used:
        result.checked = False
        result.skipped_reason = "; ".join(skipped) or "no checker available"
    return result


def check_file(
    relative_path: str,
    *,
    root: Path,
    run_linters: bool = True,
    max_bytes: int = 2_000_000,
) -> CheckResult:
    """Read the file from disk and check it."""
    absolute = (root / relative_path).resolve()
    try:
        raw = absolute.read_bytes()[:max_bytes]
        source = raw.decode("utf-8")
    except FileNotFoundError:
        return CheckResult(
            path=relative_path,
            language="",
            checked=False,
            skipped_reason="file does not exist",
        )
    except UnicodeDecodeError:
        return CheckResult(
            path=relative_path,
            language="",
            checked=False,
            skipped_reason="binary file",
        )
    return check_source(
        relative_path,
        source,
        root=root,
        absolute_path=absolute,
        run_linters=run_linters,
    )
