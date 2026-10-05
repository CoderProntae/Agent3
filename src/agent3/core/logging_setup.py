"""Centralised logging configuration with rotating files and crash dumps.

Every subsystem obtains its logger through :func:`get_logger`.  The agent loop
relies on :func:`log_exception` to persist full stack traces so that the
self-correction step can feed them back to the model.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent3.core.paths import app_paths

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def ensure_std_streams() -> None:
    """Guarantee that ``sys.stdout``/``sys.stderr`` exist.

    A PyInstaller *windowed* build has no console, so both streams are
    ``None``; any library that writes to them (argparse ``--version``, a
    stray ``print``) would raise ``AttributeError``.  Pointing them at the
    null device keeps the frozen executable robust.
    """
    import io

    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            try:
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
            except OSError:  # pragma: no cover - extremely defensive
                setattr(sys, name, io.StringIO())


def configure_logging(level: int = logging.INFO, *, console: bool = True) -> Path:
    """Configure the root logger exactly once.

    Returns the path of the active log file.
    """
    global _CONFIGURED
    paths = app_paths()
    log_file = paths.log_file
    if _CONFIGURED:
        return log_file

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    if console and sys.stderr is not None:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root.addHandler(stream_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured -> %s", log_file)
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, exc: BaseException, context: str = "") -> str:
    """Log *exc* with a full traceback and return the formatted trace.

    The returned string is deliberately plain text: the agent loop injects it
    back into the conversation so the model can repair its own mistake.
    """
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error("%s%s", f"{context}: " if context else "", trace)
    return trace


def write_crash_dump(exc: BaseException, context: str = "") -> Optional[Path]:
    """Persist an unhandled exception to ``crashes/`` and return the path."""
    try:
        paths = app_paths()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        target = paths.crash_dir / f"crash-{stamp}.log"
        body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        target.write_text(f"context: {context}\n\n{body}", encoding="utf-8")
        return target
    except Exception:  # pragma: no cover - last resort, never raise from here
        return None


def install_excepthook() -> None:
    """Route unhandled exceptions to the log file and a crash dump."""
    logger = get_logger("agent3.crash")

    def _hook(exc_type, exc_value, exc_tb):  # pragma: no cover - global hook
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log_exception(logger, exc_value, "unhandled exception")
        write_crash_dump(exc_value, "unhandled exception")

    sys.excepthook = _hook
