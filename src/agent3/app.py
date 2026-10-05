"""Application bootstrap for the Agent3 desktop client."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from agent3 import APP_NAME, __version__
from agent3.core.config import ConfigManager
from agent3.core.logging_setup import (
    configure_logging,
    ensure_std_streams,
    get_logger,
    install_excepthook,
)
from agent3.core.paths import app_paths, resource_dir


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line of the desktop application."""
    parser = argparse.ArgumentParser(
        prog="agent3", description=f"{APP_NAME} - local autonomous AI coding workspace"
    )
    parser.add_argument("workspace", nargs="?", help="folder to mount as the agent workspace")
    parser.add_argument("--model", help="override the Ollama model tag")
    parser.add_argument("--host", help="override the Ollama host (default: localhost)")
    parser.add_argument("--port", type=int, help="override the Ollama port (default: 11435)")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point used by both ``python -m agent3`` and the frozen binary."""
    ensure_std_streams()
    args = parse_args(argv)
    log_file = configure_logging(logging.DEBUG if args.debug else logging.INFO)
    install_excepthook()
    logger = get_logger("agent3.app")
    logger.info("starting %s %s (log: %s)", APP_NAME, __version__, log_file)

    # Imported late so that ``--version`` works without a display server.
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from agent3.ui.main_window import MainWindow
    from agent3.ui.theme import apply_theme

    config_manager = ConfigManager()
    config = config_manager.config
    if args.model:
        config.ollama.model = args.model
    if args.host:
        config.ollama.host = args.host
    if args.port:
        config.ollama.port = int(args.port)
    if args.workspace:
        candidate = Path(args.workspace).expanduser()
        if candidate.is_dir():
            config.ui.last_workspace = str(candidate.resolve())
        else:
            logger.warning("workspace %s does not exist - ignoring", candidate)
    if args.model or args.host or args.port or args.workspace:
        config_manager.save()

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, False)
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(APP_NAME)

    for icon_name in ("agent3.png", "agent3.ico", "agent3.svg"):
        icon_file = resource_dir() / icon_name
        if icon_file.exists():
            app.setWindowIcon(QIcon(str(icon_file)))
            break

    apply_theme(app, accent=config.ui.accent, font_size=config.ui.ui_font_size)

    window = MainWindow(config_manager)
    window.show()
    logger.info("data directory: %s", app_paths().base)
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
