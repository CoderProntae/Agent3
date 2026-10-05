"""Entry point for the standalone ``UsageLimitEditor`` executable."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from agent3.core.logging_setup import (
    configure_logging,
    ensure_std_streams,
    get_logger,
    install_excepthook,
)
from usage_limit_editor import APP_TITLE, __version__


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="UsageLimitEditor", description="Administer Agent3 usage quotas"
    )
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="build the window off-screen, then exit (smoke test for frozen builds)",
    )
    parser.add_argument("--version", action="version", version=f"{APP_TITLE} {__version__}")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Launch the administrator GUI."""
    ensure_std_streams()
    args = parse_args(argv)
    configure_logging(logging.DEBUG if args.debug else logging.INFO)
    install_excepthook()
    logger = get_logger("usage_limit_editor")
    logger.info("starting %s %s", APP_TITLE, __version__)

    if args.self_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("QT_OPENGL", "software")

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from agent3.core.paths import resource_dir
    from agent3.ui.theme import apply_theme
    from usage_limit_editor.main_window import UsageLimitEditorWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_TITLE)
    app.setApplicationVersion(__version__)
    for icon_name in ("agent3.png", "agent3.ico", "agent3.svg"):
        icon_file = resource_dir() / icon_name
        if icon_file.exists():
            app.setWindowIcon(QIcon(str(icon_file)))
            break
    apply_theme(app)

    window = UsageLimitEditorWindow()

    if args.self_test:
        app.processEvents()
        window.close()
        logger.info("self test OK")
        print(f"{APP_TITLE} {__version__} self test OK")
        return 0

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
