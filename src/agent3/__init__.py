"""Agent3 - a local, autonomous AI coding agent and workspace desktop app.

The package is split into clearly separated layers so that every piece can be
unit tested without a running Qt event loop or a live Ollama server:

``agent3.core``
    Cross cutting concerns: paths, logging, encrypted storage, configuration.
``agent3.llm``
    Transport to the local Ollama inference server plus token accounting.
``agent3.workspace``
    Sandboxed filesystem, diffing, terminal execution and git integration.
``agent3.agent``
    The autonomous tool-calling loop and session persistence.
``agent3.ui``
    PySide6 desktop front-end (dark, multi-pane IDE style shell).
"""

from __future__ import annotations

__all__ = ["__version__", "APP_NAME", "APP_ID"]

__version__ = "1.0.0"
APP_NAME = "Agent3"
APP_ID = "agent3"
