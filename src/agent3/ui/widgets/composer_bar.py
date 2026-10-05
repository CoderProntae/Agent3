"""The control strip that lives directly under the message composer.

Everything you change *per message* belongs here, next to the box you type
in - not in a toolbar at the opposite end of the window:

* the model picker,
* a thinking on/off switch,
* a reasoning-effort selector.

The thinking controls are driven entirely by what the selected model really
supports. Ollama reports that in ``/api/show`` as a ``thinking`` object with a
``values`` list - booleans for on/off models, named levels (``low``,
``medium``, ``high``, sometimes ``max``) for models like gpt-oss, which cannot
turn reasoning off at all. Nothing in this widget is invented: a model with no
thinking support shows a disabled, explanatory control instead of fake
choices.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

from agent3.llm.ollama_client import THINK_AUTO, THINK_OFF, THINK_ON, ThinkingSupport
from agent3.ui.theme import COLORS

#: Combo entry that lets the server decide.
AUTO_LABEL = "Auto"


class ComposerBar(QFrame):
    """Model + reasoning controls shown under the input box."""

    model_changed = Signal(str)
    think_changed = Signal(str)
    refresh_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ComposerBar")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"""
            QFrame#ComposerBar {{
                background-color: {COLORS.panel};
                border: 1px solid {COLORS.border_soft};
                border-radius: 8px;
            }}
            QFrame#ComposerBar QLabel {{ color: {COLORS.text_faint}; }}
            QFrame#ComposerBar QComboBox {{
                background-color: {COLORS.bg_alt};
                border: 1px solid {COLORS.border};
                border-radius: 6px;
                padding: 3px 8px;
                color: {COLORS.text};
            }}
            QFrame#ComposerBar QComboBox:disabled {{ color: {COLORS.text_faint}; }}
            QFrame#ComposerBar QCheckBox {{ color: {COLORS.text_dim}; spacing: 6px; }}
            QFrame#ComposerBar QCheckBox:disabled {{ color: {COLORS.text_faint}; }}
            """
        )

        self._support = ThinkingSupport()
        self._spec = THINK_AUTO
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)

        # ------------------------------------------------------- model
        model_icon = QLabel("◆")
        model_icon.setStyleSheet(f"color: {COLORS.accent}; font-size: 12px;")
        layout.addWidget(model_icon, 0)

        self.model_box = QComboBox()
        self.model_box.setEditable(True)
        self.model_box.setMinimumWidth(230)
        self.model_box.setToolTip("Model used for the next message")
        self.model_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.model_box.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(self.model_box, 0)

        layout.addWidget(self._separator(), 0)

        # ---------------------------------------------------- thinking
        self.think_toggle = QCheckBox("Thinking")
        self.think_toggle.setToolTip("Ask the model to reason before answering")
        self.think_toggle.toggled.connect(self._on_toggle)
        layout.addWidget(self.think_toggle, 0)

        # The caption goes BEFORE the value, so the strip reads as a sentence:
        # "Thinking · effort [High]".
        self.capability_label = QLabel("")
        self.capability_label.setStyleSheet(f"color: {COLORS.text_faint}; font-size: 11px;")
        layout.addWidget(self.capability_label, 0)

        self.level_box = QComboBox()
        self.level_box.setMinimumWidth(96)
        self.level_box.setToolTip("Reasoning effort reported by this model")
        self.level_box.currentIndexChanged.connect(self._on_level_changed)
        layout.addWidget(self.level_box, 0)

        layout.addStretch(1)

        # -------------------------------------------------- connection
        self.connection_label = QLabel("● connecting…")
        self.connection_label.setStyleSheet(f"color: {COLORS.warning}; font-size: 11px;")
        self.connection_label.setToolTip("Ollama endpoint status")
        layout.addWidget(self.connection_label, 0)

        self._apply_support()

    # ------------------------------------------------------------- helpers
    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFixedWidth(1)
        line.setStyleSheet(f"background-color: {COLORS.border}; border: none;")
        return line

    # --------------------------------------------------------------- model
    def set_models(self, models: Sequence[str], current: str = "") -> None:
        """Populate the picker, keeping the current text when possible."""
        self._updating = True
        try:
            text = current or self.model_box.currentText()
            self.model_box.clear()
            for name in models:
                if name:
                    self.model_box.addItem(name)
            if text:
                index = self.model_box.findText(text)
                if index >= 0:
                    self.model_box.setCurrentIndex(index)
                else:
                    self.model_box.setEditText(text)
        finally:
            self._updating = False

    def current_model(self) -> str:
        return self.model_box.currentText().strip()

    def set_model(self, model: str) -> None:
        self._updating = True
        try:
            index = self.model_box.findText(model)
            if index >= 0:
                self.model_box.setCurrentIndex(index)
            else:
                self.model_box.setEditText(model)
        finally:
            self._updating = False

    def _on_model_changed(self, text: str) -> None:
        if self._updating:
            return
        self.model_changed.emit(text.strip())

    # ------------------------------------------------------------ thinking
    def set_thinking_support(self, support: ThinkingSupport, spec: Optional[str] = None) -> None:
        """Rebuild the reasoning controls from a model's real capabilities."""
        self._support = support or ThinkingSupport()
        if spec is not None:
            self._spec = str(spec or THINK_AUTO)
        if not self._support.accepts(self._spec):
            self._spec = THINK_AUTO
        self._apply_support()

    def _apply_support(self) -> None:
        support = self._support
        self._updating = True
        try:
            self.level_box.clear()
            if not support.supported:
                self.think_toggle.setChecked(False)
                self.think_toggle.setEnabled(False)
                self.think_toggle.setToolTip(
                    "This model does not report a thinking capability to Ollama"
                )
                self.level_box.setVisible(False)
                self.capability_label.setText("no reasoning")
                self.capability_label.setToolTip("")
                self._spec = THINK_AUTO
                return

            self.think_toggle.setEnabled(support.can_disable)
            self.think_toggle.setToolTip(
                "Ask the model to reason before answering"
                if support.can_disable
                else f"{support.model or 'This model'} always reasons - it cannot be turned off"
            )

            levels = list(support.levels)
            if levels:
                self.level_box.setVisible(True)
                self.level_box.addItem(AUTO_LABEL, THINK_AUTO)
                for level in levels:
                    self.level_box.addItem(level.capitalize(), level)
                self.capability_label.setText("effort")
            elif True in support.values or not support.known:
                # On/off model: the checkbox alone is the whole control.
                self.level_box.setVisible(False)
                self.capability_label.setText("")
            else:  # pragma: no cover - defensive
                self.level_box.setVisible(False)
                self.capability_label.setText("")

            self.capability_label.setToolTip(support.describe())

            spec = self._spec
            if spec == THINK_OFF and support.can_disable:
                self.think_toggle.setChecked(False)
                self._select_level(THINK_AUTO)
            elif spec == THINK_AUTO:
                self.think_toggle.setChecked(support.forced or not support.can_disable)
                self._select_level(THINK_AUTO)
            else:
                self.think_toggle.setChecked(True)
                self._select_level(spec if spec in levels else THINK_AUTO)
            if support.forced:
                self.think_toggle.setChecked(True)
            self.level_box.setEnabled(self.think_toggle.isChecked())
        finally:
            self._updating = False

    def _select_level(self, value: str) -> None:
        index = self.level_box.findData(value)
        self.level_box.setCurrentIndex(index if index >= 0 else 0)

    def _current_level(self) -> str:
        data = self.level_box.currentData()
        return str(data) if data else THINK_AUTO

    def _recompute_spec(self) -> str:
        support = self._support
        if not support.supported:
            return THINK_AUTO
        if not self.think_toggle.isChecked():
            return THINK_OFF if support.can_disable else THINK_AUTO
        level = self._current_level()
        if level and level != THINK_AUTO:
            return level
        # Checked with no explicit level: "on" for boolean models, and the
        # model's own default for level-only models.
        return THINK_ON if (True in support.values or not support.known) else THINK_AUTO

    def _emit_spec(self) -> None:
        spec = self._recompute_spec()
        if spec == self._spec:
            return
        self._spec = spec
        self.think_changed.emit(spec)

    def _on_toggle(self, checked: bool) -> None:
        self.level_box.setEnabled(checked)
        if self._updating:
            return
        self._emit_spec()

    def _on_level_changed(self, _index: int) -> None:
        if self._updating:
            return
        if not self.think_toggle.isChecked() and self._current_level() != THINK_AUTO:
            self.think_toggle.setChecked(True)  # choosing an effort implies "on"
            return
        self._emit_spec()

    def think_spec(self) -> str:
        """The value to send as Ollama's ``think`` field."""
        return self._spec

    def set_think_spec(self, spec: str) -> None:
        self._spec = str(spec or THINK_AUTO)
        self._apply_support()

    def thinking_summary(self) -> str:
        """Short human description of the current reasoning setting."""
        if not self._support.supported:
            return "reasoning: unsupported"
        spec = self._spec
        if spec == THINK_AUTO:
            return "reasoning: model default"
        if spec == THINK_OFF:
            return "reasoning: off"
        if spec == THINK_ON:
            return "reasoning: on"
        return f"reasoning: {spec}"

    # ---------------------------------------------------------- connection
    def set_connection(self, ok: bool, text: str, tooltip: str = "") -> None:
        color = COLORS.success if ok else COLORS.danger
        self.connection_label.setText(f"● {text}")
        self.connection_label.setStyleSheet(f"color: {color}; font-size: 11px;")
        if tooltip:
            self.connection_label.setToolTip(tooltip)

    def set_busy(self, busy: bool) -> None:
        """Lock the controls while a run is in flight."""
        self.model_box.setEnabled(not busy)
        self.think_toggle.setEnabled(not busy and self._support.can_disable and self._support.supported)
        self.level_box.setEnabled(
            not busy and self._support.supported and self.think_toggle.isChecked()
        )

    def available_models(self) -> List[str]:
        return [self.model_box.itemText(i) for i in range(self.model_box.count())]
