"""Sidebar panel showing live quota consumption."""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from agent3.limits.manager import UsageSnapshot
from agent3.ui.theme import COLORS


class _Gauge(QWidget):
    """One ``label / used-of-limit`` progress row."""

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(3)

        header = QHBoxLayout()
        header.setSpacing(6)
        self._label = QLabel(label)
        self._label.setStyleSheet(f"color: {COLORS.text_dim};")
        self._value = QLabel("-")
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._value.setStyleSheet(f"color: {COLORS.text};")
        header.addWidget(self._label, 1)
        header.addWidget(self._value, 0)
        layout.addLayout(header)

        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setRange(0, 100)
        layout.addWidget(self._bar)

    def update_values(self, used: float, limit: float, unit: str = "") -> None:
        unlimited = limit <= 0
        suffix = f" {unit}" if unit else ""
        used_text = f"{used:,.1f}".rstrip("0").rstrip(".") if isinstance(used, float) else f"{used:,}"
        if unlimited:
            self._value.setText(f"{used_text}{suffix} / ∞")
            self._bar.setValue(0)
            color = COLORS.accent_dim
        else:
            percent = int(min(100, max(0, round(used / limit * 100)))) if limit else 0
            limit_text = f"{limit:,.0f}" if float(limit).is_integer() else f"{limit:,.1f}"
            self._value.setText(f"{used_text} / {limit_text}{suffix}")
            self._bar.setValue(percent)
            if percent >= 100:
                color = COLORS.danger
            elif percent >= 80:
                color = COLORS.warning
            else:
                color = COLORS.success
        self._bar.setStyleSheet(
            f"QProgressBar {{ background-color: {COLORS.bg_alt}; border: 1px solid {COLORS.border};"
            f" border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}"
        )


class UsagePanel(QFrame):
    """Shows requests / tokens / runtime against the administrator policy."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        title = QLabel("USAGE & QUOTA")
        title.setObjectName("SectionTitle")
        title.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title)

        self._gauges: Dict[str, _Gauge] = {}
        for key, label in (
            ("requests", "Requests today"),
            ("tokens_day", "Tokens today"),
            ("tokens_session", "Tokens this session"),
            ("runtime", "Active runtime"),
            ("runs", "Agent runs today"),
        ):
            gauge = _Gauge(label)
            self._gauges[key] = gauge
            layout.addWidget(gauge)

        self._status = QLabel("Quota: OK")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLORS.text_faint};")
        layout.addWidget(self._status)

        self._editor_button = QPushButton("Open Usage Limit Editor")
        self._editor_button.setObjectName("Ghost")
        self._editor_button.setToolTip(
            "Launch the administrator tool that edits daily quotas and token limits"
        )
        layout.addWidget(self._editor_button)

    @property
    def editor_button(self) -> QPushButton:
        return self._editor_button

    def update_snapshot(self, snapshot: UsageSnapshot) -> None:
        """Refresh every gauge from a :class:`UsageSnapshot`."""
        for gauge_view in snapshot.gauges:
            widget = self._gauges.get(gauge_view.key)
            if widget is not None:
                widget.update_values(gauge_view.used, gauge_view.limit, gauge_view.unit)

        policy = snapshot.policy
        if policy.developer_mode:
            self._status.setText("Developer mode: quotas are bypassed.")
            self._status.setStyleSheet(f"color: {COLORS.purple};")
        elif not policy.enabled:
            self._status.setText("Quota enforcement disabled by the administrator.")
            self._status.setStyleSheet(f"color: {COLORS.text_faint};")
        elif snapshot.blocked:
            self._status.setText(f"BLOCKED - {snapshot.block_reason}")
            self._status.setStyleSheet(f"color: {COLORS.danger};")
        elif snapshot.warning:
            self._status.setText(snapshot.warning)
            self._status.setStyleSheet(f"color: {COLORS.warning};")
        else:
            self._status.setText("Quota: OK")
            self._status.setStyleSheet(f"color: {COLORS.text_faint};")
