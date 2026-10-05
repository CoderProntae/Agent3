"""Administrator GUI for editing Agent3's encrypted quota policy."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from agent3.core.logging_setup import get_logger
from agent3.core.paths import app_paths
from agent3.limits.policy import PolicyStore, QuotaPolicy
from agent3.limits.store import UsageStore, today_key
from agent3.ui.theme import COLORS
from usage_limit_editor import APP_TITLE, __version__

logger = get_logger(__name__)

UNLIMITED_HINT = "0 = unlimited"


class UsageLimitEditorWindow(QMainWindow):
    """Edit quotas, inspect consumption and reset counters."""

    def __init__(
        self,
        policy_store: Optional[PolicyStore] = None,
        usage_store: Optional[UsageStore] = None,
    ) -> None:
        super().__init__()
        self.policies = policy_store or PolicyStore()
        self.usage = usage_store or UsageStore()
        self.policy: QuotaPolicy = self.policies.load(force=True)
        self._unlocked = not self.policy.locked

        self.setWindowTitle(APP_TITLE)
        self.resize(760, 760)
        self.setMinimumSize(640, 600)

        self._build_ui()
        self._load_policy_into_form()
        self._refresh_usage()

        self._timer = QTimer(self)
        self._timer.setInterval(5000)
        self._timer.timeout.connect(self._refresh_usage)
        self._timer.start()

        if not self._unlocked:
            QTimer.singleShot(100, self._prompt_unlock)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        header = QLabel(f"<b>Agent3 Usage Limit Editor</b> &nbsp; <span>v{__version__}</span>")
        header.setStyleSheet(f"color: {COLORS.text};")
        layout.addWidget(header)

        self._lock_label = QLabel("")
        self._lock_label.setWordWrap(True)
        layout.addWidget(self._lock_label)

        tabs = QTabWidget()
        tabs.addTab(self._build_limits_tab(), "Limits")
        tabs.addTab(self._build_security_tab(), "Security")
        tabs.addTab(self._build_usage_tab(), "Usage")
        tabs.addTab(self._build_about_tab(), "Storage")
        layout.addWidget(tabs, 1)

        buttons = QHBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {COLORS.text_faint};")
        buttons.addWidget(self._status, 1)

        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self._reload)
        buttons.addWidget(reload_button)

        defaults_button = QPushButton("Restore defaults")
        defaults_button.clicked.connect(self._restore_defaults)
        buttons.addWidget(defaults_button)

        self._save_button = QPushButton("Save policy")
        self._save_button.setObjectName("Primary")
        self._save_button.clicked.connect(self._save)
        buttons.addWidget(self._save_button)
        layout.addLayout(buttons)

    def _build_limits_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setSpacing(12)

        enforcement = QGroupBox("Enforcement")
        enforcement_form = QFormLayout(enforcement)
        self._enabled = QCheckBox("Enforce quotas on this workstation")
        enforcement_form.addRow("", self._enabled)
        self._developer_mode = QCheckBox("Developer mode (bypass every limit)")
        self._developer_mode.setToolTip(
            "Temporarily lifts all quotas. The main application shows a purple banner while active."
        )
        enforcement_form.addRow("", self._developer_mode)
        self._warn_threshold = QDoubleSpinBox()
        self._warn_threshold.setRange(0.1, 1.0)
        self._warn_threshold.setSingleStep(0.05)
        self._warn_threshold.setToolTip("Fraction of a quota at which the UI starts warning.")
        enforcement_form.addRow("Warning threshold", self._warn_threshold)
        outer.addWidget(enforcement)

        quotas = QGroupBox(f"Quotas ({UNLIMITED_HINT})")
        form = QFormLayout(quotas)
        form.setSpacing(8)

        self._requests_per_day = QSpinBox()
        self._requests_per_day.setRange(0, 1_000_000)
        self._requests_per_day.setSingleStep(25)
        form.addRow("Max requests / day", self._requests_per_day)

        self._tokens_per_day = QSpinBox()
        self._tokens_per_day.setRange(0, 1_000_000_000)
        self._tokens_per_day.setSingleStep(50_000)
        self._tokens_per_day.setGroupSeparatorShown(True)
        form.addRow("Max tokens / day", self._tokens_per_day)

        self._tokens_per_session = QSpinBox()
        self._tokens_per_session.setRange(0, 1_000_000_000)
        self._tokens_per_session.setSingleStep(10_000)
        self._tokens_per_session.setGroupSeparatorShown(True)
        form.addRow("Max tokens / session", self._tokens_per_session)

        self._tokens_per_request = QSpinBox()
        self._tokens_per_request.setRange(0, 10_000_000)
        self._tokens_per_request.setSingleStep(2_000)
        self._tokens_per_request.setGroupSeparatorShown(True)
        form.addRow("Max prompt tokens / request", self._tokens_per_request)

        self._runtime_per_day = QSpinBox()
        self._runtime_per_day.setRange(0, 24 * 3600)
        self._runtime_per_day.setSingleStep(600)
        self._runtime_per_day.setSuffix(" s")
        form.addRow("Max active runtime / day", self._runtime_per_day)

        self._runs_per_day = QSpinBox()
        self._runs_per_day.setRange(0, 100_000)
        form.addRow("Max agent runs / day", self._runs_per_day)

        self._tool_calls_per_run = QSpinBox()
        self._tool_calls_per_run.setRange(0, 10_000)
        form.addRow("Max tool calls / run", self._tool_calls_per_run)

        self._cooldown = QDoubleSpinBox()
        self._cooldown.setRange(0.0, 600.0)
        self._cooldown.setSingleStep(0.5)
        self._cooldown.setSuffix(" s")
        form.addRow("Min delay between requests", self._cooldown)
        outer.addWidget(quotas)

        notes_box = QGroupBox("Administrator notes")
        notes_layout = QVBoxLayout(notes_box)
        self._notes = QPlainTextEdit()
        self._notes.setMaximumHeight(90)
        self._notes.setPlaceholderText("Free-form note stored inside the encrypted policy file.")
        notes_layout.addWidget(self._notes)
        outer.addWidget(notes_box)

        outer.addStretch(1)
        return page

    def _build_security_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        box = QGroupBox("Administrator password")
        form = QFormLayout(box)
        info = QLabel(
            "When a password is set, this editor asks for it on launch. The policy file itself "
            "is always AES-256-GCM encrypted and integrity protected, so hand editing it makes "
            "Agent3 fall back to the conservative default quotas."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {COLORS.text_dim};")
        form.addRow(info)

        self._new_password = QLineEdit()
        self._new_password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("New password", self._new_password)

        self._confirm_password = QLineEdit()
        self._confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Confirm password", self._confirm_password)

        row = QHBoxLayout()
        set_button = QPushButton("Set password")
        set_button.clicked.connect(self._set_password)
        row.addWidget(set_button)
        clear_button = QPushButton("Remove password")
        clear_button.setObjectName("Danger")
        clear_button.clicked.connect(self._clear_password)
        row.addWidget(clear_button)
        row.addStretch(1)
        form.addRow("", row)

        outer.addWidget(box)
        outer.addStretch(1)
        return page

    def _build_usage_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        self._today_label = QLabel("")
        self._today_label.setStyleSheet(f"color: {COLORS.text};")
        outer.addWidget(self._today_label)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Day", "Requests", "Tokens", "Runtime (min)"])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        outer.addWidget(self._table, 1)

        row = QHBoxLayout()
        reset_today = QPushButton("Reset today's counters")
        reset_today.clicked.connect(self._reset_today)
        row.addWidget(reset_today)

        reset_all = QPushButton("Delete all usage history")
        reset_all.setObjectName("Danger")
        reset_all.clicked.connect(self._reset_all)
        row.addWidget(reset_all)
        row.addStretch(1)
        outer.addLayout(row)
        return page

    def _build_about_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        paths = app_paths()
        for label, value in (
            ("Policy file", str(self.policies.path)),
            ("Usage database", str(self.usage.path)),
            ("Configuration", str(paths.config_file)),
            ("Log folder", str(paths.log_dir)),
            ("Data folder", str(paths.base)),
        ):
            field = QLineEdit(value)
            field.setReadOnly(True)
            field.setCursorPosition(0)
            layout.addRow(label, field)
        note = QLabel(
            "The main Agent3 application re-reads the policy automatically whenever this file "
            "changes - no restart required."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLORS.text_faint};")
        layout.addRow(note)
        return page

    # ------------------------------------------------------------- policy
    def _load_policy_into_form(self) -> None:
        policy = self.policy
        self._enabled.setChecked(policy.enabled)
        self._developer_mode.setChecked(policy.developer_mode)
        self._warn_threshold.setValue(policy.warn_threshold)
        self._requests_per_day.setValue(policy.max_requests_per_day)
        self._tokens_per_day.setValue(policy.max_tokens_per_day)
        self._tokens_per_session.setValue(policy.max_tokens_per_session)
        self._tokens_per_request.setValue(policy.max_tokens_per_request)
        self._runtime_per_day.setValue(policy.max_runtime_seconds_per_day)
        self._runs_per_day.setValue(policy.max_agent_runs_per_day)
        self._tool_calls_per_run.setValue(policy.max_tool_calls_per_run)
        self._cooldown.setValue(policy.min_seconds_between_requests)
        self._notes.setPlainText(policy.notes)
        self._update_lock_label()

    def _collect_policy(self) -> QuotaPolicy:
        policy = QuotaPolicy(
            enabled=self._enabled.isChecked(),
            developer_mode=self._developer_mode.isChecked(),
            warn_threshold=float(self._warn_threshold.value()),
            max_requests_per_day=int(self._requests_per_day.value()),
            max_tokens_per_day=int(self._tokens_per_day.value()),
            max_tokens_per_session=int(self._tokens_per_session.value()),
            max_tokens_per_request=int(self._tokens_per_request.value()),
            max_runtime_seconds_per_day=int(self._runtime_per_day.value()),
            max_agent_runs_per_day=int(self._runs_per_day.value()),
            max_tool_calls_per_run=int(self._tool_calls_per_run.value()),
            min_seconds_between_requests=float(self._cooldown.value()),
            notes=self._notes.toPlainText()[:4000],
        )
        policy.admin_password_hash = self.policy.admin_password_hash
        policy.admin_password_salt = self.policy.admin_password_salt
        policy.locked = self.policy.locked
        return policy

    def _save(self) -> None:
        if not self._ensure_unlocked():
            return
        try:
            self.policy = self.policies.save(self._collect_policy(), updated_by="UsageLimitEditor")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._status.setText(f"Saved at {self.policy.updated_at}")
        self._status.setStyleSheet(f"color: {COLORS.success};")

    def _reload(self) -> None:
        self.policy = self.policies.load(force=True)
        self._load_policy_into_form()
        self._status.setText("Reloaded from disk")
        self._status.setStyleSheet(f"color: {COLORS.text_faint};")

    def _restore_defaults(self) -> None:
        if not self._ensure_unlocked():
            return
        confirm = QMessageBox.question(
            self,
            "Restore defaults",
            "Reset every quota to its factory value?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.policy = self.policies.reset_to_defaults(keep_password=True)
        self._load_policy_into_form()
        self._status.setText("Defaults restored")
        self._status.setStyleSheet(f"color: {COLORS.warning};")

    # ----------------------------------------------------------- security
    def _update_lock_label(self) -> None:
        if self.policy.locked and not self._unlocked:
            self._lock_label.setText("🔒 Locked - enter the administrator password to make changes.")
            self._lock_label.setStyleSheet(f"color: {COLORS.danger};")
        elif self.policy.locked:
            self._lock_label.setText("🔓 Unlocked for this session (password protected).")
            self._lock_label.setStyleSheet(f"color: {COLORS.success};")
        else:
            self._lock_label.setText("No administrator password set - anyone can edit these limits.")
            self._lock_label.setStyleSheet(f"color: {COLORS.text_faint};")
        self._save_button.setEnabled(self._unlocked)

    def _prompt_unlock(self) -> None:
        password, ok = QInputDialog.getText(
            self,
            "Administrator password",
            "Enter the administrator password:",
            QLineEdit.EchoMode.Password,
        )
        if not ok:
            return
        if self.policy.verify_admin_password(password):
            self._unlocked = True
            self._status.setText("Unlocked")
            self._status.setStyleSheet(f"color: {COLORS.success};")
        else:
            QMessageBox.warning(self, "Access denied", "Incorrect password.")
        self._update_lock_label()

    def _ensure_unlocked(self) -> bool:
        if self._unlocked:
            return True
        self._prompt_unlock()
        return self._unlocked

    def _set_password(self) -> None:
        if not self._ensure_unlocked():
            return
        password = self._new_password.text()
        confirm = self._confirm_password.text()
        if len(password) < 4:
            QMessageBox.warning(self, "Password", "Use at least 4 characters.")
            return
        if password != confirm:
            QMessageBox.warning(self, "Password", "The two passwords do not match.")
            return
        policy = self._collect_policy()
        policy.set_admin_password(password)
        self.policy = self.policies.save(policy, updated_by="UsageLimitEditor")
        self._new_password.clear()
        self._confirm_password.clear()
        self._unlocked = True
        self._update_lock_label()
        QMessageBox.information(self, "Password", "Administrator password updated.")

    def _clear_password(self) -> None:
        if not self._ensure_unlocked():
            return
        policy = self._collect_policy()
        policy.set_admin_password("")
        self.policy = self.policies.save(policy, updated_by="UsageLimitEditor")
        self._update_lock_label()
        QMessageBox.information(self, "Password", "Administrator password removed.")

    # -------------------------------------------------------------- usage
    def _refresh_usage(self) -> None:
        totals = self.usage.totals_for_day()
        self._today_label.setText(
            f"Today ({today_key()}): {totals.requests} requests · {totals.total_tokens:,} tokens · "
            f"{totals.runtime_seconds / 60:.1f} min · {totals.agent_runs} runs · "
            f"{totals.tool_calls} tool calls · {totals.errors} errors"
        )
        history = self.usage.daily_history(21)
        self._table.setRowCount(len(history))
        for row, entry in enumerate(history):
            values = [
                entry["day"],
                f"{entry['requests']:,}",
                f"{entry['tokens']:,}",
                f"{entry['runtime_ms'] / 60000:.1f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row, column, item)

    def _reset_today(self) -> None:
        if not self._ensure_unlocked():
            return
        removed = self.usage.reset_day()
        self._refresh_usage()
        self._status.setText(f"Removed {removed} event(s) for today")
        self._status.setStyleSheet(f"color: {COLORS.warning};")

    def _reset_all(self) -> None:
        if not self._ensure_unlocked():
            return
        confirm = QMessageBox.question(
            self,
            "Delete history",
            "Permanently delete the whole usage history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        removed = self.usage.reset_all()
        self._refresh_usage()
        self._status.setText(f"Deleted {removed} event(s)")
        self._status.setStyleSheet(f"color: {COLORS.danger};")

    # ---------------------------------------------------------- lifecycle
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._timer.stop()
        self.usage.close()
        event.accept()
