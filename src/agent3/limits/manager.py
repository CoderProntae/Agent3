"""The rate limiting / quota enforcement engine.

:class:`UsageManager` is the single gate every LLM call and every tool call
has to pass through.  It combines the administrator :class:`QuotaPolicy` with
the recorded :class:`UsageStore` telemetry and answers one question:

    *May this action run right now - and if not, why?*

It also produces the :class:`UsageSnapshot` consumed by the sidebar widget.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent3.core.logging_setup import get_logger
from agent3.limits.policy import PolicyStore, QuotaPolicy
from agent3.limits.store import UsageEvent, UsageStore, UsageTotals, today_key

logger = get_logger(__name__)


@dataclass
class QuotaDecision:
    """Outcome of a quota check."""

    allowed: bool
    reason: str = ""
    code: str = "ok"
    #: ``info`` | ``warning`` | ``blocked``
    severity: str = "info"
    retry_after_seconds: float = 0.0

    def __bool__(self) -> bool:  # pragma: no cover - sugar
        return self.allowed


@dataclass
class GaugeView:
    """A single ``used / limit`` gauge rendered by the usage panel."""

    key: str
    label: str
    used: float
    limit: float
    unit: str = ""

    @property
    def unlimited(self) -> bool:
        return self.limit <= 0

    @property
    def ratio(self) -> float:
        if self.unlimited:
            return 0.0
        return min(1.0, max(0.0, self.used / self.limit))

    @property
    def percent(self) -> int:
        return int(round(self.ratio * 100))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "used": self.used,
            "limit": self.limit,
            "unit": self.unit,
            "percent": self.percent,
            "unlimited": self.unlimited,
        }


@dataclass
class UsageSnapshot:
    """Everything the UI needs to render the usage sidebar in one object."""

    day: str
    session_id: str
    policy: QuotaPolicy
    day_totals: UsageTotals
    session_totals: UsageTotals
    gauges: List[GaugeView] = field(default_factory=list)
    blocked: bool = False
    warning: str = ""
    block_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.day,
            "session_id": self.session_id,
            "blocked": self.blocked,
            "warning": self.warning,
            "block_reason": self.block_reason,
            "day_totals": self.day_totals.to_dict(),
            "session_totals": self.session_totals.to_dict(),
            "gauges": [g.to_dict() for g in self.gauges],
            "developer_mode": self.policy.developer_mode,
            "enabled": self.policy.enabled,
        }


class UsageManager:
    """Quota gate + telemetry recorder.

    Parameters
    ----------
    store:
        Telemetry database.  Created from the default path when omitted.
    policy_store:
        Encrypted policy file shared with ``UsageLimitEditor.exe``.
    session_id:
        Identifier for the *current* chat session (per-session token quota).
    """

    def __init__(
        self,
        store: Optional[UsageStore] = None,
        policy_store: Optional[PolicyStore] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.store = store or UsageStore()
        self.policies = policy_store or PolicyStore()
        self._session_id = session_id or uuid.uuid4().hex[:16]
        self._lock = threading.RLock()
        self._tool_calls_this_run = 0
        self._last_request_at: Optional[datetime] = self.store.last_event_time("llm_request")

    # ------------------------------------------------------------- session
    @property
    def session_id(self) -> str:
        return self._session_id

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Begin a new quota session (resets the per-session token budget)."""
        with self._lock:
            self._session_id = session_id or uuid.uuid4().hex[:16]
            return self._session_id

    def begin_run(self) -> None:
        """Mark the start of one autonomous agent run."""
        with self._lock:
            self._tool_calls_this_run = 0
        self.store.record(UsageEvent(kind="agent_run", session_id=self._session_id))

    @property
    def policy(self) -> QuotaPolicy:
        """Hot-reloaded administrator policy."""
        return self.policies.load()

    # --------------------------------------------------------------- checks
    def check_request(self, estimated_tokens: int = 0) -> QuotaDecision:
        """Pre-authorise one LLM request of *estimated_tokens* prompt tokens."""
        policy = self.policy
        if not policy.enabled or policy.developer_mode:
            return QuotaDecision(True, "quota enforcement disabled", "bypass")

        day = self.store.totals_for_day()
        session = self.store.totals_for_session(self._session_id)

        if policy.max_requests_per_day and day.requests >= policy.max_requests_per_day:
            return QuotaDecision(
                False,
                f"Daily request limit reached ({day.requests}/{policy.max_requests_per_day}).",
                "requests_per_day",
                "blocked",
            )
        if policy.max_tokens_per_day and day.total_tokens >= policy.max_tokens_per_day:
            return QuotaDecision(
                False,
                f"Daily token limit reached ({day.total_tokens:,}/{policy.max_tokens_per_day:,}).",
                "tokens_per_day",
                "blocked",
            )
        if policy.max_tokens_per_session and session.total_tokens >= policy.max_tokens_per_session:
            return QuotaDecision(
                False,
                (
                    f"Session token limit reached ({session.total_tokens:,}/"
                    f"{policy.max_tokens_per_session:,}). Start a new session to continue."
                ),
                "tokens_per_session",
                "blocked",
            )
        if (
            policy.max_runtime_seconds_per_day
            and day.runtime_seconds >= policy.max_runtime_seconds_per_day
        ):
            return QuotaDecision(
                False,
                (
                    f"Daily active runtime limit reached "
                    f"({day.runtime_seconds / 60:.1f}/{policy.max_runtime_seconds_per_day / 60:.0f} min)."
                ),
                "runtime_per_day",
                "blocked",
            )
        if policy.max_agent_runs_per_day and day.agent_runs > policy.max_agent_runs_per_day:
            return QuotaDecision(
                False,
                f"Daily agent run limit reached ({day.agent_runs}/{policy.max_agent_runs_per_day}).",
                "runs_per_day",
                "blocked",
            )
        if policy.max_tokens_per_request and estimated_tokens > policy.max_tokens_per_request:
            return QuotaDecision(
                False,
                (
                    f"This prompt is too large: ~{estimated_tokens:,} tokens exceeds the "
                    f"per-request cap of {policy.max_tokens_per_request:,}."
                ),
                "tokens_per_request",
                "blocked",
            )

        cooldown = self._cooldown_remaining(policy)
        if cooldown > 0:
            return QuotaDecision(
                False,
                f"Rate limited: wait {cooldown:.1f}s before the next request.",
                "cooldown",
                "warning",
                retry_after_seconds=cooldown,
            )

        warning = self._warning_text(policy, day, session)
        return QuotaDecision(True, warning, "ok", "warning" if warning else "info")

    def check_tool_call(self) -> QuotaDecision:
        """Guard against runaway loops inside a single agent run."""
        policy = self.policy
        if not policy.enabled or policy.developer_mode:
            return QuotaDecision(True)
        with self._lock:
            used = self._tool_calls_this_run
        if policy.max_tool_calls_per_run and used >= policy.max_tool_calls_per_run:
            return QuotaDecision(
                False,
                f"Tool call limit for this run reached ({used}/{policy.max_tool_calls_per_run}).",
                "tool_calls_per_run",
                "blocked",
            )
        return QuotaDecision(True)

    def _cooldown_remaining(self, policy: QuotaPolicy) -> float:
        if policy.min_seconds_between_requests <= 0 or self._last_request_at is None:
            return 0.0
        elapsed = (datetime.now(timezone.utc) - self._last_request_at).total_seconds()
        remaining = policy.min_seconds_between_requests - elapsed
        return max(0.0, remaining)

    def _warning_text(
        self, policy: QuotaPolicy, day: UsageTotals, session: UsageTotals
    ) -> str:
        threshold = policy.warn_threshold
        messages: List[str] = []
        if policy.max_requests_per_day:
            ratio = day.requests / policy.max_requests_per_day
            if ratio >= threshold:
                messages.append(
                    f"{day.requests}/{policy.max_requests_per_day} daily requests used"
                )
        if policy.max_tokens_per_day:
            ratio = day.total_tokens / policy.max_tokens_per_day
            if ratio >= threshold:
                messages.append(
                    f"{day.total_tokens:,}/{policy.max_tokens_per_day:,} daily tokens used"
                )
        if policy.max_tokens_per_session:
            ratio = session.total_tokens / policy.max_tokens_per_session
            if ratio >= threshold:
                messages.append(
                    f"{session.total_tokens:,}/{policy.max_tokens_per_session:,} session tokens used"
                )
        return "Quota warning: " + ", ".join(messages) if messages else ""

    # -------------------------------------------------------------- record
    def record_request(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the cost of a completed LLM call."""
        self._last_request_at = datetime.now(timezone.utc)
        self.store.record(
            UsageEvent(
                kind="llm_request",
                session_id=self._session_id,
                model=model,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                duration_ms=int(duration_ms),
                meta=meta or {},
            )
        )

    def record_tool_call(
        self, tool: str, *, duration_ms: int = 0, ok: bool = True, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock:
            self._tool_calls_this_run += 1
        payload = {"tool": tool, "ok": ok}
        payload.update(meta or {})
        self.store.record(
            UsageEvent(
                kind="tool_call",
                session_id=self._session_id,
                duration_ms=int(duration_ms),
                meta=payload,
            )
        )

    def record_error(self, message: str, *, where: str = "") -> None:
        self.store.record(
            UsageEvent(
                kind="error",
                session_id=self._session_id,
                meta={"message": message[:2000], "where": where},
            )
        )

    # ------------------------------------------------------------ snapshot
    def snapshot(self) -> UsageSnapshot:
        """Build the full usage view for the sidebar / admin tool."""
        policy = self.policy
        day = self.store.totals_for_day()
        session = self.store.totals_for_session(self._session_id)
        gauges = [
            GaugeView("requests", "Requests today", day.requests, policy.max_requests_per_day),
            GaugeView("tokens_day", "Tokens today", day.total_tokens, policy.max_tokens_per_day),
            GaugeView(
                "tokens_session",
                "Tokens this session",
                session.total_tokens,
                policy.max_tokens_per_session,
            ),
            GaugeView(
                "runtime",
                "Active runtime today",
                round(day.runtime_seconds / 60, 1),
                round(policy.max_runtime_seconds_per_day / 60, 1)
                if policy.max_runtime_seconds_per_day
                else 0,
                "min",
            ),
            GaugeView("runs", "Agent runs today", day.agent_runs, policy.max_agent_runs_per_day),
        ]
        decision = self.check_request(0)
        snapshot = UsageSnapshot(
            day=today_key(),
            session_id=self._session_id,
            policy=policy,
            day_totals=day,
            session_totals=session,
            gauges=gauges,
            blocked=not decision.allowed and decision.code != "cooldown",
            warning=decision.reason if decision.allowed else "",
            block_reason="" if decision.allowed else decision.reason,
        )
        return snapshot

    def close(self) -> None:
        self.store.close()
