"""SQLite backed usage telemetry (tokens, requests, runtime, tool calls).

The database is intentionally tiny and append-only; aggregation happens in SQL
so the UI can refresh a usage panel on a timer without keeping state.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent3.core.logging_setup import get_logger
from agent3.core.paths import app_paths

logger = get_logger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc            TEXT    NOT NULL,
    day               TEXT    NOT NULL,
    session_id        TEXT    NOT NULL DEFAULT '',
    kind              TEXT    NOT NULL,
    model             TEXT    NOT NULL DEFAULT '',
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    duration_ms       INTEGER NOT NULL DEFAULT 0,
    meta              TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_usage_day      ON usage_events(day);
CREATE INDEX IF NOT EXISTS idx_usage_session  ON usage_events(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_kind     ON usage_events(kind);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def today_key(now: Optional[datetime] = None) -> str:
    """Return the UTC day bucket (``YYYY-MM-DD``) used for daily quotas."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:  # pragma: no cover - defensive
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).date().isoformat()


@dataclass
class UsageEvent:
    """One recorded billable action."""

    kind: str
    session_id: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    ts_utc: str = ""

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


@dataclass
class UsageTotals:
    """Aggregated counters for a day or a session."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    runtime_ms: int = 0
    agent_runs: int = 0
    tool_calls: int = 0
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def runtime_seconds(self) -> float:
        return self.runtime_ms / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "runtime_ms": self.runtime_ms,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "agent_runs": self.agent_runs,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
        }


class UsageStore:
    """Thread-safe SQLite wrapper for usage telemetry."""

    #: Event kinds that count as a billable LLM request.
    REQUEST_KINDS = ("llm_request",)

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else app_paths().usage_db
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=15.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    # ------------------------------------------------------------- lifecycle
    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass

    def __enter__(self) -> "UsageStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------------- write
    def record(self, event: UsageEvent) -> int:
        """Append *event* and return its row id."""
        ts = event.ts_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
        day = today_key(datetime.fromisoformat(ts)) if event.ts_utc else today_key()
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO usage_events
                    (ts_utc, day, session_id, kind, model,
                     prompt_tokens, completion_tokens, duration_ms, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    day,
                    event.session_id or "",
                    event.kind,
                    event.model or "",
                    int(event.prompt_tokens),
                    int(event.completion_tokens),
                    int(event.duration_ms),
                    json.dumps(event.meta, ensure_ascii=False),
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    # ------------------------------------------------------------------ read
    def _totals(self, where: str, params: tuple) -> UsageTotals:
        sql = f"""
            SELECT
                COALESCE(SUM(CASE WHEN kind = 'llm_request'  THEN 1 ELSE 0 END), 0) AS requests,
                COALESCE(SUM(prompt_tokens), 0)                                     AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0)                                 AS completion_tokens,
                COALESCE(SUM(duration_ms), 0)                                       AS runtime_ms,
                COALESCE(SUM(CASE WHEN kind = 'agent_run'    THEN 1 ELSE 0 END), 0) AS agent_runs,
                COALESCE(SUM(CASE WHEN kind = 'tool_call'    THEN 1 ELSE 0 END), 0) AS tool_calls,
                COALESCE(SUM(CASE WHEN kind = 'error'        THEN 1 ELSE 0 END), 0) AS errors
            FROM usage_events WHERE {where}
        """
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        if row is None:  # pragma: no cover - aggregate always returns a row
            return UsageTotals()
        return UsageTotals(
            requests=int(row["requests"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            runtime_ms=int(row["runtime_ms"]),
            agent_runs=int(row["agent_runs"]),
            tool_calls=int(row["tool_calls"]),
            errors=int(row["errors"]),
        )

    def totals_for_day(self, day: Optional[str] = None) -> UsageTotals:
        return self._totals("day = ?", (day or today_key(),))

    def totals_for_session(self, session_id: str) -> UsageTotals:
        return self._totals("session_id = ?", (session_id,))

    def totals_all_time(self) -> UsageTotals:
        return self._totals("1 = 1", ())

    def last_event_time(self, kind: str = "llm_request") -> Optional[datetime]:
        with self._lock:
            row = self._conn.execute(
                "SELECT ts_utc FROM usage_events WHERE kind = ? ORDER BY id DESC LIMIT 1",
                (kind,),
            ).fetchone()
        if not row:
            return None
        try:
            parsed = datetime.fromisoformat(str(row["ts_utc"]))
        except ValueError:  # pragma: no cover
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM usage_events ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["meta"] = json.loads(item.get("meta") or "{}")
            except json.JSONDecodeError:  # pragma: no cover
                item["meta"] = {}
            result.append(item)
        return result

    def daily_history(self, days: int = 14) -> List[Dict[str, Any]]:
        """Return per-day aggregates, newest first (for the admin tool)."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT day,
                       SUM(CASE WHEN kind='llm_request' THEN 1 ELSE 0 END) AS requests,
                       SUM(prompt_tokens + completion_tokens)              AS tokens,
                       SUM(duration_ms)                                    AS runtime_ms
                FROM usage_events
                GROUP BY day ORDER BY day DESC LIMIT ?
                """,
                (int(days),),
            ).fetchall()
        return [
            {
                "day": row["day"],
                "requests": int(row["requests"] or 0),
                "tokens": int(row["tokens"] or 0),
                "runtime_ms": int(row["runtime_ms"] or 0),
            }
            for row in rows
        ]

    # ----------------------------------------------------------------- admin
    def reset_day(self, day: Optional[str] = None) -> int:
        """Delete every event of *day* (defaults to today). Returns row count."""
        target = day or today_key()
        with self._lock:
            cursor = self._conn.execute("DELETE FROM usage_events WHERE day = ?", (target,))
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def reset_session(self, session_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM usage_events WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def reset_all(self) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM usage_events")
            self._conn.execute("DELETE FROM sqlite_sequence WHERE name='usage_events'")
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def purge_older_than(self, days: int) -> int:
        """Housekeeping: drop events older than *days* days."""
        cutoff = date.fromordinal(date.today().toordinal() - max(0, int(days))).isoformat()
        with self._lock:
            cursor = self._conn.execute("DELETE FROM usage_events WHERE day < ?", (cutoff,))
            self._conn.commit()
            return int(cursor.rowcount or 0)
