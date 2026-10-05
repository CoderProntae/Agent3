"""Tests for the quota policy, telemetry store and rate limiting engine."""

from __future__ import annotations

import time

from agent3.limits.manager import UsageManager
from agent3.limits.policy import PolicyStore, QuotaPolicy
from agent3.limits.store import UsageEvent, UsageStore, today_key


class TestQuotaPolicy:
    def test_defaults_match_the_specification(self):
        policy = QuotaPolicy()
        assert policy.max_requests_per_day == 500
        assert policy.max_tokens_per_session == 100_000
        assert policy.enabled is True
        assert policy.developer_mode is False

    def test_normalisation_clamps_bad_values(self):
        policy = QuotaPolicy(
            max_requests_per_day=-5, warn_threshold=9.0, min_seconds_between_requests=-1
        ).normalised()
        assert policy.max_requests_per_day == 0
        assert policy.warn_threshold == 1.0
        assert policy.min_seconds_between_requests == 0.0

    def test_password_hashing(self):
        policy = QuotaPolicy()
        assert policy.verify_admin_password("anything") is True  # no password set
        policy.set_admin_password("s3cret")
        assert policy.locked is True
        assert policy.admin_password_hash and "s3cret" not in policy.admin_password_hash
        assert policy.verify_admin_password("s3cret") is True
        assert policy.verify_admin_password("wrong") is False
        policy.set_admin_password("")
        assert policy.locked is False

    def test_round_trip_through_dict(self):
        original = QuotaPolicy(max_tokens_per_day=123, notes="hello")
        restored = QuotaPolicy.from_dict(original.to_dict())
        assert restored.max_tokens_per_day == 123
        assert restored.notes == "hello"


class TestPolicyStore:
    def test_creates_default_policy_file(self):
        store = PolicyStore()
        policy = store.load()
        assert store.exists()
        assert policy.max_requests_per_day == 500

    def test_save_and_hot_reload(self):
        store = PolicyStore()
        store.load()
        updated = QuotaPolicy(max_requests_per_day=42)
        store.save(updated, updated_by="test")
        time.sleep(0.01)
        other = PolicyStore()
        assert other.load().max_requests_per_day == 42
        assert other.load().updated_by == "test"

    def test_tampered_file_falls_back_to_safe_defaults(self):
        store = PolicyStore()
        store.save(QuotaPolicy(max_requests_per_day=999_999))
        store.path.write_text('{"v": 1, "ct": "garbage", "salt": "AAAA", "nonce": "AAAA"}', encoding="utf-8")
        fresh = PolicyStore().load()
        assert fresh.max_requests_per_day == 500  # conservative default, not 999999

    def test_reset_keeps_password(self):
        store = PolicyStore()
        policy = store.load()
        policy.set_admin_password("admin")
        store.save(policy)
        reset = store.reset_to_defaults(keep_password=True)
        assert reset.locked is True
        assert reset.verify_admin_password("admin") is True
        assert reset.max_requests_per_day == 500


class TestUsageStore:
    def test_record_and_aggregate(self):
        store = UsageStore()
        store.record(UsageEvent(kind="llm_request", session_id="s1", prompt_tokens=100, completion_tokens=50, duration_ms=1200))
        store.record(UsageEvent(kind="llm_request", session_id="s1", prompt_tokens=10, completion_tokens=5))
        store.record(UsageEvent(kind="tool_call", session_id="s1"))
        totals = store.totals_for_day()
        assert totals.requests == 2
        assert totals.total_tokens == 165
        assert totals.tool_calls == 1
        assert totals.runtime_ms == 1200
        store.close()

    def test_session_isolation(self):
        store = UsageStore()
        store.record(UsageEvent(kind="llm_request", session_id="a", prompt_tokens=10))
        store.record(UsageEvent(kind="llm_request", session_id="b", prompt_tokens=20))
        assert store.totals_for_session("a").total_tokens == 10
        assert store.totals_for_session("b").total_tokens == 20
        assert store.totals_for_day().total_tokens == 30
        store.close()

    def test_reset_operations(self):
        store = UsageStore()
        for _ in range(3):
            store.record(UsageEvent(kind="llm_request", session_id="x", prompt_tokens=1))
        assert store.reset_day(today_key()) == 3
        assert store.totals_for_day().requests == 0
        store.close()

    def test_daily_history(self):
        store = UsageStore()
        store.record(UsageEvent(kind="llm_request", prompt_tokens=5, completion_tokens=5))
        history = store.daily_history(5)
        assert history and history[0]["day"] == today_key()
        assert history[0]["tokens"] == 10
        store.close()


class TestUsageManager:
    def test_allows_by_default(self, usage_manager):
        assert usage_manager.check_request(100).allowed is True

    def test_blocks_on_daily_request_limit(self, usage_manager):
        policy = usage_manager.policies.load()
        policy.max_requests_per_day = 2
        usage_manager.policies.save(policy)
        for _ in range(2):
            usage_manager.record_request(model="m", prompt_tokens=1, completion_tokens=1, duration_ms=1)
        decision = usage_manager.check_request(0)
        assert decision.allowed is False
        assert decision.code == "requests_per_day"
        assert "Daily request limit" in decision.reason

    def test_blocks_on_session_tokens(self, usage_manager):
        policy = usage_manager.policies.load()
        policy.max_tokens_per_session = 100
        usage_manager.policies.save(policy)
        usage_manager.record_request(model="m", prompt_tokens=80, completion_tokens=40, duration_ms=1)
        decision = usage_manager.check_request(0)
        assert decision.allowed is False
        assert decision.code == "tokens_per_session"

    def test_new_session_resets_session_budget(self, usage_manager):
        policy = usage_manager.policies.load()
        policy.max_tokens_per_session = 100
        usage_manager.policies.save(policy)
        usage_manager.record_request(model="m", prompt_tokens=200, completion_tokens=0, duration_ms=1)
        assert usage_manager.check_request(0).allowed is False
        usage_manager.start_session()
        assert usage_manager.check_request(0).allowed is True

    def test_blocks_oversized_prompt(self, usage_manager):
        policy = usage_manager.policies.load()
        policy.max_tokens_per_request = 1000
        usage_manager.policies.save(policy)
        assert usage_manager.check_request(5000).allowed is False

    def test_developer_mode_bypasses_everything(self, usage_manager):
        policy = usage_manager.policies.load()
        policy.max_requests_per_day = 1
        policy.developer_mode = True
        usage_manager.policies.save(policy)
        for _ in range(5):
            usage_manager.record_request(model="m", prompt_tokens=10_000, completion_tokens=0, duration_ms=1)
        assert usage_manager.check_request(10**9).allowed is True

    def test_tool_call_limit_per_run(self, usage_manager):
        policy = usage_manager.policies.load()
        policy.max_tool_calls_per_run = 2
        usage_manager.policies.save(policy)
        usage_manager.begin_run()
        for _ in range(2):
            assert usage_manager.check_tool_call().allowed is True
            usage_manager.record_tool_call("read_file")
        assert usage_manager.check_tool_call().allowed is False
        usage_manager.begin_run()
        assert usage_manager.check_tool_call().allowed is True

    def test_cooldown(self, usage_manager):
        policy = usage_manager.policies.load()
        policy.min_seconds_between_requests = 30
        usage_manager.policies.save(policy)
        usage_manager.record_request(model="m", prompt_tokens=1, completion_tokens=1, duration_ms=1)
        decision = usage_manager.check_request(0)
        assert decision.allowed is False
        assert decision.code == "cooldown"
        assert 0 < decision.retry_after_seconds <= 30

    def test_warning_near_threshold(self, usage_manager):
        policy = usage_manager.policies.load()
        policy.max_requests_per_day = 10
        policy.warn_threshold = 0.5
        usage_manager.policies.save(policy)
        for _ in range(6):
            usage_manager.record_request(model="m", prompt_tokens=1, completion_tokens=1, duration_ms=1)
        decision = usage_manager.check_request(0)
        assert decision.allowed is True
        assert "Quota warning" in decision.reason

    def test_snapshot_shape(self, usage_manager):
        usage_manager.record_request(model="m", prompt_tokens=5, completion_tokens=5, duration_ms=100)
        snapshot = usage_manager.snapshot()
        keys = {gauge.key for gauge in snapshot.gauges}
        assert keys == {"requests", "tokens_day", "tokens_session", "runtime", "runs"}
        assert snapshot.day_totals.total_tokens == 10
        assert snapshot.blocked is False
