"""Tests for paths, logging, the secure store and configuration."""

from __future__ import annotations

import json

import pytest

from agent3.core.config import DEFAULT_OLLAMA_PORT, AppConfig, ConfigManager
from agent3.core.paths import app_paths
from agent3.core.secure_store import SecureStore, SecureStoreError, machine_default_passphrase


class TestAppPaths:
    def test_respects_agent3_home(self, isolated_home):
        assert app_paths().base == isolated_home.resolve()

    def test_creates_directories(self):
        paths = app_paths()
        assert paths.base.is_dir()
        assert paths.log_dir.is_dir()
        assert paths.crash_dir.is_dir()

    def test_file_locations_are_inside_base(self):
        paths = app_paths()
        for target in (paths.config_file, paths.credentials_file, paths.sessions_db):
            assert paths.base in target.parents


class TestSecureStore:
    def test_round_trip(self, tmp_path):
        store = SecureStore(tmp_path / "secret.enc", passphrase="hunter2")
        store.save({"token": "abc", "limit": 42})
        assert store.load() == {"token": "abc", "limit": 42}

    def test_missing_file_returns_default(self, tmp_path):
        store = SecureStore(tmp_path / "nope.enc")
        assert store.load() == {}
        assert store.load({"a": 1}) == {"a": 1}

    def test_payload_is_encrypted_on_disk(self, tmp_path):
        path = tmp_path / "secret.enc"
        SecureStore(path, passphrase="pw").save({"token": "super-secret-value"})
        raw = path.read_text(encoding="utf-8")
        assert "super-secret-value" not in raw
        envelope = json.loads(raw)
        assert envelope["v"] == 1 and envelope["kdf"] == "pbkdf2-sha256"

    def test_wrong_passphrase_raises(self, tmp_path):
        path = tmp_path / "secret.enc"
        SecureStore(path, passphrase="right").save({"a": 1})
        with pytest.raises(SecureStoreError):
            SecureStore(path, passphrase="wrong").load()

    def test_tampering_is_detected(self, tmp_path):
        path = tmp_path / "secret.enc"
        store = SecureStore(path, passphrase="pw")
        store.save({"limit": 10})
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["ct"] = envelope["ct"][:-4] + "AAAA"
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(SecureStoreError):
            store.load()

    def test_rekey_preserves_payload(self, tmp_path):
        path = tmp_path / "secret.enc"
        store = SecureStore(path, passphrase="old")
        store.save({"value": 7})
        store.rekey("new")
        assert SecureStore(path, passphrase="new").load() == {"value": 7}
        with pytest.raises(SecureStoreError):
            SecureStore(path, passphrase="old").load()

    def test_machine_passphrase_is_stable(self):
        assert machine_default_passphrase() == machine_default_passphrase()
        assert len(machine_default_passphrase()) == 64


class TestConfig:
    def test_defaults_target_port_11435(self):
        config = AppConfig()
        assert config.ollama.port == DEFAULT_OLLAMA_PORT == 11435
        assert config.ollama.base_url == "http://localhost:11435"
        assert config.ollama.model == "qwen3.5-9b-abliterated"

    def test_endpoints_are_deduplicated_and_ordered(self):
        config = AppConfig()
        endpoints = config.ollama.endpoints()
        assert endpoints[0] == "http://localhost:11435"
        assert len(endpoints) == len(set(endpoints))

    def test_save_and_reload(self):
        manager = ConfigManager()
        manager.config.ollama.port = 12345
        manager.config.agent.max_iterations = 7
        manager.save()
        assert ConfigManager().config.ollama.port == 12345
        assert ConfigManager().config.agent.max_iterations == 7

    def test_corrupt_config_falls_back_to_defaults(self):
        manager = ConfigManager()
        manager.path.write_text("{not json", encoding="utf-8")
        assert ConfigManager().config.ollama.port == 11435

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AGENT3_OLLAMA_PORT", "9999")
        monkeypatch.setenv("AGENT3_OLLAMA_MODEL", "testmodel")
        config = ConfigManager().config
        assert config.ollama.port == 9999
        assert config.ollama.model == "testmodel"

    def test_secret_round_trip(self):
        manager = ConfigManager()
        manager.github_token = "ghp_example"
        assert ConfigManager().github_token == "ghp_example"
        manager.github_token = ""
        assert ConfigManager().github_token == ""

    def test_sampling_options(self):
        options = AppConfig().ollama.options()
        assert options["temperature"] == 0.2
        assert options["num_ctx"] == 16384
