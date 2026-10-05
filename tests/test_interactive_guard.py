"""Tests for the interactive-command trap in :mod:`agent3.workspace.terminal`.

Two independent layers are covered:

1. the *pre-flight* deny list, which refuses a command before spawning it, and
2. the *runtime* stall detector, which kills a process that printed a question
   and then went quiet.
"""

from __future__ import annotations

import sys
import time

import pytest

from agent3.workspace.terminal import (
    INTERACTIVE_COMMAND_RULES,
    PROMPT_PATTERNS,
    CommandRunner,
)

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX shell semantics are used by these fixtures"
)


@pytest.fixture()
def runner(tmp_path):
    return CommandRunner(tmp_path, interactive_idle_seconds=2.0, default_timeout=30)


class TestPreflight:
    @pytest.mark.parametrize(
        "command, expected_in_advice",
        [
            ("npm init", "-y"),
            ("yarn init", "-y"),
            ("cd app && npm init", "-y"),
            ("apt-get install curl", "-y"),
            ("pip uninstall requests", "-y"),
            ("conda install numpy", "-y"),
            ("git commit", "-m"),
            ("git rebase -i HEAD~3", "rebase"),
            ("vim setup.py", "read_file"),
            ("less README.md", "cat"),
            ("tail -f server.log", "cat"),
            ("python", "-c"),
            ("node", "-c"),
            ("ssh-keygen", "-N"),
            ("npx cowsay hi", "--yes"),
        ],
    )
    def test_interactive_commands_are_refused(self, runner, command, expected_in_advice):
        advice = runner.check_interactive(command)
        assert advice is not None, f"{command!r} should have been refused"
        assert expected_in_advice in advice

    @pytest.mark.parametrize(
        "command",
        [
            "npm init -y",
            "yarn init --yes",
            "apt-get install -y curl",
            "pip uninstall -y requests",
            'git commit -m "feat: thing"',
            "git commit --amend --no-edit",
            "pytest -q",
            "npm test",
            "npm install",
            "npm ci",
            "go test ./...",
            'python -c "print(1)"',
            "python manage.py migrate",
            "node server.js",
            "cat README.md",
            "tail -n 50 server.log",
            "git pull",
            "git merge main",
            "yes | apt-get install curl",
            "echo y | npm init",
        ],
    )
    def test_legitimate_commands_are_untouched(self, runner, command):
        assert runner.check_interactive(command) is None, f"{command!r} must not be refused"

    def test_refusal_never_starts_a_process(self, runner, tmp_path):
        marker = tmp_path / "touched.txt"
        started = time.monotonic()
        result = runner.run(f"npm init && touch {marker}")
        assert result.interactive is True
        assert result.ok is False
        assert result.exit_code == 125
        assert not marker.exists(), "the refused command must never run"
        assert time.monotonic() - started < 1.0

    def test_refusal_explains_the_fix(self, runner):
        result = runner.run("npm init")
        assert "npm init -y" in result.error
        assert "non-interactive" in result.error
        assert "nobody can type an answer" in result.error

    def test_detection_can_be_disabled(self, tmp_path):
        relaxed = CommandRunner(tmp_path, detect_interactive=False)
        assert relaxed.check_interactive("npm init") is None

    def test_every_rule_is_a_valid_pattern_with_advice(self):
        assert len(INTERACTIVE_COMMAND_RULES) >= 10
        for pattern, advice in INTERACTIVE_COMMAND_RULES:
            assert pattern.startswith("^"), f"{pattern} should be anchored"
            assert advice.strip(), f"{pattern} has no remediation advice"


class TestPromptSniffing:
    @pytest.mark.parametrize(
        "tail",
        [
            "Do you want to continue? (y/N) ",
            "Overwrite file? [Y/n]",
            "Password:",
            "Enter passphrase (empty for no passphrase):",
            "Are you sure you want to proceed",
            "Press any key to continue",
            "Select a template:",
            "? ",
        ],
    )
    def test_prompts_are_recognised(self, runner, tail):
        assert runner.looks_like_prompt(tail) is True

    @pytest.mark.parametrize(
        "tail",
        [
            "",
            "compiling module 41 of 400",
            "test_api.py::test_health PASSED",
            "Downloading package (3.2 MB)",
            "  File \"app.py\", line 12, in <module>",
            "x" * 500,
        ],
    )
    def test_normal_output_is_not_a_prompt(self, runner, tail):
        assert runner.looks_like_prompt(tail) is False

    def test_patterns_compile(self):
        import re

        for pattern in PROMPT_PATTERNS:
            re.compile(pattern)


class TestRuntimeStallDetection:
    def test_prompt_without_newline_is_killed_early(self, runner):
        """The hard case: a prompt has no trailing newline to flush it."""
        started = time.monotonic()
        result = runner.run(
            f'{sys.executable} -c "import time,sys;'
            " sys.stdout.write('Overwrite config? [y/N] ');"
            ' sys.stdout.flush(); time.sleep(25)"'
        )
        elapsed = time.monotonic() - started
        assert result.interactive is True
        assert result.ok is False
        assert result.exit_code == 125
        assert "Overwrite config?" in result.stdout
        assert "non-interactive" in result.error
        assert elapsed < 12, f"took {elapsed:.1f}s - the stall detector did not fire"

    def test_silent_long_command_is_never_mistaken_for_a_prompt(self, tmp_path):
        """A quiet compiler must be allowed to run to its timeout."""
        runner = CommandRunner(tmp_path, interactive_idle_seconds=1.0, default_timeout=5)
        result = runner.run(
            f'{sys.executable} -c "import time,sys;'
            " sys.stdout.write('building...\\n'); sys.stdout.flush(); time.sleep(30)\""
        )
        assert result.interactive is False
        assert result.timed_out is True
        assert result.exit_code == 124

    def test_fast_command_is_unaffected(self, runner):
        result = runner.run("echo hello")
        assert result.ok is True
        assert result.interactive is False
        assert result.stdout.strip() == "hello"

    def test_partial_output_is_still_captured(self, runner):
        result = runner.run("printf 'no trailing newline'")
        assert result.stdout == "no trailing newline"
        assert result.ok is True

    def test_live_output_callback_sees_every_line(self, runner):
        seen = []
        runner.run("echo one; echo two; printf three", on_output=lambda s, t: seen.append(t))
        assert "".join(seen) == "one\ntwo\nthree"

    def test_result_dict_exposes_the_flag(self, runner):
        payload = runner.run("npm init").to_dict()
        assert payload["interactive"] is True
        assert payload["ok"] is False

    def test_summary_labels_the_status(self, runner):
        assert "NEEDS INTERACTIVE INPUT" in runner.run("npm init").summary()


class TestNonInteractiveEnvironment:
    def test_environment_announces_the_absence_of_a_human(self, runner):
        result = runner.run(
            'echo "${CI} ${DEBIAN_FRONTEND} ${GIT_TERMINAL_PROMPT}'
            ' ${PIP_NO_INPUT} ${NPM_CONFIG_YES}"'
        )
        ci, frontend, git_prompt, pip_input, npm_yes = result.stdout.strip().split()
        # `setdefault` semantics: a value inherited from the host (GitHub
        # Actions exports CI=true) must win over our own default.
        assert ci.lower() in {"1", "true"}
        assert frontend == "noninteractive"
        assert git_prompt == "0"
        assert pip_input == "1"
        assert npm_yes == "true"

    def test_host_environment_is_not_clobbered(self, tmp_path):
        runner = CommandRunner(tmp_path, env_overrides={"CI": "host-value"})
        assert runner.run("echo $CI").stdout.strip() == "host-value"

    def test_editor_is_neutralised(self, runner):
        result = runner.run("echo $GIT_EDITOR $EDITOR")
        assert result.stdout.strip() == "true true"
