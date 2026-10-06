"""The shell-mistake guard, written against a real failing transcript.

A 9B model was handed a workspace containing ``projede Duz/snake_game.py``
and spent eleven steps failing to run it: unquoted spaces, invented directory
levels, ``mv`` on Windows, ``cd x & y`` after the ``cd`` had already failed.
Each test below is one of those steps.
"""

from __future__ import annotations

import pytest

from agent3.workspace.command_paths import (
    UNIX_ONLY_ON_WINDOWS,
    closest_path,
    explain_failure,
    looks_like_path,
    normalise,
    preflight,
    split_command,
)
from agent3.workspace.fs import WorkspaceFS


@pytest.fixture
def workspace(tmp_path):
    """A workspace whose folder name contains a space, as in the transcript."""
    root = tmp_path / "Agent3"
    (root / "projede Duz").mkdir(parents=True)
    (root / "projede Duz" / "snake_game.py").write_text("print('snake')\n")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('main')\n")
    (root / "README.md").write_text("# readme\n")
    return WorkspaceFS(root)


class TestTokenising:
    def test_quoted_paths_stay_in_one_piece(self):
        assert split_command('python "projede Duz/snake_game.py"') == [
            "python",
            '"projede Duz/snake_game.py"',
        ]

    def test_an_unquoted_space_splits_the_path(self):
        """This is the model's first mistake, reproduced exactly."""
        assert split_command("python projede Duz/snake_game.py") == [
            "python",
            "projede",
            "Duz/snake_game.py",
        ]

    def test_windows_backslashes_survive(self):
        assert split_command(r"python projede\Duz\snake.py")[1] == r"projede\Duz\snake.py"

    def test_an_unbalanced_quote_does_not_raise(self):
        assert split_command('echo "oops') == ["echo", '"oops']

    def test_backslashes_normalise_to_posix(self):
        assert normalise(r"projede\Duz\snake.py") == "projede/Duz/snake.py"
        assert normalise("./src/") == "src"

    def test_flags_and_urls_are_not_paths(self):
        assert not looks_like_path("--verbose")
        assert not looks_like_path("https://example.com/a/b")
        assert not looks_like_path("install")

    def test_real_paths_are_recognised(self):
        assert looks_like_path("src/main.py")
        assert looks_like_path(r"src\main.py")
        assert looks_like_path("README.md")


class TestClosestPath:
    CANDIDATES = ["projede Duz/snake_game.py", "projede Duz", "src/main.py", "src"]

    def test_a_missing_directory_level_is_recovered(self):
        assert closest_path("Duz/snake_game.py", self.CANDIDATES) == (
            "projede Duz/snake_game.py"
        )

    def test_a_wrong_separator_style_is_recovered(self):
        assert closest_path("projede/Duz/snake_game.py", self.CANDIDATES) == (
            "projede Duz/snake_game.py"
        )

    def test_a_typo_is_recovered(self):
        assert closest_path("src/mian.py", self.CANDIDATES) == "src/main.py"

    def test_nothing_plausible_returns_nothing(self):
        assert closest_path("totally/unrelated/thing.xyz", self.CANDIDATES) == ""

    def test_empty_inputs_are_safe(self):
        assert closest_path("", self.CANDIDATES) == ""
        assert closest_path("a.py", []) == ""


class TestPreflight:
    """Only refuse what is certainly wrong."""

    def test_mv_is_refused_on_windows(self, workspace):
        audit = preflight('mv "a.py" "b.py"', workspace, windows=True)
        assert audit.blocked
        assert "rename_file" in audit.message()

    def test_every_unix_command_names_a_replacement(self, workspace):
        for name in UNIX_ONLY_ON_WINDOWS:
            audit = preflight(f"{name} something", workspace, windows=True)
            assert audit.blocked, name
            assert audit.message().strip()

    def test_the_same_command_is_fine_on_posix(self, workspace):
        assert not preflight('mv "a.py" "b.py"', workspace, windows=False).blocked

    def test_cd_into_a_missing_directory_is_refused(self, workspace):
        audit = preflight(r"cd projede\Duz & python snake_game.py", workspace, windows=True)
        assert audit.blocked
        message = audit.message()
        assert "projede Duz" in message
        assert "cwd" in message

    def test_cd_into_a_real_directory_is_allowed(self, workspace):
        assert not preflight('cd "projede Duz"', workspace, windows=True).blocked

    def test_cd_without_an_argument_is_allowed(self, workspace):
        assert not preflight("cd", workspace, windows=True).blocked

    def test_ordinary_commands_are_never_touched(self, workspace):
        for command in (
            "pytest -q",
            "pip install tk",
            "npm run build",
            'python "projede Duz/snake_game.py"',
            "git commit -m 'wip'",
            "python -c \"import sys; print(sys.version)\"",
        ):
            assert not preflight(command, workspace, windows=True).blocked, command

    def test_an_empty_command_is_safe(self, workspace):
        assert not preflight("   ", workspace, windows=True)


class TestExplainFailure:
    """After the shell has complained, say something useful."""

    def test_the_unquoted_space_is_diagnosed(self, workspace):
        audit = explain_failure("python projede Duz/snake_game.py", workspace, windows=True)
        message = audit.message()
        assert "projede Duz/snake_game.py" in message
        assert "quoted" in message

    def test_the_invented_directory_is_diagnosed(self, workspace):
        audit = explain_failure(r"python projede\Duz\snake_game.py", workspace, windows=True)
        assert "projede Duz/snake_game.py" in audit.message()

    def test_a_correct_command_produces_no_noise(self, workspace):
        assert not explain_failure('python "projede Duz/snake_game.py"', workspace)

    def test_flags_are_never_reported(self, workspace):
        assert not explain_failure("pytest -q --maxfail=1", workspace)

    def test_a_package_name_is_not_mistaken_for_a_path(self, workspace):
        assert not explain_failure("pip install tk", workspace)

    def test_the_ampersand_trap_is_explained(self, workspace):
        audit = explain_failure('cd "projede Duz" & python snake_game.py', workspace, windows=True)
        assert "cwd" in audit.message()

    def test_each_bad_token_is_reported_once(self, workspace):
        audit = explain_failure("python a/b.py a/b.py", workspace)
        assert len(audit.suggestions) <= 1

    def test_an_unrelated_failure_stays_quiet(self, workspace):
        assert not explain_failure("python --version", workspace)
