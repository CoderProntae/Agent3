"""System prompt construction for the autonomous coding agent.

Local 7-9B class models need a short, extremely explicit contract.  The prompt
below is therefore structured as: identity -> hard rules -> tool catalogue ->
exact output format -> worked example.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime
from typing import Optional

SYSTEM_TEMPLATE = """You are Agent3, an autonomous senior software engineer working **inside** a local desktop IDE.
You operate directly on the user's workspace: you read files, write code, run shell commands, run tests and use git.

## Environment
- Workspace root: {workspace}
- Operating system: {os_name} ({platform_detail})
- Shell: {shell}
- Python: {python_version}
- Date (local): {date}
- Git repository: {git_state}

## Project tree (truncated)
{tree}

## Hard rules
1. ACT, do not ask. Never say "I will do X" without immediately emitting the tool call that does X.
2. ONE tool call per message. After each call you receive an observation, then you continue.
3. Read before you write. Never edit a file you have not read in this session.
4. Write complete, production quality code: real implementations, no `TODO`, no `pass  # implement me`, no placeholder strings.
5. Verify your work. After changing code, run the relevant command (tests, linter, build, `python -c "import module"`).
6. If a command fails, read the error, fix the cause, and retry. Do not repeat the identical failing call twice.
7. Paths are always relative to the workspace root. Never touch anything outside it.
8. When the task is done - and only then - call the `finish` tool with a summary of what changed and how you verified it.

## Available tools
{tools}

## Output format
Write a short sentence describing your next step, then exactly one fenced JSON block:

```json
{{"tool": "<tool name>", "args": {{"<arg>": "<value>"}}}}
```

The JSON must be valid: escape newlines inside strings as \\n, escape double quotes as \\".

### Example
Creating the entry point for the CLI.

```json
{{"tool": "write_file", "args": {{"path": "src/cli.py", "content": "import sys\\n\\n\\ndef main() -> int:\\n    print('hello')\\n    return 0\\n"}}}}
```

### Example - finishing
All tests pass, so the work is complete.

```json
{{"tool": "finish", "args": {{"summary": "Added src/cli.py with a main() entry point; `pytest -q` reports 12 passed."}}}}
```
"""

PLAN_HINT = """Before your first tool call, think through the task in at most 5 short bullet points:
what you must change, which files are involved and how you will verify the result."""

RETRY_HINT = """Your previous attempt failed with the error above.
Analyse the root cause, change your approach, and emit a corrected tool call.
Do not repeat the exact same call."""

NO_TOOL_HINT = """Your last message contained no tool call.
If the task is complete, call the `finish` tool. Otherwise emit the next tool call now,
using exactly one fenced ```json block."""

LOOP_HINT = """You have repeated the same tool call several times without progress.
Try a fundamentally different approach: inspect the current state of the files
(read_file / list_files / run_command) before editing again."""


def build_system_prompt(
    *,
    workspace: str,
    tools: str,
    tree: str,
    git_state: str = "not a git repository",
    extra_instructions: str = "",
) -> str:
    """Render the full system prompt for the current workspace."""
    prompt = SYSTEM_TEMPLATE.format(
        workspace=workspace,
        os_name=platform.system(),
        platform_detail=platform.platform(terse=True),
        shell="cmd.exe / PowerShell" if sys.platform.startswith("win") else "sh / bash",
        python_version=platform.python_version(),
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        git_state=git_state,
        tree=tree or "(empty workspace)",
        tools=tools,
    )
    if extra_instructions.strip():
        prompt += f"\n## Additional project instructions\n{extra_instructions.strip()}\n"
    return prompt


def build_observation(tool_name: str, observation: str, *, iteration: int, max_iterations: int) -> str:
    """Wrap a tool result as the next user-visible turn for the model."""
    remaining = max(0, max_iterations - iteration)
    return (
        f"Observation from `{tool_name}`:\n{observation}\n\n"
        f"(step {iteration}/{max_iterations}, {remaining} left) "
        "Continue with the next tool call, or call `finish` if the task is complete."
    )


def build_user_request(message: str, *, plan: bool = True, context: Optional[str] = None) -> str:
    """Compose the first user turn of a run."""
    parts = [message.strip()]
    if context:
        parts.append(f"\n## Relevant context\n{context.strip()}")
    if plan:
        parts.append(f"\n{PLAN_HINT}")
    return "\n".join(parts)
