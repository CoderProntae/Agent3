<div align="center">

# Agent3

**A local, autonomous AI coding agent and workspace — as a native desktop application.**

Chat · File explorer · Code editor with live diffs · Embedded terminal · Git · Background processes
Powered entirely by a **local Ollama** server on `http://localhost:11435`. No cloud, no telemetry.

</div>

---

## What it is

Agent3 is a desktop IDE-shell (PySide6 / Qt 6) in which an autonomous agent works **inside a folder you
choose**. It reads your files, writes code, runs shell commands, executes your tests, inspects the
failure output, fixes itself, and commits with git.

CI produces a single portable executable, `Agent3.exe`: no installer, no runtime to deploy, and all
of its state in `%APPDATA%\Agent3`.

---

## Quick start (from a GitHub build)

1. **Actions → Build & Release → Run workflow** (tick *create_release* if you want a release draft),
   or push a tag such as `v1.0.0`.
2. Download the artifact **`Agent3-windows-x64-bundle`** and unzip it.
3. Install [Ollama](https://ollama.com) and serve it on **port 11435**:

   ```bat
   set OLLAMA_HOST=127.0.0.1:11435
   ollama serve
   ```

4. Pull a model (the default tag is `qwen3.5-9b-abliterated`; any local model works):

   ```bat
   ollama pull qwen2.5-coder:7b
   ```

5. Run `Agent3.exe` → `Ctrl+O` to mount a workspace folder → describe your task → `Ctrl+Enter`.

> The toolbar shows a live connection badge. If it is red, open **File → Settings → Connection**
> and press *Test connection*; Agent3 also falls back to `127.0.0.1:11435` and `localhost:11434`.

### Running from source

```bash
git clone https://github.com/CoderProntae/Agent3
cd Agent3
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m agent3                 # main application
python -m agent3 --self-test     # verify the install without a model
pytest -q                        # 530+ tests
```

---

## Architecture

```
src/
├── agent3/
│   ├── app.py                  bootstrap: CLI flags, logging, theme, main window
│   ├── core/                   paths · rotating logs · AES-GCM secure store · typed config
│   ├── llm/                    Ollama REST client (streaming, retries) · token estimator
│   ├── workspace/              sandboxed fs · diffing · shell runner · git wrapper
│   ├── agent/                  tool registry · prompt contract · autonomous loop · sessions
│   └── ui/                     dark theme · syntax highlighting · widgets · QThread workers
```

Every layer below `ui/` is pure Python and unit-tested without a Qt event loop or a live model.

### The agent loop

```
user instruction
      │
      ▼
 build system prompt  ──────────────────────────────►  Ollama /api/chat (streamed)
      ▲                                                        │
      │                                              parse the JSON tool call
   observation                                                 │
      │                                                        ▼
      └────────────  execute tool (fs / shell / git)  ◄─────────┘
                              │
                      error?  └─► stack trace is fed back + "fix your mistake" hint
                              │
                      finish  └─► summary shown in the chat
```

* **One tool call per turn**, parsed from a fenced ` ```json ` block. The parser also accepts bare
  JSON, OpenAI-style `function.arguments` and `{"tool_calls": [...]}` wrappers, because small local
  models are inconsistent.
* **Self-correction**: failed tool calls and `OllamaError`s are re-injected with a retry hint,
  bounded by `self_correction_retries`; repeating the same call three times triggers a
  "try a different approach" nudge; `max_iterations` is the hard stop.
* **Cancellation** is cooperative: *Stop* sets an event that aborts the HTTP stream and kills the
  running child process tree.

### Agent tools

| Tool | What it does |
|---|---|
| `project_overview` | File tree, file-type statistics and git state in one shot |
| `list_files` / `read_file` | Browse and read (optionally a line range) |
| `view_outline` | Structural map of a file - classes, functions, signatures, docstrings, line numbers - without spending the context on its body. Python via `ast`, 15 other languages via scanners |
| `write_file` | Create or fully overwrite a file (returns a unified diff) |
| `edit_file` | Anchored search/replace with a whitespace-tolerant fallback |
| `patch_file` | Apply a multi-hunk unified diff atomically. Hunks are located by context, so later hunks still land after earlier ones shift the line numbers |
| `delete_file` / `rename_file` / `make_directory` | Filesystem mutations |
| `search_code` | Literal or regex grep across the workspace |
| `run_command` | Shell execution with capture, timeout, a destructive-command deny-list and an interactive-command trap |
| `git` | `status · init · add · commit · diff · log · branch · checkout · push · pull` |
| `start_process` | Launch a dev server / watcher in the **background** and keep working; returns a `process_id` |
| `get_process_logs` | Tail a background process's captured output, or list every process and its state |
| `stop_process` | Terminate a background process tree (`process_id="all"` stops everything) |
| `manage_tasks` | The agent's visible plan: `add · update · list · set · remove · clear`, statuses `pending / in_progress / completed / cancelled` (alias `todo_list`) |
| `check_syntax` | Parse and lint a file or a snippet without executing it |
| `undo_file_change` | Roll a file back to the state it had before the last tool touched it (alias `rollback_file`) |
| `finish` | Ends the run - refused while code is broken, the plan is unfinished, or the changes are unverified |

### Reasoning control

Reasoning support is detected from **evidence, never from the model's name**.
A model called `gpt-oss` with no capability reported gets no controls, and an
obscure community repack gets full effort levels if its template really has
them. `/api/show` is consulted once per model and read in this order:

| # | Source | What it proves |
|---|---|---|
| 1 | `thinking: {"values": [...], "default": ...}` | Authoritative - the server honours the `think` field itself |
| 2 | The model's **chat template** | `enable_thinking` means on/off; `reasoning_effort` validated against a literal tuple gives the exact level names |
| 3 | `capabilities` contains `"thinking"` | On/off only, no levels |
| 4 | Nothing | The model does not reason |

The strip under the message box shows the **levels**, not the source: a
dropdown of the values this model accepts, live even while the switch is off
(picking an effort turns reasoning on). Where the information came from is in
the tooltip, not in the label.

Step 2 is what makes GGUF repacks work. A Qwen3.x template contains

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
```

so Agent3 offers exactly `xhigh`, `medium` and `low` for that file - whatever
the model is called - and marks `xhigh` as its default.

#### Making the setting stick

Ollama does **not** forward the `think` field into every chat template. A
template that begins `{%- if enable_thinking is undefined or enable_thinking
is true %}` therefore treats "no opinion" as "reason at full effort", which is
why models keep thinking after the switch is turned off. Two defences run
together whenever the server did not report native `thinking` metadata:

* **The instruction is repeated in the prompt.** For a level, Agent3 injects
  the template's *own* sentence (`"Reasoning effort is set to low. Keep your
  thinking brief and focused…"`) in front of the first system message - the
  same place the template would have rendered it. For "off", the documented
  `/no_think` soft switch is appended to the last user turn and an explicit
  instruction is added. Nothing is invented: the sentences are lifted out of
  the template when it carries them.
* **Inline traces are filtered out.** When a model reasons anyway and the
  server returns the trace inside `message.content`, a streaming state machine
  strips `<think>`, `<thinking>`, `<reasoning>` and `◁think▷` blocks out of
  the answer and routes them to the reasoning channel - tolerating tags split
  across chunk boundaries. The tool-call parser therefore never sees the
  scratchpad, and the status bar says once that the model is ignoring the
  switch.

Both behaviours can be turned off in **Settings → Transport**
(`strip_inline_reasoning`, `enforce_think_in_prompt`). A value the model does
not accept is still dropped before the request is sent, instead of making the
server reject the whole call, and the trace is always shown in its own
collapsed block in the transcript.

### Prompt cache and latency

llama.cpp reuses its KV cache only for the longest common **prefix** of two
consecutive prompts. Agent3 therefore treats prompt layout as a performance
contract:

* the invariant ~2700 tokens (identity, rules, tool catalogue, output format)
  come first, and the workspace snapshot - tree, git state, date - goes last;
* the snapshot is rendered **once per run** and held steady between steps, so
  a step that created a file does not rewrite message 0;
* the environment line carries the date only, never a clock that would tick
  the cache away on its own.

Without this the server logs `forcing full prompt re-processing due to lack
of cache data` and spends 15-20 s re-evaluating the conversation before the
first token of every step - which is indistinguishable from a frozen UI.
Measured on the real tool catalogue, the reusable prefix goes from **48** to
**2726** tokens.

While a hidden reasoning trace is streaming, the status bar reports the
character count, so a model that thinks for a minute before answering never
looks like a hang.

### Long-running processes

`run_command` waits for the command to exit - which a dev server never does.
`start_process` spawns the command in its own session/process group, streams
stdout and stderr into a bounded ring buffer, and hands back a `process_id`.
The same deny-list and non-interactive environment as `run_command` apply, the
terminal panel shows a `● N background` badge, and every process is killed when
the window closes. No orphan servers.

### The plan

For anything with more than two steps the system prompt requires the model to
write its plan down with `manage_tasks` before touching code, and to flip each
item as it goes. The list is mirrored live into the sidebar, and `finish` is
refused once while items are still open - a run that quietly abandons half the
request is the failure mode this closes.

### Checked on write

Every file the agent writes is parsed immediately: Python through `ast`, JSON,
TOML, YAML, XML and INI through the standard library, JavaScript through
`node --check`, TypeScript through `tsc --noEmit`, plus `eslint`/`ruff` when
they are installed and configured. The verdict is appended to the tool result,
so a syntax error is visible in the step that caused it rather than three calls
later. A broken file turns the tool call into a failure and blocks `finish`.
External linters are strictly optional: a missing binary is reported as
"skipped", never as an error.

### Undo

Every mutating tool snapshots the file's previous content in memory before it
writes. `undo_file_change` restores the newest snapshot - deleting the file
again if it did not exist before - so a bad edit is one call away from being
reverted instead of being patched over by hand. The buffer is bounded by both
entry count and total bytes.

### Cheap exploration

`read_file` on a 900-line module costs thousands of context tokens to answer
"which methods does this class have?". `view_outline` answers the same question
in a few dozen lines, and the model then reads only the range it needs. The
system prompt requires it for anything over ~150 lines.

### Editing in one shot

Three changes in one file used to mean three `edit_file` calls, with the model
guessing the file's state between each one - the classic drift failure of small
local models. `patch_file` takes a real unified diff and applies every hunk in a
single atomic step; a hunk that cannot be placed is reported by number, with the
context it expected, and nothing is written.

### No command may wait for a human

Nobody can answer a prompt inside an autonomous agent, so a command that asks a
question would simply burn the 240 s timeout. Two layers prevent that:

1. **Pre-flight deny list** - `npm init`, `apt-get install`, `git commit`
   without `-m`, `vim`, `less`, a bare `python` REPL and ~15 more patterns are
   refused *before* the process is spawned, and the model is told the
   non-interactive form to use instead (`npm init -y`, ...).
2. **Runtime stall detector** - output is read as raw chunks rather than lines,
   so a prompt with no trailing newline (`Continue? [y/N] `) is still visible.
   If output stops for 15 s *and* the tail looks like a question, the process
   tree is killed and the model gets the remediation advice. A silent compiler
   is never mistaken for a prompt.

The child environment also advertises that no human is present:
`CI=1`, `DEBIAN_FRONTEND=noninteractive`, `GIT_TERMINAL_PROMPT=0`,
`GIT_EDITOR=true`, `PIP_NO_INPUT=1`, `NPM_CONFIG_YES=true`.

### Definition of done

`finish` is not a free action. If files changed during the run and no command
has succeeded since the last edit, the loop **refuses the call once** and pushes
the model back to its test command (`pytest -q`, `npm test`, `go test ./...`, a
build or an import check). A failing verification produces a different,
sharper hint. A stubborn model is never deadlocked: the second attempt is
accepted but the summary is stamped with a visible warning, and
`AgentRunResult` exposes `verified`, `verification_command` and `changed_files`.
Set `agent.require_verification = false` in `config.json` to opt out.

### Safety model

* Every path the model emits goes through `WorkspaceFS.resolve()`, which rejects `..` traversal,
  absolute paths outside the root and symlink escapes — **nothing outside the mounted folder is reachable**.
* `CommandRunner` refuses `rm -rf /`, `mkfs`, `format C:`, fork bombs, `curl … | sh`, and friends
  (the pattern list is user-editable in the config), confines the working directory to the workspace,
  enforces timeouts and kills the whole process group on stop.
* Secrets (the GitHub token) are stored **AES-256-GCM encrypted** with a PBKDF2-derived key;
  tampering with the file makes decryption fail loudly instead of silently returning a default.

---

## Path and shell awareness

Small local models fail at the shell in a narrow, repetitive way: an unquoted folder name with a
space, an invented directory level, `mv` on Windows, `cd x & y` that keeps going after the `cd`
already failed. `agent3/workspace/command_paths.py` turns each of those into a sentence the model
can act on:

| Situation | What happens |
|---|---|
| `mv`, `cp`, `rm`, `ls`, `cat`, `touch`, `grep`, `sed` on Windows | Refused **before execution**, with the tool that replaces it (`rename_file`, `delete_file`, …) |
| `cd into-a-missing-folder & …` | Refused before execution; the real folder is named and the `cwd` argument is shown as JSON |
| A command fails and mentions a path that is not there | The closest real workspace path is appended to the error — plus a quoting reminder when it contains a space |

Missing paths are only reported *after* a failure, never before: `mkdir`, `git clone` and compiler
outputs are supposed to name paths that do not exist yet.

---

## User interface

| Region | Contents |
|---|---|
| **Toolbar** | Workspace path · model picker (populated from `/api/tags`) · live connection badge |
| **Left sidebar** | File explorer (lazy tree, context menu) · agent plan · session switcher |
| **Centre** | Chat with markdown + syntax-highlighted code, and live **action cards** (`[AGENT] write_file … ✓ 12 ms`) with expandable output |
| **Right** | Tabbed editor (line numbers, highlighting, dirty markers) + inline / side-by-side **diff viewer** |
| **Bottom** | Embedded terminal: live agent output *and* your own commands with history |
| **Status bar** | Agent state · tokens used by the current run |

Shortcuts: `Ctrl+O` open workspace · `Ctrl+Enter` run · `Esc` stop · `Ctrl+S` save · `Ctrl+N` new
session · `Ctrl+,` settings · ``Ctrl+` `` toggle terminal.

---

## Configuration

Everything lives in one per-user folder (override with the `AGENT3_HOME` environment variable):

| OS | Location |
|---|---|
| Windows | `%APPDATA%\Agent3` |
| macOS | `~/Library/Application Support/Agent3` |
| Linux | `~/.config/agent3` |

```
config.json          non-sensitive settings (endpoint, model, agent behaviour, UI state)
credentials.enc      AES-256-GCM encrypted GitHub token
sessions.sqlite3     chat history
logs/agent3.log      rotating log (5 × 2 MiB)
crashes/             unhandled-exception dumps
```

Useful CLI flags and environment variables:

```bash
python -m agent3 /path/to/project --model qwen2.5-coder:7b --port 11435 --debug
AGENT3_OLLAMA_HOST=127.0.0.1 AGENT3_OLLAMA_PORT=11435 AGENT3_OLLAMA_MODEL=... python -m agent3
```

---

## CI/CD

`.github/workflows/build.yml` runs on every push, pull request, tag `v*` and manual dispatch:

1. **test** (ubuntu) — installs the Qt runtime libraries, runs the full pytest suite headless
   (`QT_QPA_PLATFORM=offscreen`) with coverage, then byte-compiles every module.
2. **build-windows** — freezes `Agent3.exe` with PyInstaller, validates that the binary exists, is
   of a plausible size and answers `--version`, then uploads two artifacts (the `.exe` itself plus
   `Agent3-windows-x64.zip` with `INSTALL.txt`, `README.md`, `LICENSE`).
3. **release** — on a `v*` tag or `create_release: true`, publishes a **draft** GitHub Release with
   the artifacts attached.

Build locally with:

```bash
python packaging/build.py --clean          # freeze Agent3.exe
```

---

## Development

```bash
pytest -q                                   # whole suite
pytest -q tests/test_agent.py               # the agent loop
QT_QPA_PLATFORM=offscreen pytest -q -m gui  # Qt widgets only
```

The suite covers the sandbox escape attempts, the deny-listed commands, the path/shell guard, the
streaming/retry/fallback behaviour of the Ollama client (with a scripted fake transport), the tool
protocol parser, the self-correcting loop (with a scripted fake model) and a headless smoke test of
every widget — no Ollama server and no display required.

Turkish documentation: [`docs/TR-KULLANIM.md`](docs/TR-KULLANIM.md).

## License

MIT — see [LICENSE](LICENSE).
