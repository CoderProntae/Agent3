<div align="center">

# Agent3

**A local, autonomous AI coding agent and workspace — as a native desktop application.**

Chat · File explorer · Code editor with live diffs · Embedded terminal · Git · Enterprise usage quotas
Powered entirely by a **local Ollama** server on `http://localhost:11435`. No cloud, no telemetry.

</div>

---

## What it is

Agent3 is a desktop IDE-shell (PySide6 / Qt 6) in which an autonomous agent works **inside a folder you
choose**. It reads your files, writes code, runs shell commands, executes your tests, inspects the
failure output, fixes itself, and commits with git — while an administrator-controlled quota engine
tracks every token, request and second of runtime.

Two executables are produced by CI:

| Executable | Purpose |
|---|---|
| `Agent3.exe` | The main workspace application |
| `UsageLimitEditor.exe` | Standalone administrator tool for quotas, token limits and developer-mode override |

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
python -m usage_limit_editor     # administrator tool
pytest -q                        # 160+ tests
```

---

## Architecture

```
src/
├── agent3/
│   ├── app.py                  bootstrap: CLI flags, logging, theme, main window
│   ├── core/                   paths · rotating logs · AES-GCM secure store · typed config
│   ├── llm/                    Ollama REST client (streaming, retries) · token estimator
│   ├── limits/                 quota policy · SQLite telemetry · rate-limiting engine
│   ├── workspace/              sandboxed fs · diffing · shell runner · git wrapper
│   ├── agent/                  tool registry · prompt contract · autonomous loop · sessions
│   └── ui/                     dark theme · syntax highlighting · widgets · QThread workers
└── usage_limit_editor/         standalone administrator GUI
```

Every layer below `ui/` is pure Python and unit-tested without a Qt event loop or a live model.

### The agent loop

```
user instruction
      │
      ▼
 build system prompt  ──►  quota pre-authorisation  ──►  Ollama /api/chat (streamed)
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
| `write_file` | Create or fully overwrite a file (returns a unified diff) |
| `edit_file` | Anchored search/replace with a whitespace-tolerant fallback |
| `delete_file` / `rename_file` / `make_directory` | Filesystem mutations |
| `search_code` | Literal or regex grep across the workspace |
| `run_command` | Shell execution with capture, timeout and a destructive-command deny-list |
| `git` | `status · init · add · commit · diff · log · branch · checkout · push · pull` |
| `finish` | Ends the run with a summary |

### Safety model

* Every path the model emits goes through `WorkspaceFS.resolve()`, which rejects `..` traversal,
  absolute paths outside the root and symlink escapes — **nothing outside the mounted folder is reachable**.
* `CommandRunner` refuses `rm -rf /`, `mkfs`, `format C:`, fork bombs, `curl … | sh`, and friends
  (the pattern list is user-editable in the config), confines the working directory to the workspace,
  enforces timeouts and kills the whole process group on stop.
* Secrets (the GitHub token) and the quota policy are stored **AES-256-GCM encrypted** with a
  PBKDF2-derived key; tampering with the file makes decryption fail and Agent3 falls back to the
  *conservative* default quotas rather than unlimited access.

---

## Enterprise quotas

The quota engine (`agent3/limits/`) gates every LLM request and every tool call:

| Limit | Default | Meaning |
|---|---|---|
| `max_requests_per_day` | 500 | Daily LLM requests |
| `max_tokens_per_day` | 1,000,000 | Daily prompt + completion tokens |
| `max_tokens_per_session` | 100,000 | Per chat session (reset with *New session*) |
| `max_tokens_per_request` | 32,000 | Rejects an oversized prompt before it is sent |
| `max_runtime_seconds_per_day` | 14,400 | Active model runtime |
| `max_agent_runs_per_day` | 100 | Autonomous runs |
| `max_tool_calls_per_run` | 60 | Runaway-loop guard |
| `min_seconds_between_requests` | 0 | Cooldown / rate limit |

`0` always means *unlimited*. The sidebar shows live gauges (green → amber at 80 % → red), a banner
appears when a limit is hit, and the agent refuses to start.

**`UsageLimitEditor.exe`** edits the encrypted policy, can set an administrator password (PBKDF2,
constant-time verification), toggles *developer mode* (bypass everything), shows a 21-day consumption
table and can reset today's counters or the whole history. The main app re-reads the policy whenever
the file changes — **no restart required**.

---

## User interface

| Region | Contents |
|---|---|
| **Toolbar** | Workspace path · model picker (populated from `/api/tags`) · live connection badge |
| **Left sidebar** | File explorer (lazy tree, context menu) · session switcher · usage gauges |
| **Centre** | Chat with markdown + syntax-highlighted code, and live **action cards** (`[AGENT] write_file … ✓ 12 ms`) with expandable output |
| **Right** | Tabbed editor (line numbers, highlighting, dirty markers) + inline / side-by-side **diff viewer** |
| **Bottom** | Embedded terminal: live agent output *and* your own commands with history |
| **Status bar** | Agent state · tokens today · request quota |

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
limits.policy.enc    AES-256-GCM encrypted quota policy
usage.sqlite3        telemetry (requests, tokens, runtime, tool calls, errors)
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
2. **build-windows** — freezes `Agent3.exe` and `UsageLimitEditor.exe` with PyInstaller, validates
   that both binaries exist and are of a plausible size, and uploads three artifacts
   (each `.exe` separately plus `Agent3-windows-x64.zip` with `INSTALL.txt`, `README.md`, `LICENSE`).
3. **release** — on a `v*` tag or `create_release: true`, publishes a **draft** GitHub Release with
   the artifacts attached.

Build locally with:

```bash
python packaging/build.py --clean          # both targets
python packaging/build.py --only agent     # just Agent3
```

---

## Development

```bash
pytest -q                                   # whole suite
pytest -q tests/test_agent.py               # the agent loop
QT_QPA_PLATFORM=offscreen pytest -q -m gui  # Qt widgets only
```

The suite covers the sandbox escape attempts, the deny-listed commands, every quota rule, the
streaming/retry/fallback behaviour of the Ollama client (with a scripted fake transport), the tool
protocol parser, the self-correcting loop (with a scripted fake model) and a headless smoke test of
every widget — no Ollama server and no display required.

Turkish documentation: [`docs/TR-KULLANIM.md`](docs/TR-KULLANIM.md).

## License

MIT — see [LICENSE](LICENSE).
