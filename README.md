# Llama Loader GUI

Web GUI for configuring, saving profiles for, and launching
[llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` instances.

## Features

- **Profile system** — save/load named profiles (model + all settings) to `profiles.json`
- **One-click launch** — starts `llama-server` with the current config, live CLI preview
- **Model & binary scan** — finds `.gguf` models under your models directory and
  offers several llama.cpp builds to choose from
- **Full tuning** — network, CPU/GPU layers, batch/context, KV-cache quantization,
  speculative decoding (MTP/draft), sampling, flash attention, logging
- **Server log tail** — live view of `llama-server.log`
- **Remote / headless targets** — launch `llama-server` on another machine over
  SSH (a GPU box with no desktop, a server, a Pi cluster master); status, logs
  and stop all work against the remote target

## Tech stack

FastAPI + vanilla JS (Jinja templates, no build step), served on
`http://127.0.0.1:7890`.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp profiles.example.json profiles.json   # optional starting profile set
```

## Running

```bash
.venv/bin/python server.py
```

Open http://127.0.0.1:7890 — stop with Ctrl+C (or `llamaloader-stop.sh`,
which frees port 7890).

### Prerequisites

- A built `llama-server` binary (any llama.cpp fork) — pick or add its path in
  the GUI's binary dropdown
- One or more `.gguf` model files; the scan root defaults to
  `~/.lmstudio/models` and can be extended with custom paths in the UI

### Remote / headless targets (SSH)

Fill in **Remote / SSH → Target** (`user@host`, or `host` for the current user)
and the **Remote binary** path *on that machine*, then Launch. Everything else
in the form works the same — the model path, workdir and binary are resolved on
the target, not here.

```text
Target         gpuuser@192.0.2.10
SSH port       22
Remote binary  ~/llama.cpp/build/bin/llama-server
Remote workdir ~/srv            # optional, cd'd into before launch
```

How it works:

- The server is started **detached** on the target
  (`nohup … & echo $! > ~/.llamaloader/llama-server-<port>.pid`), so it keeps
  running after the GUI closes, the SSH session ends, or you navigate away.
  One log + pid file per listen-port under `~/.llamaloader/` on the target.
- **Status** is decided *on the target* (pidfile + a loopback probe there), not
  by probing a port on this machine — otherwise a local server would be
  mistaken for the remote one. `running` = started by this GUI, `external` =
  something else holds the port there (started by hand), `stopped` = neither.
- **Logs** are tailed on the target and shipped as text; **Stop** signals the
  recorded PID.
- Key-based auth only (`BatchMode=yes`): the target must already accept this
  machine's key — `ssh-copy-id user@host`. If it doesn't, "Test connection"
  says so instead of hanging on a password prompt nobody can answer.
  `SSH_AUTH_SOCK` is auto-detected (systemd's `ssh-agent.socket`) because
  desktop launchers often start without a session environment.
- **Nothing is ever killed on trust of a pid file alone.** Before signalling,
  the process name on the target is compared with the binary this GUI launched
  (`/proc/<pid>/comm`, 15-char truncated like Linux reports it). If the pid was
  recycled by an unrelated process, Stop refuses and tells you to clean the
  stale file yourself rather than kill a stranger.
- Command injection is closed off at the boundary: the target must match
  `[A-Za-z0-9_.@:+-]`, and every remote path is shell-quoted — with a leading
  `~` deliberately left expandable so home-relative paths work on the target.
- Launching onto a port the target already uses (by this GUI or by hand) is
  refused up front, since a detached server that dies instantly is easy to
  miss.

Paste a command in remote mode and it still runs **verbatim** on the target —
see *Verbatim import* below.

### Verbatim import

Pasting an existing `llama-server` command line into the CLI field reproduces
that command **exactly** — flags the form would otherwise add (`--no-mmap`,
`-ctk/-ctv`, `--fit`, sampling defaults…) stay out until you actually edit a
field, at which point the command is rebuilt from the form.

## Notes

- `profiles.json` holds your local machine-specific configuration (absolute
  paths, build names) and is gitignored; `profiles.example.json` shows the format.
- The GUI binds to `127.0.0.1` only — local access, no auth needed.
