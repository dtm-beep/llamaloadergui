# Llama Loader GUI

A small web GUI for configuring, saving profiles for, and launching
[llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`. On your own
machine or **detached on a remote/headless box over SSH**.

It is a launcher and settings manager: it builds the `llama-server` command
line, starts it, tails its log, and stops it. Model files and the
`llama-server` binary itself are your own. Nothing here talks to any service
outside your machine(s).

Default view, every setting section is collapsible:

![Llama Loader GUI default (wide) view: profile list on the left; settings 
(model, network, remote/SSH, CPU, GPU offloading, batch/context, sampling and
more) as collapsible sections; Save/Launch pinned to the bottom
bar](docs/screenshot.jpg)

With the log rail shown (one click, or screens under 1500px): the form moves to
one column and the right rail carries the live command preview and the
`llama-server` log tail:

![Llama Loader GUI with the log rail visible: command Preview and the rolling 
server log on the right](docs/screenshot-logs.jpg)

The log rail is drag-resizable (grab the amber bar between the panels) when the
long `llama-server` banner in the preview or log needs more room; the width is
remembered.

## Features

- **Profile system** - save/load named profiles (model + all settings) to `profiles.json`
- **One-click launch** - starts `llama-server` with the current config, live CLI preview
- **Paste a command = run exactly that** - a pasted `llama-server` command line is
  reproduced *verbatim* (no silently added flags) until you edit a field
- **Model & binary scan** - finds `.gguf` models under your models directory and
  offers the llama.cpp builds it finds on disk
- **Full tuning** - network, GPU layers, batch/context, KV-cache quantization,
  speculative decoding (MTP/draft), sampling, flash attention, logging
- **Server log tail** - live view of `llama-server.log`
- **Remote / headless targets (SSH)** - run `llama-server` on a machine without a
  desktop; launch, status, logs and stop all work against the remote target

## Tech stack

FastAPI backend + vanilla JS single page (no build step), served on
`http://127.0.0.1:7890`. Linux-oriented; should work on macOS. Requires
Python **3.10+**.

## Installation

```bash
git clone git@github.com:dtm-beep/llamaloadergui.git   # or HTTPS: …/llamaloadergui.git
cd llamaloadergui
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp profiles.example.json profiles.json                # optional example profiles
```

### Prerequisites

- A built `llama-server` binary ([llama.cpp](https://github.com/ggml-org/llama.cpp)
  or any fork). The GUI scans for `*/build*/bin/llama-server` under your home
  directory and offers them in a dropdown; any other path can be typed in.
  Conventional default: `~/llama.cpp/build/bin/llama-server`.
- One or more `.gguf` model files. The scan root defaults to
  `~/.lmstudio/models`; additional directories can be added in the UI.
- *(Remote mode only)* an OpenSSH client (`ssh`) and key-based access to the
  target. See [Remote / headless targets](#remote--headless-targets-ssh).

## Running

```bash
.venv/bin/python server.py          # or: ./run.sh
```

Open <http://127.0.0.1:7890>. Stop with `Ctrl+C`.

The server binds to `127.0.0.1` only: local access, no auth, nothing exposed.

### First steps

1. Pick a **model** (scanned) or type a path, and a **binary** (scanned) or your own.
2. Tune what you need, every field shows the exact `llama-server` flag it maps to.
3. **Launch** (`Ctrl+Enter`). The status badge turns green; the log pane follows
   `llama-server.log`.
4. **Save** the setup as a named profile for one-click relaunch.

### Run at login (optional, systemd user unit)

```ini
# ~/.config/systemd/user/llamaloadergui.service
[Unit]
Description=Llama Loader GUI

[Service]
WorkingDirectory=%h/llamaloadergui
ExecStart=%h/llamaloadergui/.venv/bin/python server.py
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now llamaloadergui
```

## Remote / headless targets (SSH)

Fill in **Remote / SSH → Target** (`user@host`, or `host` for the current user)
and the **Remote binary** path *on that machine*, then Launch. Everything else
in the form works the same — model path, workdir and binary are resolved on
the target, not here.

```text
Target         gpuuser@192.0.2.10        # empty = launch locally (RFC 5737 doc address)
SSH port       22
Remote binary  ~/llama.cpp/build/bin/llama-server
Remote workdir ~/srv                     # optional, cd'd into before launch
```

How it works:

- The server is started **detached** on the target
  (`nohup … & echo $! > ~/.llamaloader/llama-server-<port>.pid`), so it keeps
  running after the GUI closes, the SSH session ends, or you navigate away.
  One log + pid file per listen-port under `~/.llamaloader/` on the target.
- **Status** is decided *on the target* (pidfile + a loopback probe there), not
  by probing a port on this machine, otherwise a local server would be
  mistaken for the remote one. `running` = started by this GUI, `external` =
  something else holds the port there (started by hand), `stopped` = neither.
- **Logs** are tailed on the target and shipped as text; **Stop** signals the
  recorded PID.
- Launching onto a port the target already uses (by this GUI or by hand) is
  refused up front, since a detached server that dies instantly is easy to miss.

### One-time setup for a target

Key-based auth only. The GUI never prompts for a password (it would hang with
nobody able to answer; `BatchMode=yes`), so set the key up once:

```bash
ssh-copy-id gpuuser@gpu-box        # once, interactively
```

If your key has a passphrase, your SSH agent must serve it (a plain
`ssh-add ~/.ssh/id_ed25519`; on a systemd distro
`systemctl --user enable --now ssh-agent.socket` keeps one around. The GUI
also auto-detects `$XDG_RUNTIME_DIR/ssh-agent.socket`, because desktop
launchers usually start without a session environment).

Use **Test connection** in the Remote/SSH section to check all of this before
launching. It reports the usual failures in plain text (key not authorized,
host unreachable, unknown host key).

### Security notes

See [SECURITY.md](SECURITY.md) for the full policy, scope and how to report a
vulnerability privately. In short:

- **Nothing is ever killed on a pid file's word alone.** Before signalling,
  the process name on the target is compared with the binary this GUI launched
  (`/proc/<pid>/comm`, truncated like Linux reports it). If the PID was
  recycled by an unrelated process, Stop refuses and tells you to remove the
  stale file yourself rather than kill a stranger.
- Command injection is closed at the boundary: the target must match
  `[A-Za-z0-9_.@:+-]`, and every remote path is shell-quoted with a leading
  `~` deliberately left expandable so home-relative paths work on the target.
- The GUI listens on `127.0.0.1` only, on purpose: it can launch processes, so
  it is deliberately not reachable from the network.

### Verbatim import

Pasting an existing `llama-server` command line into the CLI field reproduces
that command **exactly**. Flags that the form would otherwise add (`--no-mmap`,
`-ctk/-ctv`, `--fit`, sampling defaults…) stay out until you actually edit a
field, at which point the command is rebuilt from the form. This composes with
remote mode: a pasted command runs verbatim **on the target**.

## Development

```bash
.venv/bin/python tests/test_remote.py       # remote-path tests, no network needed
.venv/bin/python tests/test_markup.py       # templates/gui.html tag balance & structure
.venv/bin/python tests/test_parse_core.py   # CLI-import parser (driven under node)
```

`test_markup.py` is the one that matters most when touching `templates/gui.html`:
a single unbalanced `<div>` silently collapses the whole layout (the template
engine renders it happily and only the browser notices).

The GUI is served as a single template (`templates/gui.html`); the backend is
a single FastAPI module (`server.py`).

## Notes

- `profiles.json` holds your local machine-specific configuration (absolute
  paths, build names) and is gitignored; `profiles.example.json` shows the
  format.
- No telemetry, no outbound requests. The only things this program connects
  to are your own machines (localhost, and the SSH target you fill in).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

```text
Copyright (c) 2026 DTM-beep — https://github.com/DTM-beep
SPDX-License-Identifier: GPL-3.0-or-later
```

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details —
you should have received a copy of the license along with this program
(`<https://www.gnu.org/licenses/gpl-3.0.html>`).

What this means in practice: use it however you like, but any modified
version or derivative you distribute must ship under the same GPL and come
with its source — the freedoms travel with the code, in both directions.

The dependency set is GPL-compatible: everything this app is built on
(FastAPI, uvicorn, Jinja2 & friends) is MIT/BSD-licensed, no third-party code
is bundled here — dependencies are fetched from PyPI at install time, each
with its own (permissive) license.
