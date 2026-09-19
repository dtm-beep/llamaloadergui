<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- SPDX-FileCopyrightText: 2026 DTM-beep  -  https://github.com/DTM-beep -->
# Security Policy

This project is a **process launcher with root-of-trust over your own machine(s)**.
That is exactly what it does, so the security posture is documented as seriously
as if it were a network service.

## Design stance (what is actually true today)

- **Loopback only, by design.** The GUI binds `127.0.0.1:7890`. It has no
  authentication layer, so it must never be reachable from the network. Do not
  port-forward it, do not expose it through a reverse proxy, and do not change
  the bind address unless you fully trust every host that can then reach it.
- **It executes processes.** The launch path builds an `argv` list and hands it
  to `Popen` as a list (never through a shell), so form values cannot inject
  shell syntax.
- **`--help` scanning stays inside the launch trust model.** On binary select,
  the GUI may run that binary once with `--help` (cached per file) to learn
  which flags it understands. This execs a binary the operator already offered
  to launch, so it adds no capability, and it is used only to **warn** about
  unknown flags, never to block or rewrite a command. A pasted CLI command is reproduced *verbatim* by design: pasting
  is an explicit, trusted act by the operator, so what you paste is what runs.
- **Command injection is closed at the remote boundary.** An SSH target must
  match `[A-Za-z0-9_.@:+-]`, and every remote path is shell-quoted with `~`
  deliberately left expandable for home-relative paths.
- **Nothing is killed on a PID file's word alone.** Before signalling, the
  process name on the target is compared with the binary this GUI launched
  (`/proc/<pid>/comm`, truncated the way Linux reports it). A recycled PID makes
  Stop refuse rather than kill a stranger.
- **Key-based SSH only.** Remote calls run with `BatchMode=yes`: no password
  prompts, no credential storage, no fallback to interactive auth.
- **No telemetry, no outbound requests.** The only things this program connects
  to are localhost and the SSH target you fill in.
- **Config on disk is plaintext.** `profiles.json` holds paths, hosts and flags
  and is written with normal umask permissions (`0644`). It is gitignored, and
  `profiles.example.json` is the shipped, sanitised format. The API key field
  included: a key typed into the form lands in `profiles.json` in plain text,
  so on a shared machine prefer the API Key **File** field. Either way,
  tighten it yourself: `chmod 600 profiles.json`.

## Scope

**In scope**

- Anything reachable on `127.0.0.1:7890` doing something it should not be able
  to (e.g. a browser page driving the launch API to run an arbitrary binary).
- Shell / path injection through the GUI's own fields, including remote mode.
- Path traversal in the model/binary scan, log tail or profile handling.
- A launch or stop affecting processes it was never asked to affect.

**Out of scope**

- Attacks from a user who already has your account, your key material, or
  loopback access on the machine: they can run `llama-server` themselves.
- DoS against your own server, or resource exhaustion you cause on purpose.
- Hardening `llama.cpp`/`llama-server` itself: report those upstream at
  [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/security).
- Vulnerabilities in dependencies beyond the version this app pins. Keep the
  venv updated.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

- Preferred: **Private vulnerability reporting** via the *Report a vulnerability*
  form on the repository *Security* tab.
- Alternative: open a normal issue titled `contact` and I will follow up in
  private, or reach me through the contact on
  [github.com/DTM-beep](https://github.com/DTM-beep).

I aim to acknowledge within **7 days** and to ship a fix (or a documented
mitigation) within **30 days** for anything confirmed. Credit in the advisory is
yours if you want it.

## Supported versions

| Version | Supported |
|---|---|
| `main` (current development) | ✅ |
| Any earlier commit | best effort, please reproduce on `main` first |

## For operators: a short hardening list

- Keep the loopback bind. If you need remote UI access, tunnel it
  (`ssh -L 7890:127.0.0.1:7890 …`) instead of re-binding.
- Give the SSH target key-only access with a passphrase-protected key, and
  restrict it with `authorized_keys` `from="…"` if the target matters.
- The GUI has no auth: on a shared or multi-user machine, keep the process to
  your own login session and consider firewalling the port.
- Review the command preview before launching a **pasted** command: verbatim
  mode is a feature, and it runs what you pasted.
- A flag the selected build does not advertise produces a warning and a launch
  confirmation, not a block (for example a new flag on an old fork). That
  check is awareness, not validation: `llama-server` itself still decides
  whether a command line works.
- `chmod 600 profiles.json` (and the log file, which contains prompts) if other
  accounts can read your home directory.
