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

## Notes

- `profiles.json` holds your local machine-specific configuration (absolute
  paths, build names) and is gitignored; `profiles.example.json` shows the format.
- The GUI binds to `127.0.0.1` only — local access, no auth needed.
