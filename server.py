#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 DTM-beep  -  https://github.com/DTM-beep
"""Llama Loader GUI  -  scan models, save profiles, launch llama-server."""

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse, JSONResponse, RedirectResponse
from typing import Optional
from pydantic import BaseModel, Field

log = logging.getLogger("llamaloader")

# No CORS middleware on purpose: the UI is same-origin, and a wildcard
# ("allow_origins=['*']") would let any webpage in the user's browser POST
# to this launch-an-arbitrary-process endpoint. Same-origin + JSON body
# preflight keeps the attack surface at "the GUI itself".
app = FastAPI()

LMSTUDIO_DIR = Path.home() / ".lmstudio" / "models"
# Conventional build location (~ = launching user's home). Missing path is
# handled as a clean "binary not found" 400 at launch, or overridden per
# profile / the binary dropdown / the Remote-binary field.
LLAMA_SERVER = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
PROFILES_FILE = Path("profiles.json")

# Track running server
running_server: dict = {"pid": None, "proc": None}


# ── Profile persistence ──────────────────────────────────────────

def load_profiles() -> dict:
    if not PROFILES_FILE.exists():
        return {}
    try:
        data = json.loads(PROFILES_FILE.read_text() or "{}")
        if not isinstance(data, dict):
            raise ValueError("profiles.json is not a JSON object")
        return data
    except Exception as e:
        # Never silently destroy a corrupt-but-recoverable profiles file:
        # move it aside so it can be repaired by hand, start from empty.
        bad = PROFILES_FILE.with_name(f"profiles.corrupt-{int(time.time())}.json")
        try:
            PROFILES_FILE.rename(bad)
            print(f"WARN profiles.json unreadable ({e}); moved to {bad.name}")
        except OSError:
            print(f"WARN profiles.json unreadable ({e}); starting empty")
        return {}


def save_profiles(profiles: dict) -> None:
    # Atomic write: a crash mid-save must not leave a half-written profiles.json.
    tmp = PROFILES_FILE.with_name(PROFILES_FILE.name + ".tmp")
    tmp.write_text(json.dumps(profiles, indent=2))
    os.replace(tmp, PROFILES_FILE)


# ── Model scanner ────────────────────────────────────────────────

def scan_models() -> list[dict]:
    if not LMSTUDIO_DIR.exists():
        return []
    models = []
    for path in sorted(LMSTUDIO_DIR.rglob("*.gguf")):
        models.append({
            "path": str(path),
            "name": path.name,
            "rel": str(path.relative_to(LMSTUDIO_DIR)),
            "size_gb": round(path.stat().st_size / 1e9, 2),
        })
    return models


# ── llama-server binary discovery ────────────────────────────────
# Builds come and go (git worktrees, forks, -dspark, build-mtp...); a hardcoded
# dropdown rots immediately. Scan for real binaries instead. We never execute a
# llama-server binary to identify it (no --version probe)  -  metadata only.
BINARY_SCAN_ROOTS = [Path.home(), Path("/mnt/games")]


def scan_binaries() -> list[dict]:
    found: dict[str, dict] = {}
    for root in BINARY_SCAN_ROOTS:
        if not root.is_dir():
            continue
        for depth in (3, 4, 5):  # <repo>/build/bin | <repo>/build-mtp/bin
            pattern = "/".join(["*"] * depth) + "/llama-server"
            for path in sorted(root.glob(pattern)):
                if not path.is_file() or not os.access(path, os.X_OK):
                    continue
                try:
                    key = str(path.resolve())
                except OSError:
                    continue
                if key in found:
                    continue
                # label: "llamaDTM / build-mtp"  -  repo name + build dir
                repo = path.relative_to(root).parts[0]
                build = path.parent.parent.name
                kind = "dev"
                # LM Studio ships versioned runtime backends; label them by
                # flavor+version instead of an opaque ".lmstudio (backends)".
                if "extensions" in path.parts and "backends" in path.parts:
                    kind = "lmstudio"
                    pkg = path.parent.name
                    flavor = "cpu"
                    if "nvidia-cuda" in pkg: flavor = "cuda"
                    elif "vulkan" in pkg: flavor = "vulkan"
                    elif "metal" in pkg: flavor = "metal"
                    ver = pkg.rsplit("-", 1)[-1]
                    label = f"LM Studio {flavor} {ver}"
                else:
                    label = repo if build in ("build",) else f"{repo} ({build})"
                try:
                    st = path.stat()
                    mtime, size = st.st_mtime, st.st_size
                except OSError:
                    mtime, size = 0, 0
                found[key] = {
                    "path": str(path),
                    "label": label,
                    "kind": kind,
                    "mtime": mtime,
                    "size": size,
                    "is_default": key == str(LLAMA_SERVER.resolve()),
                }
    # Own dev builds first (the point of this GUI), then LM Studio runtimes;
    # newest build within each group.
    return sorted(found.values(), key=lambda b: (b["kind"] != "dev", not b["is_default"], -b["mtime"]))


# ── Command builder ──────────────────────────────────────────────

def _add(tokens: list, flag: str, value=None):
    """Append flag (and its value, as separate argv entries) to the token list.

    Values are NOT shell-quoted anymore: build_argv() output goes straight to
    subprocess.Popen, so a system prompt containing quotes survives intact.
    Use render_command() for a human-readable (shlex-quoted) preview.
    """
    if value is not None:
        tokens.extend([flag, str(value)])
    else:
        # Bare boolean flag (valueless)  -  always emit the flag itself
        tokens.append(flag)


def launch_env(cfg: dict):
    """Extra environment variables for the child process ('K=V K=V' / JSON).

    Env assignments and wrappers (taskset, nice) are how the server gets
    started, not llama-server flags  -  an importer that drops them silently
    changes behaviour (e.g. GGML_CUDA_NO_PINNED, LLAMA_ATTN_ROT_DISABLE).
    """
    raw = cfg.get("env") or ""
    if isinstance(raw, dict):
        return dict(raw) if raw else None
    raw = str(raw).strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            d = json.loads(raw)
            return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else None
        except json.JSONDecodeError:
            return None
    env = {}
    for tok in shlex.split(raw):
        if "=" in tok:
            k, _, v = tok.partition("=")
            if k:
                env[k] = v
    return env or None


def exec_wrapper(cfg: dict) -> list:
    """Verbatim command prefix (e.g. 'taskset -c 0-11') placed before argv."""
    raw = str(cfg.get("exec_prefix") or "").strip()
    return shlex.split(raw) if raw else []


# ── Per-binary flag scan (llama-server --help) ─────────────────
# Builds drift: forks (llamaDTM, ik_llama.cpp, buun) and 2026 builds accept
# flags older ones reject, and vice versa. Instead of pretending one flag set
# fits all, the GUI asks the SELECTED binary what it understands and warns
# (never blocks) when the command uses something it does not know.
_HELP_CACHE: dict = {}   # "path|mtime" -> {"flags": {...}, "version": str}


def parse_help_flags(text: str) -> dict:
    """Parse `llama-server --help` output into {longname: has_value}.

    Help lines look like `-n,    --predict, --n-predict N   description...`.
    A flag takes a value when a metavar (N, KEY, [on|off|auto], {a,b}, lo-hi,
    PATHNAME...) sits between the last flag token and the description column.
    Descriptions start lower-case; metavars never do, except `lo-hi`.
    """
    flags: dict = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        tokens = stripped.split()
        i, names = 0, []
        while i < len(tokens):
            raw = tokens[i]
            tok = raw.rstrip(",")
            if not re.fullmatch(r"-{1,2}[A-Za-z][A-Za-z0-9_-]*", tok):
                break
            names.append(tok.lstrip("-"))
            i += 1
            if not raw.endswith(","):   # comma-joined aliases keep parsing
                break
        if not names:
            continue
        metavar = False
        if i < len(tokens):
            nxt = tokens[i]
            metavar = bool(re.match(r"^[A-Z[<{]", nxt)) or nxt == "lo-hi"
        for name in names:
            if len(name) > 1:  # long form only; short forms collide with prose
                flags[name] = metavar
    return flags


def binary_flags(path: str, min_flags: int = 30) -> dict:
    """Run `--help` on the given binary and return its flag table (cached).

    min_flags guards against junk output (a wrong path that still executes,
    a wrapper printing an error); a real llama-server --help lists hundreds.
    """
    # Same single-user trust decision as _resolve_server(): the path is the
    # local user's own binary choice, gated by is_file()/min_flags below.
    p = Path(path).expanduser()  # codeql[py/path-injection] suppressed
    key = f"{p}|{p.stat().st_mtime if p.exists() else 0}|{min_flags}"
    if key in _HELP_CACHE:
        return _HELP_CACHE[key]
    if not p.is_file():
        raise ValueError(f"binary not found: {p}")
    proc = subprocess.run([str(p), "--help"], capture_output=True, text=True, timeout=15)
    out = proc.stdout + proc.stderr
    first = next((line.strip() for line in out.splitlines() if line.strip()), "")
    result = {"flags": parse_help_flags(out), "version": first[:120]}
    if len(result["flags"]) < min_flags:
        raise ValueError("could not parse --help output (is this llama-server?)")
    _HELP_CACHE[key] = result
    return result


def _resolve_server(raw) -> str:
    """Normalize the llama-server binary path (empty/relative -> known default).

    The path is a user-provided *design feature* of this GUI: it selects the
    binary the user is about to launch anyway. The server binds to 127.0.0.1
    only (single-user trust boundary: anyone who can reach this API is a
    logged-in user on this machine, who can exec any binary directly).
    Input is still normalized: NUL bytes and control characters are rejected,
    relative paths resolve under $HOME, never under an attacker-chosen root.
    """
    raw = str(raw or "").strip()
    if not raw:
        return str(LLAMA_SERVER)
    if any(ord(c) < 32 or ord(c) == 127 for c in raw):
        raise ValueError("binary path contains control characters")
    # Local-only GUI (127.0.0.1): the binary path is the user's own launch
    # choice, the same one the GUI execs on Start.
    p = Path(raw).expanduser()  # codeql[py/path-injection] suppressed
    return str(p if p.is_absolute() else (Path.home() / p).resolve())


def build_argv(cfg: dict) -> list:
    """Build the argv list (list of plain strings, no shell quoting).

    Accepts a PARTIAL dict (e.g. a freshly parsed CLI command): missing keys
    fall back to the Config defaults, so a half-filled config can never raise
    KeyError or silently lose a flag.
    """
    cfg = {**_config_defaults(), **cfg}
    server_path = _resolve_server(cfg.get("llama_server"))
    parts = [server_path]
    # Model
    parts.extend(["-m", str(cfg.get("model", ""))])
    if cfg.get("model_url"):
        _add(parts, "--model-url", cfg["model_url"])
    if cfg.get("hf_repo"):
        _add(parts, "--hf-repo", cfg["hf_repo"])
    if cfg.get("hf_file"):
        _add(parts, "--hf-file", cfg["hf_file"])

    # Network
    _add(parts, "--host", cfg["host"])
    _add(parts, "--port", cfg["port"])
    # -np and --parallel are the same flag (server slots); emit it once.
    _add(parts, "--parallel", cfg["parallel"])
    if cfg.get("alias"):
        _add(parts, "--alias", cfg["alias"])

    # CPU
    _add(parts, "-t", cfg["threads"])
    _add(parts, "--threads-batch", cfg["threads_batch"])
    _add(parts, "--prio", cfg["prio"])
    _add(parts, "--poll", cfg["poll"])
    _add(parts, "--prio-batch", cfg["prio_batch"])
    if cfg.get("cpu_mask"):
        _add(parts, "-C", cfg["cpu_mask"])
    if cfg.get("cpu_range"):
        _add(parts, "-Cr", cfg["cpu_range"])
    if cfg.get("cpu_strict"):
        _add(parts, "--cpu-strict", "1")
    if cfg.get("cpu_mask_batch"):
        _add(parts, "-Cb", cfg["cpu_mask_batch"])
    if cfg.get("cpu_range_batch"):
        _add(parts, "-Crb", cfg["cpu_range_batch"])
    if cfg.get("cpu_strict_batch"):
        _add(parts, "--cpu-strict-batch", "1")

    # GPU
    # -1 == llama.cpp's own default (auto): emit nothing, so a profile that
    # never set -ngl cannot silently pin a layer count the source command left
    # to auto/fit.
    if cfg.get("ngl") is not None and int(cfg["ngl"]) >= 0:
        _add(parts, "--n-gpu-layers", cfg["ngl"])
    if cfg.get("device"):
        _add(parts, "--device", cfg["device"])
    if cfg.get("split_mode") and cfg["split_mode"] != "layer":
        _add(parts, "--split-mode", cfg["split_mode"])
    if cfg.get("tensor_split"):
        _add(parts, "--tensor-split", cfg["tensor_split"])
    _add(parts, "--main-gpu", cfg["main_gpu"])
    # --fit is binary (on/off); emit it explicitly so imports round-trip
    fit_on = bool(cfg.get("fit"))
    _add(parts, "--fit", "on" if fit_on else "off")
    if fit_on and cfg.get("fit_target"):
        _add(parts, "--fit-target", cfg["fit_target"])
    if fit_on and cfg.get("fit_ctx") and cfg["fit_ctx"] != 4096:
        _add(parts, "--fit-ctx", cfg["fit_ctx"])
    if cfg.get("op_offload") is False:
        _add(parts, "--no-op-offload")
    if cfg.get("cmoe"):
        _add(parts, "--cpu-moe")
    if cfg.get("n_cmoe"):
        _add(parts, "--n-cpu-moe", cfg["n_cmoe"])

    # Context & Batch
    _add(parts, "--ctx-size", cfg["ctx_size"])
    _add(parts, "--n-predict", cfg["n_predict"])
    _add(parts, "--batch-size", cfg["batch"])
    _add(parts, "--ubatch-size", cfg["ubatch"])
    if cfg.get("keep") and cfg["keep"] != 0:
        _add(parts, "--keep", cfg["keep"])
    if cfg.get("context_shift"):
        _add(parts, "--context-shift")

    # Loading
    if cfg.get("load_mode") and cfg["load_mode"] != "mmap":
        _add(parts, "--load-mode", cfg["load_mode"])
    else:
        if cfg.get("no_mmap"):
            _add(parts, "--no-mmap")
    if cfg.get("mlock"):
        _add(parts, "--mlock")
    if cfg.get("direct_io"):
        _add(parts, "--direct-io")
    if cfg.get("numa") and cfg["numa"] != "none":
        _add(parts, "--numa", cfg["numa"])
    if cfg.get("no_repack"):
        _add(parts, "--no-repack")
    if cfg.get("no_host"):
        _add(parts, "--no-host")

    # Flash Attention
    if cfg.get("flash_attn") and cfg["flash_attn"] != "auto":
        _add(parts, "-fa", cfg["flash_attn"])

    # KV Cache
    _add(parts, "-ctk", cfg["cache_k"])
    _add(parts, "-ctv", cfg["cache_v"])
    if cfg.get("kv_offload") is False:
        _add(parts, "--no-kv-offload")
    if cfg.get("swa_full"):
        _add(parts, "--swa-full")
    if cfg.get("kv_unified"):
        _add(parts, "--kv-unified")
    if cfg.get("cache_ram") and cfg["cache_ram"] != 8192:
        _add(parts, "--cache-ram", cfg["cache_ram"])
    if cfg.get("ctx_checkpoints") and cfg["ctx_checkpoints"] != 32:
        _add(parts, "--ctx-checkpoints", cfg["ctx_checkpoints"])

    # RoPE
    if cfg.get("rope_scaling") and cfg["rope_scaling"] != "linear":
        _add(parts, "--rope-scaling", cfg["rope_scaling"])
    if cfg.get("rope_scale") and cfg["rope_scale"] != 1.0:
        _add(parts, "--rope-scale", cfg["rope_scale"])
    if cfg.get("rope_freq_base"):
        _add(parts, "--rope-freq-base", cfg["rope_freq_base"])
    if cfg.get("rope_freq_scale") and cfg["rope_freq_scale"] != 1.0:
        _add(parts, "--rope-freq-scale", cfg["rope_freq_scale"])

    # YaRN
    if cfg.get("yarn_orig_ctx"):
        _add(parts, "--yarn-orig-ctx", cfg["yarn_orig_ctx"])
    if cfg.get("yarn_ext_factor") and cfg["yarn_ext_factor"] != -1.0:
        _add(parts, "--yarn-ext-factor", cfg["yarn_ext_factor"])
    if cfg.get("yarn_attn_factor") and cfg["yarn_attn_factor"] != -1.0:
        _add(parts, "--yarn-attn-factor", cfg["yarn_attn_factor"])
    if cfg.get("yarn_beta_slow") and cfg["yarn_beta_slow"] != -1.0:
        _add(parts, "--yarn-beta-slow", cfg["yarn_beta_slow"])
    if cfg.get("yarn_beta_fast") and cfg["yarn_beta_fast"] != -1.0:
        _add(parts, "--yarn-beta-fast", cfg["yarn_beta_fast"])

    # LoRA
    if cfg.get("lora"):
        _add(parts, "--lora", cfg["lora"])
    if cfg.get("lora_scaled"):
        _add(parts, "--lora-scaled", cfg["lora_scaled"])

    # Control Vector
    if cfg.get("control_vector"):
        _add(parts, "--control-vector", cfg["control_vector"])
    if cfg.get("control_vector_scaled"):
        _add(parts, "--control-vector-scaled", cfg["control_vector_scaled"])

    # Speculative Decoding
    if cfg.get("speculative") and not cfg.get("no_speculative"):
        if cfg.get("spec_draft_model"):
            _add(parts, "--spec-draft-model", cfg["spec_draft_model"])
        _add(parts, "--spec-type", cfg.get("spec_type", "draft-mtp"))
        _add(parts, "--spec-draft-n-max", cfg["spec_n_max"])
        _add(parts, "--spec-draft-n-min", cfg["spec_n_min"])
        _add(parts, "--spec-draft-p-split", cfg["spec_p_split"])
        _add(parts, "--spec-draft-p-min", cfg["spec_p_min"])
        if cfg.get("spec_draft_threads"):
            _add(parts, "--spec-draft-threads", cfg["spec_draft_threads"])
        if cfg.get("spec_draft_ngl"):
            _add(parts, "--spec-draft-n-gpu-layers", cfg["spec_draft_ngl"])
        if cfg.get("spec_draft_device"):
            _add(parts, "--spec-draft-device", cfg["spec_draft_device"])
        _add(parts, "--spec-draft-type-k", cfg.get("spec_draft_cache_k", "f16"))
        _add(parts, "--spec-draft-type-v", cfg.get("spec_draft_cache_v", "f16"))
        if cfg.get("spec_draft_cpu_moe"):
            _add(parts, "--spec-draft-cpu-moe")
        if cfg.get("spec_draft_n_cpu_moe"):
            _add(parts, "--spec-draft-n-cpu-moe", cfg["spec_draft_n_cpu_moe"])
        if cfg.get("spec_draft_backend_sampling") is False:
            _add(parts, "--no-spec-draft-backend-sampling")

    # Sampling
    _add(parts, "--temp", cfg["temp"])
    _add(parts, "--top-p", cfg["top_p"])
    _add(parts, "--top-k", cfg["top_k"])
    _add(parts, "--min-p", cfg["min_p"])
    if cfg.get("top_n_sigma") and cfg["top_n_sigma"] != -1.0:
        _add(parts, "--top-n-sigma", cfg["top_n_sigma"])
    if cfg.get("typical_p") and cfg["typical_p"] != 1.0:
        _add(parts, "--typical-p", cfg["typical_p"])
    if cfg.get("xtc_probability") and cfg["xtc_probability"] > 0:
        _add(parts, "--xtc-probability", cfg["xtc_probability"])
        _add(parts, "--xtc-threshold", cfg["xtc_threshold"])
    if cfg.get("seed") and cfg["seed"] != -1:
        _add(parts, "-s", cfg["seed"])
    if cfg.get("samplers"):
        _add(parts, "--samplers", cfg["samplers"])
    if cfg.get("ignore_eos"):
        _add(parts, "--ignore-eos")

    # Repetition
    if cfg.get("repeat_last_n") and cfg["repeat_last_n"] != 64:
        _add(parts, "--repeat-last-n", cfg["repeat_last_n"])
    if cfg.get("repeat_penalty") and cfg["repeat_penalty"] != 1.0:
        _add(parts, "--repeat-penalty", cfg["repeat_penalty"])
    if cfg.get("presence_penalty") and cfg["presence_penalty"] != 0.0:
        _add(parts, "--presence-penalty", cfg["presence_penalty"])
    if cfg.get("frequency_penalty") and cfg["frequency_penalty"] != 0.0:
        _add(parts, "--frequency-penalty", cfg["frequency_penalty"])

    # DRY
    if cfg.get("dry_multiplier") and cfg["dry_multiplier"] > 0:
        _add(parts, "--dry-multiplier", cfg["dry_multiplier"])
        _add(parts, "--dry-base", cfg["dry_base"])
        _add(parts, "--dry-allowed-length", cfg["dry_allowed_length"])
        if cfg.get("dry_penalty_last_n") and cfg["dry_penalty_last_n"] != -1:
            _add(parts, "--dry-penalty-last-n", cfg["dry_penalty_last_n"])

    # Adaptive-p
    if cfg.get("adaptive_target") and cfg["adaptive_target"] >= 0:
        _add(parts, "--adaptive-target", cfg["adaptive_target"])
        _add(parts, "--adaptive-decay", cfg["adaptive_decay"])

    # Dynamic Temperature
    if cfg.get("dynatemp_range") and cfg["dynatemp_range"] > 0:
        _add(parts, "--dynatemp-range", cfg["dynatemp_range"])
        _add(parts, "--dynatemp-exp", cfg["dynatemp_exp"])

    # Mirostat
    if cfg.get("mirostat") and cfg["mirostat"] > 0:
        _add(parts, "--mirostat", cfg["mirostat"])
        _add(parts, "--mirostat-lr", cfg["mirostat_lr"])
        _add(parts, "--mirostat-ent", cfg["mirostat_ent"])

    # Grammar / JSON Schema
    if cfg.get("grammar"):
        _add(parts, "--grammar", cfg["grammar"])
    if cfg.get("grammar_file"):
        _add(parts, "--grammar-file", cfg["grammar_file"])
    if cfg.get("json_schema"):
        _add(parts, "-j", json.dumps(cfg["json_schema"]))
    if cfg.get("json_schema_file"):
        _add(parts, "-jf", cfg["json_schema_file"])

    # Logit bias
    if cfg.get("logit_bias"):
        _add(parts, "--logit-bias", cfg["logit_bias"])

    # Chat / Conversation
    if cfg.get("tools"):
        _add(parts, "--tools", cfg["tools"])
    if cfg.get("jinja") is False:
        _add(parts, "--no-jinja")
    if cfg.get("system_prompt"):
        _add(parts, "--sys", cfg["system_prompt"])
    if cfg.get("conversation") is False:
        _add(parts, "--no-conversation")
    if cfg.get("single_turn"):
        _add(parts, "--single-turn")
    if cfg.get("reasoning") and cfg["reasoning"] != "auto":
        _add(parts, "--reasoning", cfg["reasoning"])
    if cfg.get("reasoning_budget") and cfg["reasoning_budget"] != -1:
        _add(parts, "--reasoning-budget", cfg["reasoning_budget"])
    if cfg.get("reasoning_budget_message"):
        _add(parts, "--reasoning-budget-message", cfg["reasoning_budget_message"])
    if cfg.get("reasoning_preserve") is False:
        _add(parts, "--no-reasoning-preserve")
    elif cfg.get("reasoning_preserve") is True:
        _add(parts, "--reasoning-preserve")
    if cfg.get("reasoning_format"):
        _add(parts, "--reasoning-format", cfg["reasoning_format"])
    # Rebuild --chat-template-kwargs from the raw JSON string so arbitrary keys
    # (reasoning_effort, the fork-specific ones, ...) survive import -> launch.
    kwargs = {}
    raw_kwargs = cfg.get("chat_template_kwargs") or ""
    if isinstance(raw_kwargs, dict):
        kwargs = dict(raw_kwargs)
    elif str(raw_kwargs).strip():
        try:
            parsed_kwargs = json.loads(raw_kwargs)
            if isinstance(parsed_kwargs, dict):
                kwargs = parsed_kwargs
            else:
                kwargs = {"value": parsed_kwargs}  # llama.cpp wraps non-objects
        except json.JSONDecodeError:
            kwargs = {}
    if cfg.get("preserve_thinking") and "preserve_thinking" not in kwargs:
        kwargs["preserve_thinking"] = True
    if kwargs:
        _add(parts, "--chat-template-kwargs", json.dumps(kwargs, separators=(",", ":")))
    if cfg.get("chat_template") and cfg["chat_template"] != "auto":
        _add(parts, "--chat-template", cfg["chat_template"])
    if cfg.get("chat_template_file"):
        _add(parts, "--chat-template-file", cfg["chat_template_file"])
    if cfg.get("skip_chat_parsing"):
        _add(parts, "--skip-chat-parsing")
    if cfg.get("multiline_input"):
        _add(parts, "--multiline-input")
    if cfg.get("reverse_prompt"):
        _add(parts, "-r", cfg["reverse_prompt"])
    if cfg.get("no_warmup"):
        _add(parts, "--no-warmup")
    if cfg.get("no_display_prompt"):
        _add(parts, "--no-display-prompt")
    if cfg.get("special"):
        _add(parts, "--special")
    if cfg.get("backend_sampling"):
        _add(parts, "--backend-sampling")

    # Multimodal
    if cfg.get("mmproj"):
        _add(parts, "--mmproj", cfg["mmproj"])
    if cfg.get("mmproj_auto") is False:
        _add(parts, "--no-mmproj")
    if cfg.get("mmproj_offload") is False:
        _add(parts, "--no-mmproj-offload")
    if cfg.get("image_min_tokens"):
        _add(parts, "--image-min-tokens", cfg["image_min_tokens"])
    if cfg.get("image_max_tokens"):
        _add(parts, "--image-max-tokens", cfg["image_max_tokens"])
    if cfg.get("media_files"):
        _add(parts, "--image", cfg["media_files"])

    # Override
    if cfg.get("override_tensor"):
        _add(parts, "--override-tensor", cfg["override_tensor"])
    if cfg.get("override_kv"):
        _add(parts, "--override-kv", cfg["override_kv"])
    if cfg.get("check_tensors"):
        _add(parts, "--check-tensors")

    # Logging
    if cfg.get("log_file"):
        _add(parts, "--log-file", cfg["log_file"])
    if cfg.get("log_colors") and cfg["log_colors"] != "auto":
        _add(parts, "--log-colors", cfg["log_colors"])
    if cfg.get("log_verbosity") and cfg["log_verbosity"] != 1:
        _add(parts, "--log-verbosity", cfg["log_verbosity"])
    if cfg.get("log_verbose"):
        _add(parts, "--verbose")
    if cfg.get("log_disable"):
        _add(parts, "--log-disable")
    if cfg.get("log_prefix") is False:
        _add(parts, "--no-log-prefix")
    if cfg.get("log_timestamps"):
        _add(parts, "--log-timestamps")

    # Misc
    if cfg.get("offline"):
        _add(parts, "--offline")
    if cfg.get("no_perf"):
        _add(parts, "--no-perf")
    if cfg.get("show_timings") is False:
        _add(parts, "--no-show-timings")

    # Server security. All opt-in: nothing is emitted at the neutral default,
    # so saving a profile never injects a flag an older binary would reject.
    if cfg.get("api_key"):
        _add(parts, "--api-key", cfg["api_key"])
    if cfg.get("api_key_file"):
        _add(parts, "--api-key-file", cfg["api_key_file"])
    if cfg.get("api_prefix"):
        _add(parts, "--api-prefix", cfg["api_prefix"])
    if cfg.get("ssl_key_file"):
        _add(parts, "--ssl-key-file", cfg["ssl_key_file"])
    if cfg.get("ssl_cert_file"):
        _add(parts, "--ssl-cert-file", cfg["ssl_cert_file"])
    if cfg.get("cors_origins"):
        _add(parts, "--cors-origins", cfg["cors_origins"])
    if cfg.get("cors_headers"):
        _add(parts, "--cors-headers", cfg["cors_headers"])
    if cfg.get("cors_credentials"):
        _add(parts, "--cors-credentials")
    if cfg.get("metrics"):
        _add(parts, "--metrics")

    # Serving / router.
    if cfg.get("sleep_idle_seconds") != "":
        _add(parts, "--sleep-idle-seconds", cfg["sleep_idle_seconds"])
    if cfg.get("threads_http") != "":
        _add(parts, "--threads-http", cfg["threads_http"])
    if cfg.get("cache_prompt") is False:
        _add(parts, "--no-cache-prompt")
    if cfg.get("cache_reuse") != "":
        _add(parts, "--cache-reuse", cfg["cache_reuse"])
    if cfg.get("system_prompt_file"):
        _add(parts, "--system-prompt-file", cfg["system_prompt_file"])
    if cfg.get("models_dir"):
        _add(parts, "--models-dir", cfg["models_dir"])
    if cfg.get("models_max") != "":
        _add(parts, "--models-max", cfg["models_max"])
    if cfg.get("models_autoload") is False:
        _add(parts, "--no-models-autoload")

    # Verbatim passthrough for flags not modeled by a dedicated form field.
    # shlex.split restores the quoting we preserved when parsing, so each flag
    # re-joins exactly as it was imported (e.g. --chat-template-kwargs '{...}').
    if cfg.get("extra_args"):
        parts.extend(shlex.split(cfg["extra_args"]))

    return parts


def render_command(argv: list, cfg: dict | None = None) -> str:
    """Human-readable multi-line command for the preview box (display only).

    Includes the env assignments and wrapper so the preview is the *whole*
    launch line, not just argv (otherwise the copied command lies).

    Layout: **one flag with its value on one line** (previously every token
    got its own line, which in the narrow log rail read as a zig-zag of stub
    lines: `--host` / `\\` / `127.0.0.1` / `\\`). Continuations stay valid
    shell, so the box is still copy-paste runnable; only argv[0] (the binary)
    leads the first line, and a bare non-flag token keeps its own line.
    """
    prefix = []
    if cfg is not None:
        env = launch_env(cfg)
        if env:
            prefix.extend(f"{k}={v}" for k, v in env.items())
        prefix.extend(exec_wrapper(cfg))

    def is_flag(t: str) -> bool:
        # `-1` / `-0.5` are VALUES (llama.cpp takes them after a required-argument
        # flag); matching a leading dash alone would split them onto their own line.
        return t.startswith("-") and not re.match(r"^-\d", t)

    toks = [shlex.quote(str(t)) for t in argv]
    lines: list[str] = []
    first = " ".join(prefix + ([toks[0]] if toks else []))
    if toks:
        lines.append(first)
    elif prefix:
        lines.append(first)
    i = 1
    while i < len(toks):
        tok = toks[i]
        if is_flag(tok):
            # `-f value` / `--flag value` on one line; `--flag=v` is already one
            if i + 1 < len(toks) and not is_flag(toks[i + 1]):
                lines.append(f"{tok} {toks[i + 1]}")
                i += 2
                continue
            lines.append(tok)
        else:
            lines.append(tok)
        i += 1
    return " \\\n  ".join(lines)


# ── Pydantic model ──────────────────────────────────────────────

class Config(BaseModel):
    model_config = {"extra": "allow"}

    # Profile
    name: str = ""

    # Model
    model: str
    llama_server: str = str(LLAMA_SERVER)
    # Launch environment / wrapper, captured verbatim so imports stay faithful
    # (e.g. 'LLAMA_ATTN_ROT_DISABLE=1 GGML_CUDA_NO_PINNED=1' / 'taskset -c 0-11').
    env: str = ""
    exec_prefix: str = ""
    # Remote (SSH) target  -  empty host = launch on this machine.
    remote_host: str = ""
    remote_ssh_port: int = 22
    remote_bin: str = ""
    remote_workdir: str = ""
    # Verbatim import mode: token list straight from a pasted CLI command.
    # When present it REPLACES build_argv() output for preview and launch.
    imported_argv: Optional[list[str]] = None
    model_url: str = ""
    hf_repo: str = ""
    hf_file: str = ""

    # Network
    host: str = "127.0.0.1"
    port: int = 8080
    pipeline: int = 1
    parallel: int = 1
    alias: str = ""

    # CPU
    threads: int = 6
    threads_batch: int = 0
    prio: int = 0
    poll: int = 50
    prio_batch: int = 0
    cpu_mask: str = ""
    cpu_range: str = ""
    cpu_strict: bool = False
    cpu_mask_batch: str = ""
    cpu_range_batch: str = ""
    cpu_strict_batch: bool = False

    # GPU
    ngl: int = -1  # -1 = auto (llama.cpp default); >= 0 pins the layer count
    device: str = ""
    split_mode: str = "layer"
    tensor_split: str = ""
    main_gpu: int = 0
    fit: bool = True
    fit_target: str = "1024"
    fit_ctx: int = 4096
    op_offload: bool = True
    cmoe: bool = False
    n_cmoe: int = 0

    # Context & Batch
    ctx_size: int = 100400
    n_predict: int = -1
    batch: int = 2048
    ubatch: int = 128
    keep: int = 0
    context_shift: bool = False

    # Loading
    no_mmap: bool = True
    mlock: bool = False
    direct_io: bool = False
    load_mode: str = "mmap"
    numa: str = "none"
    no_repack: bool = False
    no_host: bool = False

    # Flash Attention
    flash_attn: str = "auto"

    # KV Cache
    cache_k: str = "q4_0"
    cache_v: str = "q4_0"
    kv_offload: bool = True
    swa_full: bool = False
    cache_ram: int = 8192
    ctx_checkpoints: int = 32
    kv_unified: bool = False

    # RoPE
    rope_scaling: str = "linear"
    rope_scale: float = 1.0
    rope_freq_base: str = ""
    rope_freq_scale: float = 1.0

    # YaRN
    yarn_orig_ctx: str = ""
    yarn_ext_factor: float = -1.0
    yarn_attn_factor: float = -1.0
    yarn_beta_slow: float = -1.0
    yarn_beta_fast: float = -1.0

    # LoRA
    lora: str = ""
    lora_scaled: str = ""

    # Control Vector
    control_vector: str = ""
    control_vector_scaled: str = ""

    # Speculative Decoding
    speculative: bool = True
    spec_draft_model: str = ""
    spec_type: str = "draft-mtp"
    spec_n_max: int = 3
    spec_n_min: int = 0
    spec_p_split: float = 0.10
    spec_p_min: float = 0.0  # llama.cpp default (common.h: p_min = 0.0f)
    spec_draft_threads: str = ""
    spec_draft_ngl: str = ""
    spec_draft_device: str = ""
    spec_draft_cache_k: str = "f16"
    spec_draft_cache_v: str = "f16"
    spec_draft_cpu_moe: bool = False
    spec_draft_n_cpu_moe: int = 0
    spec_draft_backend_sampling: bool = True

    # Sampling
    temp: float = 0.8
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    top_n_sigma: float = -1.0
    typical_p: float = 1.0
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.10
    seed: int = -1
    samplers: str = ""
    ignore_eos: bool = False

    # Repetition
    repeat_last_n: int = 64
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # DRY
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = -1

    # Adaptive-p
    adaptive_target: float = -1.0
    adaptive_decay: float = 0.90

    # Dynamic Temperature
    dynatemp_range: float = 0.0
    dynatemp_exp: float = 1.0

    # Mirostat
    mirostat: int = 0
    mirostat_lr: float = 0.10
    mirostat_ent: float = 5.0

    # Grammar
    grammar: str = ""
    grammar_file: str = ""
    json_schema: str = ""
    json_schema_file: str = ""

    # Logit bias
    logit_bias: str = ""

    # Chat
    system_prompt: str = ""
    conversation: bool = True
    single_turn: bool = False
    multiline_input: bool = False
    reverse_prompt: str = ""
    no_warmup: bool = False
    no_display_prompt: bool = False
    special: bool = False
    backend_sampling: bool = False
    preserve_thinking: bool = False
    reasoning: str = "auto"
    reasoning_format: str = ""
    reasoning_budget: int = -1
    reasoning_budget_message: str = ""
    reasoning_preserve: Optional[bool] = None
    # Raw JSON object string, e.g. {"reasoning_effort":"medium","preserve_thinking":false}.
    # Kept verbatim so ANY template kwarg round-trips (only preserve_thinking
    # used to survive, which silently dropped reasoning_effort on launch).
    chat_template_kwargs: str = ""
    chat_template: str = "auto"
    chat_template_file: str = ""
    skip_chat_parsing: bool = False

    # Tools / Jinja
    tools: str = ""
    jinja: bool = True
    extra_args: str = ""

    # Multimodal
    mmproj: str = ""
    mmproj_auto: bool = True
    mmproj_offload: bool = True
    image_min_tokens: str = ""
    image_max_tokens: str = ""
    media_files: str = ""

    # Override
    override_tensor: str = ""
    override_kv: str = ""
    check_tensors: bool = False

    # Logging
    log_file: str = ""
    log_colors: str = "auto"
    log_verbosity: int = 1
    log_verbose: bool = False
    log_disable: bool = False
    log_prefix: bool = True
    log_timestamps: bool = False

    # Misc
    offline: bool = False
    no_perf: bool = False
    show_timings: bool = True

    # Server security (flag sync 2026-09-19 vs upstream arg.cpp).
    # Every default here is NEUTRAL: nothing is emitted unless the user sets
    # it, so a profile stays launchable on older builds that lack the flag.
    # NOTE: profiles.json stores these in plain text (SECURITY.md).
    api_key: str = ""
    api_key_file: str = ""
    api_prefix: str = ""
    ssl_key_file: str = ""
    ssl_cert_file: str = ""
    cors_origins: str = ""
    cors_headers: str = ""
    cors_credentials: bool = False
    metrics: bool = False

    # Serving / router (same neutral rule)
    sleep_idle_seconds: str = ""
    threads_http: str = ""
    cache_prompt: bool = True          # llama-server default: on; only --no-cache-prompt is emitted
    cache_reuse: str = ""
    system_prompt_file: str = ""
    models_dir: str = ""
    models_max: str = ""
    models_autoload: bool = True       # default on; only --no-models-autoload is emitted


_CONFIG_DEFAULTS: dict = {}


def _verbatim_argv(d: dict) -> list | None:
    """Raw argv from a pasted command (verbatim import mode), else None.

    Rebuilding from the form would add flags the pasted command never had
    (--no-mmap, -ctk/-ctv, --fit off, --temp...), silently changing server
    behaviour vs. the command the user imported. While the import is unedited,
    preview and launch must reproduce it exactly. Tokens go through Popen as a
    list (no shell), so no further quoting is needed or wanted.
    """
    argv = d.get("imported_argv")
    if not isinstance(argv, list) or not argv:
        return None
    argv = [str(t) for t in argv if str(t) != ""]
    return argv or None


def _config_defaults() -> dict:
    """Complete default config (computed once, after Config is defined)."""
    if not _CONFIG_DEFAULTS:
        _CONFIG_DEFAULTS.update(Config(model="").model_dump())
    return dict(_CONFIG_DEFAULTS)


# ── Remote (SSH) targets ───────────────────────────────────────────────
# First-class headless support: launch llama-server on another machine over
# SSH, detached (nohup) with a pidfile + log under ~/.llamaloader there, so
# the server survives the SSH session closing and status/logs/stop all work
# without the GUI holding a tunnel open. Key-based auth only: BatchMode
# guarantees we never hang on a passphrase/password prompt nobody can answer
# (ssh-askpass is absent on this KDE box).

import asyncio as _aio

# Desktop launchers do not always carry the session environment; point at the
# systemd ssh-agent socket (enabled on this box) when nothing else is set, or
# every remote launch dies on a key that is actually sitting in the agent.
if not os.environ.get("SSH_AUTH_SOCK"):
    _rt = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    _agent = os.path.join(_rt, "ssh-agent.socket")
    if os.path.exists(_agent):
        os.environ["SSH_AUTH_SOCK"] = _agent

SSH_RE = re.compile(r"^[A-Za-z0-9_.@:+-]+$")       # user@host / host / ipv6-zone-safe
REMOTE_DIR = "~/.llamaloader"


def _remote_target(d: dict) -> tuple[dict | None, str | None]:
    """Validate remote fields. Returns (target, error); (None, None) = local launch.

    Validation problems come back as a curated error string instead of an
    exception, so callers never turn a caught exception's message into an API
    response (no stack-trace/exception-info exposure, CodeQL py/stack-trace-exposure).
    """
    host = str(d.get("remote_host") or "").strip()
    if not host:
        return None, None
    if not SSH_RE.match(host):
        return None, "Invalid SSH target (allowed: [user@]host, no spaces or shell metacharacters)"
    try:
        ssh_port = int(d.get("remote_ssh_port") or 22)
    except (TypeError, ValueError):
        return None, "Invalid SSH port"
    rb = str(d.get("remote_bin") or "").strip()
    rb = rb if rb.startswith(("/", "~", "./")) else ("~/" + rb)
    workdir = str(d.get("remote_workdir") or "").strip() or None
    if workdir and not workdir.startswith(("/", "~", "./")):
        return None, "Invalid remote workdir (must be an absolute path)"
    return {"host": host, "ssh_port": ssh_port, "bin": rb, "workdir": workdir}, None


def _ssh_base(t: dict) -> list:
    return [
        "ssh",
        "-p", str(t["ssh_port"]),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        t["host"],
    ]


async def _run_ssh(t: dict, remote_cmd: str, timeout: float = 20) -> tuple[int, str, str]:
    """Run one SSH command; returns (rc, stdout, stderr). Never raises."""
    argv = [*_ssh_base(t), remote_cmd]
    try:
        proc = await _aio.create_subprocess_exec(
            *argv, stdout=_aio.subprocess.PIPE, stderr=_aio.subprocess.PIPE)
        out, err = await _aio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")
    except _aio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return 124, "", f"ssh: timed out after {timeout}s"
    except OSError as e:
        return 127, "", f"ssh: {e}"


def _rq(path: str) -> str:
    """Quote a remote path but keep a leading ~ expanded by the REMOTE shell
    (shlex.quote would wrap it in single quotes and kill the tilde)."""
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~" + shlex.quote(path[1:])
    if path.startswith("~") and len(path) > 1:
        # ~user/rest  -  expand user, quote the rest
        rest = path[1:].split("/", 1)
        return "~" + rest[0] + (shlex.quote("/" + rest[1]) if len(rest) > 1 else "")
    return shlex.quote(path)


def _remote_files(t: dict, listen_port: int) -> tuple[str, str]:
    """(log, pidfile) on the target  -  keyed by llama-server LISTEN port, so two
    servers on one target (8080/8081) do not fight over one pidfile."""
    p = int(listen_port)
    return (f"{REMOTE_DIR}/llama-server-{p}.log", f"{REMOTE_DIR}/llama-server-{p}.pid")


def _remote_launch_script(argv: list, t: dict, env: dict | None, listen_port: int) -> str:
    """POSIX snippet run on the target: detach (nohup), pidfile, log.

    Tokens are shlex-quoted; env assignments go INSIDE the nohup subshell (an
    `env A=1 nohup …` outer prefix dies with the shell). Log/pidfile are
    expanded remotely ($HOME), consistent with the status/stop/tail commands
    which resolve them the same way.
    """
    toks = [str(x) for x in argv]
    # argv[0] (the binary) may be ~/, ~user/, ./. or absolute: keep tilde/relative
    # readable for the remote shell; model paths etc. may also be ~ - quote via _rq.
    body = " ".join(_rq(x) if (x.startswith(("~/", "~", "./")) or x == "~") else shlex.quote(x) for x in toks)
    env_str = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
    inner = f"{env_str} {body}" if env_str else body
    log, pid = _remote_files(t, listen_port)
    cd = f"cd {_rq(t['workdir'])} || exit 2; " if t.get("workdir") else ""
    return (
        f"mkdir -p {REMOTE_DIR}; {cd}"
        f"( nohup {inner} > {log} 2>&1 < /dev/null & echo $! > {pid} )"
    )


# Status is decided ON THE TARGET (a TCP probe from this machine would test
# this machine's own port, or miss a server bound to the target's loopback).
#  RUNNING <pid>  -  pidfile process alive (ours, or a previous run of this GUI)
#  BUSY         -  nothing in our pidfile, but something listens on the port there
#  NOT_RUNNING  -  nothing
# Pidfile-reuse guard: a stale pid file can point at a recycled PID belonging
# to something else entirely (sshd, a database, even a hand-started
# llama-server). Both status and stop check the process name against the
# binary this GUI launched and ignore the pidfile when it does not match.
_REMOTE_PROG = ""

REMOTE_STATUS_CMD = (
    "d=$HOME/.llamaloader; p=$(cat $d/llama-server-{port}.pid 2>/dev/null); "
    "c=; [ -n \"$p\" ] && c=$(cat /proc/$p/comm 2>/dev/null || ps -p $p -o comm= 2>/dev/null); "
    "if [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null; then "
    "case \"$c\" in \"{expect}\"*) echo \"RUNNING $p\";; *) echo STALE; esac; "
    "elif bash -c \"exec 3<>/dev/tcp/127.0.0.1/{port}\" 2>/dev/null; then echo BUSY; "
    "else echo NOT_RUNNING; fi"
)

REMOTE_STOP_CMD = (
    "d=$HOME/.llamaloader; p=$(cat $d/llama-server-{port}.pid 2>/dev/null); "
    "if [ -z \"$p\" ]; then echo NOPID; exit 0; fi; "
    "c=$(cat /proc/$p/comm 2>/dev/null || ps -p $p -o comm= 2>/dev/null); "
    "if ! kill -0 \"$p\" 2>/dev/null; then echo GONE; rm -f $d/llama-server-{port}.pid; exit 0; fi; "
    "case \"$c\" in "
    "\"{expect}\"*) kill \"$p\" && echo \"KILLED $p\";; "
    "*) echo \"REFUSED pid=$p comm=$c\";; esac"
)


def _remembered(host: str, ssh_port: int) -> dict | None:
    """The target this GUI actually launched (bin survives a form edit)."""
    r = running_server.get("remote")
    if r and r["host"] == host and int(r["ssh_port"]) == int(ssh_port):
        return r
    return None


def _remote_expect(t: dict) -> str:
    """Process-name the pidfile must belong to (basename of the launched bin).

    /proc/PID/comm is truncated to 15 chars by Linux, so the expected value is
    truncated identically and matched as a prefix. Placeholder/empty bins fall
    back to "llama", which is what a real llama-server reports as.
    """
    base = os.path.basename(str(t.get("bin") or "")).split(" ")[0]
    if not base or base == "true":
        base = "llama"
    return base[:15]


async def _remote_status(t: dict, listen_port: int) -> tuple[str, int | None]:
    """('running'|'busy'|'down', pid) as seen on the target."""
    rc, out, _err = await _run_ssh(
        t, REMOTE_STATUS_CMD.format(port=listen_port, expect=_remote_expect(t)), timeout=15)
    out = out.strip()
    if out.startswith("RUNNING"):
        try:
            return "running", int(out.split()[1])
        except (IndexError, ValueError):
            return "running", None
    if out == "BUSY":
        return "busy", None
    return "down", None



# ── Routes ───────────────────────────────────────────────────────

@app.get("/", response_class=RedirectResponse)
async def index():
    return "/gui"


@app.get("/gui")
async def gui():
    return Response(Path("templates/gui.html").read_text(), media_type="text/html")


@app.get("/api/models")
async def api_models():
    return JSONResponse(scan_models())


@app.get("/api/binaries")
async def api_binaries():
    return JSONResponse(scan_binaries())


@app.get("/api/profiles")
async def api_profiles():
    return JSONResponse(load_profiles())


@app.post("/api/profiles")
async def api_save_profile(cfg: Config):
    name = cfg.name.strip()
    if not name:
        return JSONResponse({"error": "Profile name must not be empty"}, status_code=400)
    profiles = load_profiles()
    profiles[name] = cfg.model_dump()
    save_profiles(profiles)
    return JSONResponse({"status": "saved", "name": name})


@app.delete("/api/profiles/{name}")
async def api_delete_profile(name: str):
    profiles = load_profiles()
    if name in profiles:
        del profiles[name]
        save_profiles(profiles)
    return JSONResponse({"status": "deleted"})


@app.post("/api/build-command")
async def api_build_command(cfg: Config):
    d = cfg.model_dump()
    argv = _verbatim_argv(d) or build_argv(d)
    t, err = _remote_target(d)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    if t:
        argv = [t["bin"], *argv[1:]]
        return JSONResponse({
            "command": f"ssh {shlex.quote(t['host'])} -p {t['ssh_port']} " + shlex.quote(_remote_launch_script([t["bin"], *argv[1:]], t, launch_env(d), d["port"])),
            "verbatim": _verbatim_argv(d) is not None,
            "remote": t["host"],
        })
    return JSONResponse({"command": render_command(argv, d), "verbatim": _verbatim_argv(d) is not None})


@app.get("/api/info")
async def api_info():
    return JSONResponse({"home": str(Path.home()), "default_binary": str(LLAMA_SERVER)})


@app.get("/api/binary-flags")
async def api_binary_flags(path: str):
    """What the SELECTED binary accepts (from its own --help, cached).

    The GUI execs this binary on launch anyway, so running `--help` adds no
    capability; it just tells the frontend which flags the build knows so it
    can warn about drift instead of silently failing at launch.
    """
    resolved = _resolve_server(path)
    try:
        data = binary_flags(resolved)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "--help timed out"}, status_code=502)
    except (ValueError, OSError):
        # Static messages only: the underlying error text (paths, OS details)
        # stays in the server log, never echoed to the API client.
        log.warning("binary-flags scan failed for a client-requested path", exc_info=True)
        return JSONResponse({"error": "flag scan failed: binary not readable or not a llama-server build"},
                            status_code=400)
    return JSONResponse({"path": resolved, "flags": data["flags"], "version": data["version"]})


@app.post("/api/ssh-test")
async def api_ssh_test(payload: dict):
    host = str(payload.get("remote_host") or "").strip()
    t, err = _remote_target({**payload, "remote_host": host})
    if err:
        return JSONResponse({"error": err}, status_code=400)
    if not t:
        return JSONResponse({"error": "Fill in the SSH target first"}, status_code=400)
    if not shutil.which("ssh"):
        return JSONResponse({"error": "ssh client not found on PATH"}, status_code=400)
    rc, out, err = await _run_ssh(t, "echo llamaloader-ok; uname -sr", timeout=15)
    if rc == 0 and "llamaloader-ok" in out:
        lines = out.strip().splitlines()
        return JSONResponse({"ok": True, "host": t["host"], "uname": lines[1] if len(lines) > 1 else ""})
    hint = ""
    if "Permission denied" in err or "Permission denied" in out:
        hint = "  -  key auth failed: add this machine's key to the target (ssh-copy-id) and/or load it in the agent"
    elif "timed out" in err:
        hint = "  -  host unreachable or sshd not answering"
    elif "Host key verification" in err:
        hint = "  -  unknown host key (the GUI accepts new ones automatically; a changed key must be removed from ~/.ssh/known_hosts by hand)"
    return JSONResponse({"ok": False, "error": (err or out or "ssh failed").strip()[:300] + hint}, status_code=400)


async def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True if something already accepts TCP connections on host:port."""
    if host in ("0.0.0.0", "::", "*"):
        host = "127.0.0.1"
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 0.25)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except (OSError, asyncio.TimeoutError):
        return False


@app.post("/api/launch")
async def api_launch(cfg: Config):
    global running_server
    if running_server["proc"] and running_server["proc"].poll() is None:
        return JSONResponse({"error": f"Server already running (PID {running_server['pid']})"}, status_code=409)

    d = cfg.model_dump()
    argv = _verbatim_argv(d) or build_argv(d)

    # ── Remote branch: detached launch over SSH ──
    t, err = _remote_target(d)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    if t:
        if not shutil.which("ssh"):
            return JSONResponse({"error": "ssh client not found on PATH"}, status_code=400)
        # A detached remote server leaves no local child to poll, so the usual
        # "already running" guard cannot see it  -  ask the target instead.
        state, prior = await _remote_status(t, d["port"])
        if state == "running":
            return JSONResponse({"error": f"Server already running on {t['host']} "
                                          f"(PID {prior})  -  Stop it first"}, status_code=409)
        if state == "busy":
            return JSONResponse({"error": f"Port {d['port']} is already in use on {t['host']} "
                                          f"(started outside this GUI)  -  free it or pick another port"},
                                status_code=409)
        argv = [t["bin"], *argv[1:]]
        script = _remote_launch_script(argv, t, launch_env(d), d["port"])
        rc, out, err = await _run_ssh(t, script, timeout=25)
        if rc != 0:
            return JSONResponse({"error": f"ssh launch failed ({rc}): {(err or out or '').strip()[:300]}"}, status_code=400)
        # read back the pid the script wrote on the target
        rc2, out2, _ = await _run_ssh(t, f"cat {_remote_files(t, d['port'])[1]} 2>/dev/null", timeout=10)
        pid = int(out2.strip()) if rc2 == 0 and out2.strip().isdigit() else None
        running_server = {"pid": pid, "proc": None, "remote": t}
        return JSONResponse({"pid": pid, "remote": t["host"], "command": render_command(argv, d)})

    wrapper = exec_wrapper(d)

    # Sanity checks  -  fail with a clear 400 instead of a llama-server that
    # dies one second after launch (or a Popen FileNotFoundError 500).
    if wrapper and not shutil.which(wrapper[0]):
        return JSONResponse({"error": f"Wrapper command not found on PATH: {wrapper[0]}"}, status_code=400)
    server_bin = Path(argv[0])
    if not server_bin.is_absolute():
        # Verbatim imports may name the binary bare or relative (./build/bin/llama-server);
        # Popen resolves it the same way (PATH / cwd), so check it the same way too.
        found = shutil.which(argv[0])
        if found:
            server_bin = Path(found)
    if not server_bin.is_file() or not os.access(server_bin, os.X_OK):
        return JSONResponse({"error": f"llama-server binary not found or not executable: {server_bin}"}, status_code=400)
    if not (d.get("hf_repo") or d.get("model_url")):
        model = Path(str(d.get("model", ""))).expanduser()
        if not str(model).endswith(".gguf") or not model.is_file():
            return JSONResponse({"error": f"Model file not found: {model} (or set --hf-repo/--model-url)"}, status_code=400)
    port = int(d.get("port") or 8080)
    if await _port_in_use(port, d.get("host") or "127.0.0.1"):
        return JSONResponse({"error": f"Port {port} is already in use  -  a server (possibly started outside this GUI) is listening. Stop it before launching from here."}, status_code=409)

    # Close a stale log handle from a previously crashed child.
    if running_server.get("log"):
        try:
            running_server["log"].close()
        except OSError:
            pass
    env = launch_env(d)
    merged_env = {**os.environ, **env} if env else None
    log_file = open("llama-server.log", "w")
    proc = subprocess.Popen([*wrapper, *argv], stdout=log_file, stderr=log_file, text=True, env=merged_env)
    running_server = {"pid": proc.pid, "proc": proc, "log": log_file}
    return JSONResponse({"pid": proc.pid, "command": render_command(argv, d)})


@app.post("/api/stop")
async def api_stop(request: Request):
    global running_server
    # Body carries the current form (remote target); old callers post nothing,
    # which must keep working as a plain local stop.
    d = {}
    try:
        body = await request.json()
        if isinstance(body, dict):
            d = body
    except Exception:
        pass
    # Remote: terminate by the pidfile this GUI (or a previous run of it) wrote.
    if d.get("remote_host"):
        t, err = _remote_target(d)
        if err:
            return JSONResponse({"error": err}, status_code=400)
    else:
        t = running_server.get("remote")
    if t and not t.get("bin"):
        mem = _remembered(t["host"], t["ssh_port"])
        if mem:
            t["bin"] = mem["bin"]
    elif t:
        mem = _remembered(t["host"], t["ssh_port"])
        # An empty remote_bin in the form means "the one I launched", not "llama".
        if mem and not str(d.get("remote_bin") or "").strip():
            t["bin"] = mem["bin"]
    if t and not running_server.get("proc"):
        port = int(d.get("port") or 8080)
        rc, out, err = await _run_ssh(
            t, REMOTE_STOP_CMD.format(port=port, expect=_remote_expect(t)), timeout=15)
        out = (out or "").strip()
        if out.startswith("KILLED"):
            return JSONResponse({"status": "stopped", "pid": out.split()[1], "remote": t["host"]})
        if out in ("NOPID", "GONE"):
            return JSONResponse({"status": "not running", "remote": t["host"]})
        if out.startswith("REFUSED"):
            # Pidfile pointed at a recycled PID  -  never kill it; tell the user.
            return JSONResponse({"error": f"stale pid file on target ({out}); "
                                          f"left untouched  -  remove ~/.llamaloader/llama-server-{port}.pid there"},
                                status_code=400)
        return JSONResponse({"status": "not running", "remote": t["host"], "detail": (err or out).strip()[:200]})
    if running_server["proc"] and running_server["proc"].poll() is None:
        running_server["proc"].terminate()
        if running_server.get("log"):
            running_server["log"].close()
        running_server = {"pid": None, "proc": None}
        return JSONResponse({"status": "stopped"})
    return JSONResponse({"status": "not running"})


@app.get("/api/status")
async def api_status(port: int = 8080, remote_host: str = "", remote_ssh_port: int = 22,
                     remote_bin: str = ""):
    """GUI-owned child takes precedence; otherwise probe the model port so an
    externally started llama-server (terminal, hermes, lm studio) is visible.

    Remote mode (remote_host set): status comes from the pidfile on the target;
    an 'external' server there cannot be distinguished from one started by
    hand, so the TCP probe runs from this machine against host:port."""
    global running_server
    if remote_host:
        t, err = _remote_target({"remote_host": remote_host, "remote_ssh_port": remote_ssh_port,
                                 "remote_bin": remote_bin})
        if err:
            return JSONResponse({"error": err}, status_code=400)
        if not str(remote_bin or "").strip():
            mem = _remembered(t["host"], t["ssh_port"])
            if mem:
                t["bin"] = mem["bin"]
        state, pid = await _remote_status(t, port)
        if state == "running":
            return JSONResponse({"status": "running", "pid": pid, "port_busy": True, "remote": t["host"]})
        if state == "busy":
            return JSONResponse({"status": "external", "port_busy": True, "remote": t["host"]})
        return JSONResponse({"status": "stopped", "port_busy": False, "remote": t["host"]})
    if running_server["proc"] and running_server["proc"].poll() is None:
        return JSONResponse({"status": "running", "pid": running_server["pid"], "port_busy": True})
    if await _port_in_use(port):
        return JSONResponse({"status": "external", "port_busy": True})
    return JSONResponse({"status": "stopped", "port_busy": False})


@app.get("/api/logs")
async def api_logs(limit: int = 50, remote_host: str = "", remote_ssh_port: int = 22,
                   port: int = 8080, remote_bin: str = ""):
    limit = max(1, min(int(limit), 1000))
    if remote_host:
        # Remote log lives on the target  -  tail it there and ship the text.
        t, err = _remote_target({"remote_host": remote_host, "remote_ssh_port": remote_ssh_port,
                                 "remote_bin": remote_bin})
        if err:
            return JSONResponse({"error": err}, status_code=400)
        log, _ = _remote_files(t, port)
        rc, out, err = await _run_ssh(t, f"tail -n {limit} {log} 2>/dev/null", timeout=15)
        if rc != 0:
            return JSONResponse({"lines": [], "error": err.strip()[:200]})
        return JSONResponse({"lines": [ln for ln in out.splitlines() if ln.strip()], "remote": t["host"]})
    log_file = Path("llama-server.log")
    if not log_file.exists():
        return JSONResponse({"lines": []})
    # Tail from the end of the file: reading a multi-MB log whole every few
    # seconds (old behaviour) is pure waste.
    with log_file.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - 65536)
        f.seek(start)
        data = f.read().decode("utf-8", errors="replace")
    lines = data.splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # drop the truncated first line
    return JSONResponse({"lines": [ln for ln in lines if ln.strip()][-limit:]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7890)
