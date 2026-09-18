#!/usr/bin/env python3
"""Llama Loader GUI — scan models, save profiles, launch llama-server."""

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse, JSONResponse, RedirectResponse
from typing import Optional
from pydantic import BaseModel, Field

# No CORS middleware on purpose: the UI is same-origin, and a wildcard
# ("allow_origins=['*']") would let any webpage in the user's browser POST
# to this launch-an-arbitrary-process endpoint. Same-origin + JSON body
# preflight keeps the attack surface at "the GUI itself".
app = FastAPI()

LMSTUDIO_DIR = Path.home() / ".lmstudio" / "models"
LLAMA_SERVER = Path("/home/you/llama.cpp/build/bin/llama-server")
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
# llama-server binary to identify it (no --version probe) — metadata only.
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
                # label: "llamaDTM / build-mtp" — repo name + build dir
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
        # Bare boolean flag (valueless) — always emit the flag itself
        tokens.append(flag)


def launch_env(cfg: dict):
    """Extra environment variables for the child process ('K=V K=V' / JSON).

    Env assignments and wrappers (taskset, nice) are how the server gets
    started, not llama-server flags — an importer that drops them silently
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


def _resolve_server(raw) -> str:
    """Normalize the llama-server binary path (empty/relative -> known default)."""
    raw = str(raw or "").strip()
    if not raw:
        return str(LLAMA_SERVER)
    p = Path(raw).expanduser()
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
    """
    prefix = []
    if cfg is not None:
        env = launch_env(cfg)
        if env:
            prefix.extend(f"{k}={v}" for k, v in env.items())
        prefix.extend(exec_wrapper(cfg))
    body = " \\\n              ".join(shlex.quote(str(t)) for t in argv)
    return (" ".join(shlex.quote(str(p)) for p in prefix) + " " if prefix else "") + body


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
    return JSONResponse({"command": render_command(argv, d), "verbatim": _verbatim_argv(d) is not None})


@app.get("/api/info")
async def api_info():
    return JSONResponse({"home": str(Path.home()), "default_binary": str(LLAMA_SERVER)})


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
    wrapper = exec_wrapper(d)

    # Sanity checks — fail with a clear 400 instead of a llama-server that
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
        return JSONResponse({"error": f"Port {port} is already in use — a server (possibly started outside this GUI) is listening. Stop it before launching from here."}, status_code=409)

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
async def api_stop():
    global running_server
    if running_server["proc"] and running_server["proc"].poll() is None:
        running_server["proc"].terminate()
        if running_server.get("log"):
            running_server["log"].close()
        running_server = {"pid": None, "proc": None}
        return JSONResponse({"status": "stopped"})
    return JSONResponse({"status": "not running"})


@app.get("/api/status")
async def api_status(port: int = 8080):
    """GUI-owned child takes precedence; otherwise probe the model port so an
    externally started llama-server (terminal, hermes, lm studio) is visible."""
    global running_server
    if running_server["proc"] and running_server["proc"].poll() is None:
        return JSONResponse({"status": "running", "pid": running_server["pid"], "port_busy": True})
    if await _port_in_use(port):
        return JSONResponse({"status": "external", "port_busy": True})
    return JSONResponse({"status": "stopped", "port_busy": False})


@app.get("/api/logs")
async def api_logs(limit: int = 50):
    limit = max(1, min(int(limit), 1000))
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
