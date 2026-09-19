# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 DTM-beep — https://github.com/DTM-beep
"""Tests for the per-build flag scan and the 2026-09-19 field sync.

Two halves:

1. Python: parse_help_flags() must read real `llama-server --help` shapes
   (alias chains, metavars, wrapped descriptions), binary_flags() must exec a
   binary, cache by path+mtime, and refuse junk output. build_argv() must stay
   NEUTRAL for the new security/serving fields: a stock profile emits none of
   them, explicit values emit exactly their flags.

2. node: the shipped PARSE-CORE must map imported commands back onto the new
   form fields (--api-key, --no-cache-prompt, -sysf, ...).

    .venv/bin/python tests/test_binary_flags.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server import binary_flags, build_argv, parse_help_flags  # noqa: E402

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if not cond:
        FAIL += 1
    print(("  ok   " if cond else " FAIL  ") + name + (f"  <- {detail}" if detail and not cond else ""))


# Mirrors the real formatting: aliases comma-joined, metavar between the last
# flag token and the description column, wrapped desc lines that must be
# ignored, negation pairs, brackets/ranges as metavars.
HELP_FIXTURE = """usage: llama-server [options]

common:
  -h,    --help, --usage                    show usage and exit
  -t,    --threads N                        number of threads to use for generation
        --metrics                           show metrics endpoint
        --no-metrics                        hide metrics endpoint
        --api-key KEY                       API key required for all HTTP requests
        --api-key-file FNAME                path to file containing API key
        --ssl-key-file FNAME                path to file containing private key in
                                            PEM format (wrapped, must be ignored)
        --sleep-idle-seconds SECONDS        number of seconds an idle slot is swapped out (default: -1)
        --cpu-strict <0|1>                  use strict CPU placement
        --pooling {none,mean,cls,last,rank} pooling type
        --temp RANGE                        temperature range lo-hi
        --repack, -r                        repack tensors on quantization
        --chat-template-regex REGEX         a description that mentions --temp
                                             and --top-k as prose, not options
"""


def test_parse_help():
    flags = parse_help_flags(HELP_FIXTURE)
    check("alias chain: all three longs", all(n in flags for n in ("help", "usage")), flags.keys())
    check("threads takes a value", flags.get("threads") is True)
    check("metrics is boolean", flags.get("metrics") is False)
    check("negation pair recorded", "no-metrics" in flags)
    check("api-key takes a value", flags.get("api-key") is True)
    check("ssl-key-file takes a value", flags.get("ssl-key-file") is True)
    check("sleep-idle-seconds takes a value", flags.get("sleep-idle-seconds") is True)
    check("bracket metavar <0|1>", flags.get("cpu-strict") is True)
    check("brace metavar {none,...}", flags.get("pooling") is True)
    check("repack is boolean (short alias consumed)", flags.get("repack") is False)
    check("prose flags in descriptions ignored", "--temp" not in flags and "top-k" not in (flags or {}))
    check("wrapped desc line not a flag", "PEM" not in flags and "format" not in flags)
    check("short-only names not recorded", "h" not in flags and "t" not in flags)


def test_binary_flags(tmpdir):
    real = tmpdir + "/fake-server.sh"
    with open(real, "w") as fh:
        fh.write("#!/bin/sh\nprintf '%s' '" + HELP_FIXTURE.replace("'", "'\\''") + "'\n")
    os.chmod(real, 0o755)
    res = binary_flags(real, min_flags=5)
    check("exec fake binary, flags parsed", res["flags"].get("api-key") is True)
    check("cache hit returns same object", binary_flags(real, min_flags=5) is res)
    try:
        binary_flags(tmpdir + "/nope", min_flags=5)
        check("missing binary raises", False)
    except ValueError:
        check("missing binary raises", True)
    junk = tmpdir + "/junk.sh"
    with open(junk, "w") as fh:
        fh.write("#!/bin/sh\necho hello\n")
    os.chmod(junk, 0o755)
    try:
        binary_flags(junk, min_flags=5)
        check("junk --help rejected", False)
    except ValueError:
        check("junk --help rejected", True)


NEW_KEYS = {
    "api_key": "sekret", "api_key_file": "/run/secrets/k", "api_prefix": "/llama",
    "ssl_key_file": "/etc/ssl/k.pem", "ssl_cert_file": "/etc/ssl/c.pem",
    "cors_origins": "http://127.0.0.1:3000", "cors_headers": "X-Test",
    "cors_credentials": True, "metrics": True,
    "sleep_idle_seconds": "300", "threads_http": "8", "cache_reuse": "512",
    "system_prompt_file": "/tmp/sys.txt",
    "models_dir": "/models", "models_max": "4",
}


def test_argv():
    base = build_argv({"model": "/m/x.gguf"})
    for flag in ("--api-key", "--api-key-file", "--api-prefix", "--ssl-key-file",
                 "--ssl-cert-file", "--cors-origins", "--cors-headers", "--cors-credentials",
                 "--metrics", "--sleep-idle-seconds", "--threads-http", "--no-cache-prompt",
                 "--cache-reuse", "--system-prompt-file", "--models-dir",
                 "--models-max", "--no-models-autoload"):
        check(f"neutral default hides {flag}", flag not in base)
    cfg = {"model": "/m/x.gguf", "cache_prompt": False, "models_autoload": False, **NEW_KEYS}
    argv = build_argv(cfg)
    joined = " ".join(argv)
    check("--api-key emitted with value", "--api-key sekret" in joined)
    check("--api-prefix emitted", "--api-prefix /llama" in joined)
    check("--ssl pair emitted", "--ssl-key-file /etc/ssl/k.pem" in joined and "--ssl-cert-file /etc/ssl/c.pem" in joined)
    check("--cors-origins emitted", "--cors-origins http://127.0.0.1:3000" in joined)
    check("--cors-credentials emitted", "--cors-credentials" in argv)
    check("--metrics emitted", "--metrics" in argv)
    check("--sleep-idle-seconds emitted", "--sleep-idle-seconds 300" in joined)
    check("--threads-http emitted", "--threads-http 8" in joined)
    check("--no-cache-prompt from cache_prompt=False", "--no-cache-prompt" in argv)
    check("--cache-reuse emitted", "--cache-reuse 512" in joined)
    check("--system-prompt-file emitted", "--system-prompt-file /tmp/sys.txt" in joined)
    check("--models-dir emitted", "--models-dir /models" in joined)
    check("--models-max emitted", "--models-max 4" in joined)
    check("--no-models-autoload from models_autoload=False", "--no-models-autoload" in argv)


PARSE_HARNESS = """
const out = [];
for (const [name, cmd] of CASES) {
  let r;
  try { r = parseLlamaCommand(cmd); } catch (e) { out.push({name, error: String(e)}); continue; }
  out.push({name, cfg: r.cfg, unknown: r.unknown || []});
}
process.stdout.write(JSON.stringify(out));
"""


def test_parse_core_roundtrip():
    node = shutil.which("node")
    if not node:
        print("SKIP: node not on PATH (parser is JS)")
        return
    src = open(os.path.join(ROOT, "templates", "gui.html"), encoding="utf-8").read()
    m = re.search(r"// ===PARSE-CORE-START===(.*?)// ===PARSE-CORE-END===", src, re.S)
    check("parse core found in template", bool(m))
    if not m:
        return
    binp = "/home/you/llama.cpp/build/bin/llama-server"
    cmd = (f"{binp} -m /m/x.gguf --api-key sekret --api-key-file /run/secrets/k "
           f"--api-prefix /llama --ssl-key-file /etc/ssl/k.pem --ssl-cert-file /etc/ssl/c.pem "
           f"--cors-origins http://127.0.0.1:3000 --cors-headers X-Test --cors-credentials "
           f"--metrics --sleep-idle-seconds 300 --threads-http 8 --no-cache-prompt "
           f"--cache-reuse 512 -sysf /tmp/sys.txt "
           f"--models-dir /models --models-max 4 --no-models-autoload")
    cases = [["new flags", cmd]]
    script = m.group(1) + "\nconst CASES = " + json.dumps(cases) + ";" + PARSE_HARNESS
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        proc = subprocess.run([node, path], capture_output=True, text=True, timeout=30)
        os.unlink(path)
    finally:
        pass
    if proc.returncode != 0:
        check("node parse run", False, proc.stderr[:300])
        return
    res = json.loads(proc.stdout)[0]
    check("no parse error", "error" not in res, res.get("error", ""))
    cfg = res["cfg"]
    exp = {"api_key": "sekret", "api_key_file": "/run/secrets/k", "api_prefix": "/llama",
           "ssl_key_file": "/etc/ssl/k.pem", "ssl_cert_file": "/etc/ssl/c.pem",
           "cors_origins": "http://127.0.0.1:3000", "cors_headers": "X-Test",
           "metrics": True, "cors_credentials": True, "cache_prompt": False,
           "models_autoload": False}
    for k, v in exp.items():
        check(f"import maps {k}", cfg.get(k) == v, repr(cfg.get(k)))
    check("import maps sleep_idle_seconds", cfg.get("sleep_idle_seconds") == "300", repr(cfg.get("sleep_idle_seconds")))
    check("import maps threads_http", cfg.get("threads_http") == "8", repr(cfg.get("threads_http")))
    check("import maps cache_reuse", cfg.get("cache_reuse") == "512", repr(cfg.get("cache_reuse")))
    check("-sysf maps to system_prompt_file", cfg.get("system_prompt_file") == "/tmp/sys.txt", repr(cfg.get("system_prompt_file")))
    check("models-dir / models-max mapped", cfg.get("models_dir") == "/models" and cfg.get("models_max") == "4")
    # everything must land on fields, nothing dumped to extra args
    check("no leftovers in extra_args", not cfg.get("extra_args"), cfg.get("extra_args", ""))
    check("no unknown flags reported", not res["unknown"], res["unknown"])


def main():
    tmpdir = tempfile.mkdtemp(prefix="llg-bflags-")
    test_parse_help()
    test_binary_flags(tmpdir)
    test_argv()
    test_parse_core_roundtrip()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nFAILURES: " + str(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
