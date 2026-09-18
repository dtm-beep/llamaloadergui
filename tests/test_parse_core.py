# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 DTM-beep — https://github.com/DTM-beep
"""Parser tests for the paste-a-command import path — needs node, no network.

Why this exists: pasting a command straight from a terminal (prompt glyph and
all — "❯ llama-server -m …" from fish / powerlevel10k / oh-my-posh, "$ …" from
bash, "PS C:\\>" from PowerShell) used to make the *prompt* the binary: the
first non-flag token wins, so argv[0] became "❯", the env assignments and the
taskset wrapper fell into the flag loop instead of the Env/Wrapper fields, and
a launch tried to exec "❯". Leading prompt markers are now dropped before
anything else is parsed.

The parser lives in templates/gui.html between the PARSE-CORE markers (JS, no
DOM). It is extracted here and driven under node, so the tested code is the
code that ships.

    .venv/bin/python tests/test_parse_core.py        # needs node
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "templates", "gui.html")

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if not cond:
        FAIL += 1
    print(("  ok   " if cond else " FAIL  ") + name + (f"  <- {detail}" if detail and not cond else ""))


HARNESS = """
const out = [];
for (const [name, cmd] of CASES) {
  let r;
  try { r = parseLlamaCommand(cmd); } catch (e) { out.push({name, error: String(e)}); continue; }
  const argv = (r.server ? [r.server] : []).concat(r.rawArgv || []);
  out.push({
    name,
    server: r.server,
    env: r.cfg.env || '',
    wrapper: r.cfg.exec_prefix || '',
    argv0: argv[0] || null,
    argv: argv,
    ngl: ('ngl' in r.cfg) ? r.cfg.ngl : null,
    raw: (r.rawArgv || []).join(' ')
  });
}
process.stdout.write(JSON.stringify(out));
"""

BIN = "/home/you/llama.cpp/build/bin/llama-server"
PLAIN = f"{BIN} -m /models/x.gguf -c 129072 -b 2048 -fa on --temp 0.7"
WITH_PRE = (f"LLAMA_ATTN_ROT_DISABLE=1 GGML_CUDA_NO_PINNED=1 taskset -c 0-11 "
            f"{BIN} -m /models/x.gguf -c 129072 --alias qwen")

MULTILINE = f"{BIN} \\\n  -m /models/x.gguf \\\n  -c 32768 \\\n  --host 0.0.0.0\n"


def main():
    node = shutil.which("node")
    if not node:
        print("SKIP: node not on PATH (parser is JS)")
        return 0

    src = open(TEMPLATE, encoding="utf-8").read()
    m = re.search(r"// ===PARSE-CORE-START===(.*?)// ===PARSE-CORE-END===", src, re.S)
    check("parse core found in template", bool(m))
    if not m:
        return 1

    cases = [
        ["plain", PLAIN],
        ["env+wrapper clean", WITH_PRE],
        # the regression this file exists for
        ["fish prompt", "❯ " + PLAIN],
        ["powerlevel10k prompt", "➜ " + WITH_PRE],
        ["bash prompt", "$ " + PLAIN],
        ["user@host prompt", "user@host:~$ " + PLAIN],
        ["shell prompt with env+wrapper", "❯ " + WITH_PRE],
        ["powershell prompt", "PS C:\\> " + PLAIN],
        ["multiline continuation", MULTILINE],
        ["relative binary", "./build/bin/llama-server -m x.gguf"],
    ]

    script = m.group(1) + "\nconst CASES = " + json.dumps(cases) + ";" + HARNESS
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        p = subprocess.run([node, path], capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            check("node harness runs", False, (p.stderr or "")[:300])
            return 1
        res = {r["name"]: r for r in json.loads(p.stdout)}
    finally:
        os.unlink(path)

    check("all cases returned", len(res) == len(cases), sorted(set(r[0] for r in cases) - set(res)))

    ref = res["plain"]
    check("plain: binary detected", ref["server"] == BIN, ref["server"])

    pre_ref = res["env+wrapper clean"]
    check("env+wrapper: binary detected", pre_ref["server"] == BIN, pre_ref["server"])
    # every prompt flavour must land on exactly the same argv as its clean paste
    for name, base in [("fish prompt", ref), ("bash prompt", ref),
                       ("user@host prompt", ref), ("powershell prompt", ref),
                       ("powerlevel10k prompt", pre_ref),
                       ("shell prompt with env+wrapper", pre_ref)]:
        r = res[name]
        check(f"{name}: argv identical to clean paste",
              r["argv"] == base["argv"], f"argv0={r['argv0']!r}")
        check(f"{name}: no prompt glyph anywhere in argv",
              all(not re.fullmatch(r"[\$%>›»❯➜→]+", t) for t in r["argv"]),
              r["argv"][:2])

    # env / wrapper hoisting must survive a prompt (they are launch facts, not flags)
    pre = res["shell prompt with env+wrapper"]
    check("env+wrapper hoisted without a prompt too",
          pre_ref["env"] == "LLAMA_ATTN_ROT_DISABLE=1 GGML_CUDA_NO_PINNED=1" and pre_ref["wrapper"] == "taskset -c 0-11",
          f"env={pre_ref['env']!r} wrapper={pre_ref['wrapper']!r}")
    check("env hoisted past a prompt", pre["env"] == "LLAMA_ATTN_ROT_DISABLE=1 GGML_CUDA_NO_PINNED=1", pre["env"])
    check("taskset wrapper hoisted past a prompt", pre["wrapper"] == "taskset -c 0-11", pre["wrapper"])

    # verbatim guarantee: nothing the pasted command never had
    check("no --n-gpu-layers invented", "--n-gpu-layers" not in ref["raw"] and "-ngl" not in ref["raw"], ref["raw"])
    check("no -ctk/-ctv invented", "-ctk" not in ref["raw"] and "-ctv" not in ref["raw"], ref["raw"])
    check("context reproduced", "129072" in ref["raw"], ref["raw"])

    ml = res["multiline continuation"]
    check("line continuations flatten", ml["argv0"] == BIN and "32768" in ml["raw"], ml["argv"])

    rel = res["relative binary"]
    check("relative binary kept as typed", rel["argv0"] == "./build/bin/llama-server", rel["argv0"])

    print("\n" + ("ALL PASS" if FAIL == 0 else f"{FAIL} FAILURE(S)"))
    return 1 if FAIL else 0


sys.exit(main())
