# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 DTM-beep  -  https://github.com/DTM-beep
"""Preview-renderer tests for renderCommandPreview()  -  needs node, no network.

Why this exists: the Command Preview box is narrow (the log rail can be 360px)
and model paths run 130+ chars, so plain `pre-wrap` shredded every long token
mid-word  -  "the lines are all broken, make use of the empty space". The fix
breaks at '/' , '=' and ':' via <wbr> elements. <wbr> is the whole point: it is
an invisible *break opportunity* that adds **no character**, so selecting or
clicking Copy still yields the byte-exact command. That fidelity property is
what this file pins, plus "no markup ever built from the command text" (a
pasted path may contain `<`, so the renderer uses text nodes only).

The function lives in templates/gui.html (frontend, no DOM at test time), so a
tiny DOM stand-in with real textContent semantics (descendant concatenation)
drives it, same extraction idea as test_parse_core.py.

    .venv/bin/python tests/test_preview_render.py        # needs node
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


STUB = """
class Node {
  constructor(name, text){ this.name=name; this._text=text||''; this.className=''; this.kids=[]; }
  appendChild(c){ this.kids.push(c); return c; }
  // real DOM: textContent concatenates all descendant text, <wbr> contributes none
  get textContent(){ const f=n=> (n.kids.length ? n.kids.map(f).join('') : n._text); return this.kids.map(f).join(''); }
  set textContent(v){ this.kids=[]; this._text=v||''; }
}
const document={ createElement(t){ return new Node(t); }, createTextNode(s){ return new Node('#t', s); } };
const serialize = n => (n.name === 'wbr' ? '<wbr>' : '') + n.kids.map(serialize).join('');
"""

BIN = "/home/you/llama.cpp/build/bin/llama-server"
MODEL = ("/home/you/.lmstudio/models/AtomicChat/Qwen3.8-Flash-Next-GGUF/"
         "Qwen3.8-Flash-Next-AD4-27bpw-Q4_K_M-M64-00001-of-00033.gguf")

CASES = {
    "grouped preview": f"{BIN} \\\n  -m {MODEL} \\\n  --host 127.0.0.1 \\\n  --no-mmap \\\n  --temp 0.7",
    "remote ssh single line": "ssh user@host -p 22 'cd ~ && nohup ./llama-server -m /models/x.gguf --port 8080 > log 2>&1 &'",
    "quoted value with spaces": "/bin/llama-server --alias 'qwen 36' --temp 0.7",
    "hostile markup in value": '/bin/x --alias "<img src=x onerror=alert(1)>" --temp 1',
    "env + wrapper prefix": "GGML_CUDA_NO_PINNED=1 taskset -c 0-11 /bin/x -m /models/x.gguf",
    "single token": "/bin/llama-server",
    "empty command": "",
}

HARNESS = """
const out = [];
for (const [name, cmd] of Object.entries(CASES)) {
  const el = new Node('div');
  try { renderCommandPreview(el, cmd); } catch (e) { out.push({name, error: String(e)}); continue; }
  out.push({
    name,
    copy: el.textContent,
    breaks: (serialize(el).match(/<wbr>/g) || []).length,
    classes: (function w(n, acc){ if (n.className) acc.push(n.className); n.kids.forEach(c => w(c, acc)); return acc; })(el, []),
  });
}
process.stdout.write(JSON.stringify(out));
"""


def main():
    node = shutil.which("node")
    if not node:
        print("SKIP: node not on PATH (renderer is JS)")
        return 0

    src = open(TEMPLATE, encoding="utf-8").read()
    m = re.search(r"(function renderCommandPreview\(el, cmd\) \{.*?\n\})", src, re.S)
    check("renderCommandPreview found in template", bool(m))
    if not m:
        return 1

    script = STUB + "\n" + m.group(1) + "\nconst CASES = " + json.dumps(CASES) + ";" + HARNESS
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

    check("all cases returned", len(res) == len(CASES), sorted(set(CASES) - set(res)))

    for name, cmd in CASES.items():
        r = res.get(name)
        if not r:
            continue
        check(f"{name}: no error", "error" not in r, r.get("error"))
        # THE property: what a copy contains, character for character
        check(f"{name}: copied text == exact command", r["copy"] == cmd,
              f"in={cmd!r} out={r['copy']!r}")

    grouped = res["grouped preview"]
    check("long paths get break points at '/'", grouped["breaks"] >= 10, grouped["breaks"])
    check("flags are styled apart from values",
          "c-flag" in grouped["classes"] and "c-arg" in grouped["classes"], grouped["classes"])
    check("continuations are styled as de-emphasis", "c-cont" in grouped["classes"], grouped["classes"])

    hostile = res["hostile markup in value"]
    check("hostile value creates no element from the text",
          all(c in ("c-flag", "c-arg", "c-cont") for c in hostile["classes"]), hostile["classes"])

    print("\n" + ("ALL PASS" if FAIL == 0 else f"{FAIL} FAILURE(S)"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
