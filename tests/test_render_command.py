# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 DTM-beep — https://github.com/DTM-beep
"""Preview-formatting tests for render_command() — display only, no network.

Why this exists: the preview emitted **one argv token per line**, and the log
rail is narrow. The result read as a zig-zag of stub lines ("--host" / "\" /
"127.0.0.1" / "\") under a wall of orphaned backslashes. It now puts a flag and
its value on ONE line, keeps the continuation valid shell, and drops the old
14-space indent pad.

Copy fidelity is the property that matters: the box is the one place a user
reads the command from, so `bash` must see exactly the argv it claims. Test
that with bash's own rule — backslash-newline is a *continuation*, which Python
shlex does NOT implement (it keeps "\\␊" as a literal newline), so the string
is flattened the way a shell would before round-tripping.

    .venv/bin/python tests/test_render_command.py
"""
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import render_command  # noqa: E402

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if not cond:
        FAIL += 1
    print(("  ok   " if cond else " FAIL  ") + name + (f"  <- {detail}" if detail and not cond else ""))


def bash_sees(cmd):
    """The tokens bash would exec: join `\\`-continuations, then split."""
    return shlex.split(cmd.replace("\\\n", ""))


def body_lines(cmd):
    """Stripped source lines without their trailing continuation."""
    return [l.strip().removesuffix("\\").strip() for l in cmd.split("\n")]


BIN = "/home/you/llama.cpp/build/bin/llama-server"
MODEL = ("/home/you/.lmstudio/models/AtomicChat/Qwen3.8-Flash-Next-GGUF/"
         "Qwen3.8-Flash-Next-AD4-27bpw-Q4_K_M-M64-00001-of-00033.gguf")
ARGV = [BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", "8080",
        "-ngl", "99", "--parallel", "1", "-t", "6", "--no-mmap",
        "--flash-attn", "on", "--temp", "0.7"]


def main():
    cmd = render_command(ARGV)
    lines = body_lines(cmd)

    check("binary leads the first line", lines[0] == BIN, lines[0])
    check("flag+value share a line", "--host 127.0.0.1" in cmd and "--port 8080" in cmd, cmd)
    check("short-flag form shares a line", "-ngl 99" in cmd and "-t 6" in cmd, cmd)
    # only a genuine value-less flag may stand alone
    lone = [l for l in lines[1:] if l in {"--host", "--port", "-ngl", "-t", "--parallel", "--temp", "--flash-attn", "-m"}]
    check("no orphaned flag-only line", lone == [], lone)
    check("bare --no-mmap keeps its own line", "--no-mmap" in lines, lines)
    # binary + 9 flag entries (was 15 lines with one token per line)
    check("one line per flag/value pair", len(lines) == 10, f"{len(lines)} lines: {lines}")
    check("old 14-space indent pad is gone", "\\\n              " not in cmd)
    for flag in ("--host", "--port", "--parallel", "--flash-attn", "--temp", "--no-mmap"):
        check(f"{flag} present exactly once", cmd.split().count(flag) == 1, cmd)

    # ── copy fidelity: what bash would exec == the argv we were given
    check("bash round-trip == original argv", bash_sees(cmd) == ARGV, bash_sees(cmd)[:6])
    check("long model path stays whole on one source line",
          any(MODEL in l for l in cmd.split("\n")), [l for l in cmd.split("\n") if ".gguf" in l])

    # env + wrapper belong to the preview — the copy must not lie
    cfg = {"env": "GGML_CUDA_NO_PINNED=1", "exec_prefix": "taskset -c 0-11"}
    both = render_command([BIN, "-m", MODEL], cfg)
    check("env assignment kept", "GGML_CUDA_NO_PINNED=1" in both, both)
    check("wrapper kept", "taskset -c 0-11" in both, both)
    check("env+wrapper+argv round-trip",
          bash_sees(both) == ["GGML_CUDA_NO_PINNED=1", "taskset", "-c", "0-11", BIN, "-m", MODEL],
          bash_sees(both)[:4])

    # degenerate inputs: no raise, no stray continuation
    check("empty argv -> empty string", render_command([]) == "", repr(render_command([])))
    solo = render_command([BIN])
    check("binary only -> single line, no continuation", solo == BIN and "\\" not in solo, solo)
    check("flag without value keeps its own line",
          render_command([BIN, "--no-mmap"]) == f"{BIN} \\\n  --no-mmap", render_command([BIN, "--no-mmap"]))
    for argv in ([BIN, "-t", "-1"],                     # negative value (quoted by shlex)
                 [BIN, "-m", "-weird path.gguf"],        # value that looks like a flag
                 [BIN, "--chat-template", "{% if x %}a{% endif %}"],  # template with spaces
                 [BIN, "--alias", "qwen; rm -rf /"],     # injection-shaped value
                 ["taskset", "-c", "0-11", BIN, "-m", MODEL]):   # wrapper inside argv
        out = render_command(argv)
        check(f"round-trip: {' '.join(argv[1:4])[:40]}", bash_sees(out) == argv, out.replace("\n", " | ")[:110])
    neg = render_command([BIN, "--n-predict", "-1", "-t", "6"])
    check("negative value is not orphaned onto its own line",
          "--n-predict -1" in neg, neg.replace("\n", " | "))
    check("negative value still round-trips", bash_sees(neg) == [BIN, "--n-predict", "-1", "-t", "6"], neg)

    inj = render_command([BIN, "--alias", "qwen; rm -rf /"])
    check("injection-shaped value is quoted, never bare", "qwen; rm -rf /" not in inj or "'qwen; rm -rf /'" in inj, inj)

    print("\n" + ("ALL PASS" if FAIL == 0 else f"{FAIL} FAILURE(S)"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
