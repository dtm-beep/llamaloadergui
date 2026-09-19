# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 DTM-beep  -  https://github.com/DTM-beep
"""Direct tests for the remote (SSH) code paths  -  no ssh, no network.

server._run_ssh is replaced with an in-process emulator of the target machine
(pidfile, /proc/<pid>/comm identity check, port-busy probe). Endpoints are
called as plain coroutines, so this needs no test dependencies at all:

    .venv/bin/python tests/test_remote.py
"""
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    msg = "  ok   " if cond else "  FAIL "
    print(msg + name + (("  -  " + str(detail)[:220]) if detail and not cond else ""))
    if not cond:
        FAIL += 1


def body(resp):
    return json.loads(resp.body.decode())


def status_of(resp):
    return resp.status_code


# ── fake target ─────────────────────────────────────────────────────────
class FakeTarget:
    """State of the 'remote' machine: pidfile contents + what comm each pid has."""

    def __init__(self):
        self.pidfiles = {}          # listen port -> pid
        self.comms = {}             # pid -> comm
        self.busy_ports = set()
        self.launched = []          # remote launch scripts, in order
        self.next_pid = 500

    async def run_ssh(self, t, remote_cmd, timeout=20):
        self.launched.append((t, remote_cmd))
        if "nohup" in remote_cmd:
            pid = self.next_pid
            self.next_pid += 1
            port = re.search(r"llama-server-(\d+)\.pid", remote_cmd)
            m = re.search(r"nohup (.+?) >", remote_cmd)
            toks = (m.group(1) if m else "llama-server").split(" ")
            # comm = basename of argv[0]; strip any env assignments that precede it
            toks = [x for x in toks if "=" not in x.split("/")[0]]
            binname = os.path.basename(toks[0].strip("'\"") if toks else "llama-server")
            self.pidfiles[int(port.group(1))] = pid
            self.comms[pid] = binname[:15]        # Linux comm truncation
            return 0, "", ""
        m = re.match(r"cat ~/\.llamaloader/llama-server-(\d+)\.pid", remote_cmd)
        if m:
            p = self.pidfiles.get(int(m.group(1)))
            return (0, f"{p}\n", "") if p else (1, "", "")
        if "llama-server-" in remote_cmd and "tail" in remote_cmd:
            return 0, "line one\nline two\n", ""
        if "/proc/$p/comm" in remote_cmd and "kill \"$p\"" not in remote_cmd:
            # status command  -  must honour the same identity guard as the real
            # shell: alive AND comm matches the expected prefix.
            port = int(re.search(r"llama-server-(\d+)\.pid", remote_cmd).group(1))
            expect_txt = server._remote_expect(t)   # same rule the app applies
            p = self.pidfiles.get(port)
            c = self.comms.get(p, "")
            if p and c and c.startswith(expect_txt):
                return 0, f"RUNNING {p}\n", ""
            if port in self.busy_ports:
                return 0, "BUSY\n", ""
            return 0, "NOT_RUNNING\n", ""
        if 'kill "$p"' in remote_cmd:            # stop command
            port = int(re.search(r"llama-server-(\d+)\.pid", remote_cmd).group(1))
            p = self.pidfiles.get(port)
            if not p:
                return 0, "NOPID\n", ""
            c = self.comms.get(p, "")
            if c and c.startswith(server._remote_expect(t)):
                del self.pidfiles[port]
                return 0, f"KILLED {p}\n", ""
            return 0, f"REFUSED pid={p} comm={c}\n", ""
        return 0, "", ""


def install(fake):
    server._run_ssh = fake.run_ssh
    server.running_server = {"pid": None, "proc": None}


async def main():
    cfg = dict(model="~/m/x.gguf", port=8080, remote_host="u@10.0.0.9",
               remote_ssh_port=22, remote_bin="~/llama.cpp/build/bin/llama-server",
               remote_workdir="~/srv")

    # ── launch ──
    fake = FakeTarget()
    install(fake)
    r = await server.api_launch(server.Config(**cfg))
    check("launch: 200", status_of(r) == 200, body(r))
    check("launch: reports remote pid", body(r).get("pid") == 500, body(r))
    check("launch: remote host echoed", body(r).get("remote") == "u@10.0.0.9")
    t, script = next(x for x in fake.launched if "nohup" in x[1])
    check("launch: nohup + detach", "nohup" in script and "< /dev/null" in script)
    check("launch: pidfile keyed by listen port", "llama-server-8080.pid" in script)
    check("launch: tilde bin NOT quoted (expands remotely)", "~/llama.cpp" in script and "'~/" not in script)
    check("launch: workdir cd", "cd ~/srv" in script or "cd ~" in script)
    check("launch: host never interpolated raw", "$(" not in script and ";" not in t["host"])

    # ── double launch must be refused (detached server is invisible locally) ──
    r = await server.api_launch(server.Config(**cfg))
    check("double launch: 409", status_of(r) == 409, body(r))
    check("double launch: names the PID", "500" in body(r).get("error", ""), body(r))

    # ── status ──
    r = await server.api_status(port=8080, remote_host="u@10.0.0.9", remote_ssh_port=22,
                                remote_bin="~/llama.cpp/build/bin/llama-server")
    check("status: running", body(r)["status"] == "running" and body(r)["pid"] == 500, body(r))

    # ── status with the bin field cleared falls back to the remembered bin ──
    r = await server.api_status(port=8080, remote_host="u@10.0.0.9", remote_ssh_port=22, remote_bin="")
    check("status: remembered bin still 'running'", body(r)["status"] == "running", body(r))

    # ── port busy on target, nothing in our pidfile → external ──
    fake.pidfiles.pop(8080, None)
    fake.busy_ports.add(8080)
    r = await server.api_status(port=8080, remote_host="u@10.0.0.9", remote_ssh_port=22, remote_bin="")
    check("status: external on target", body(r)["status"] == "external", body(r))
    r = await server.api_launch(server.Config(**cfg))
    check("launch onto busy port: 409", status_of(r) == 409, body(r))
    fake.busy_ports.clear()

    # ── pidfile reuse: pid now belongs to something else → refuse to kill ──
    fake.pidfiles[8080] = 77
    fake.comms[77] = "postgres"
    server.running_server = {"pid": None, "proc": None}
    r = await server.api_status(port=8080, remote_host="u@10.0.0.9", remote_ssh_port=22, remote_bin="")
    check("reuse: stale pid not reported running", body(r)["status"] != "running", body(r))

    class FakeReq:
        async def json(self):
            return dict(cfg)

    r = await server.api_stop(FakeReq())
    check("reuse: stop refuses to kill foreign pid", status_of(r) == 400 and "REFUSED" in body(r).get("error", ""), body(r))
    check("reuse: foreign pid untouched", fake.pidfiles.get(8080) == 77, fake.pidfiles)

    # ── stop of our own process ──
    fake.pidfiles[8080] = 88
    fake.comms[88] = "llama-server"
    server.running_server = {"pid": 88, "proc": None,
                             "remote": {"host": "u@10.0.0.9", "ssh_port": 22,
                                        "bin": "~/llama.cpp/build/bin/llama-server", "workdir": None}}
    r = await server.api_stop(FakeReq())
    check("stop: kills our own remote pid", body(r).get("status") == "stopped", body(r))
    check("stop: pidfile cleared on target", 8080 not in fake.pidfiles, fake.pidfiles)

    # ── logs come from the target ──
    r = await server.api_logs(limit=10, remote_host="u@10.0.0.9", remote_ssh_port=22, port=8080,
                              remote_bin="~/llama.cpp/build/bin/llama-server")
    check("logs: tailed remotely", body(r).get("lines") == ["line one", "line two"], body(r))

    # ── injection attempts rejected before ssh is ever called ──
    for bad in ["host; rm -rf /", "$(curl evil)", "`x`", "host\nnewline", "|sh", "a&&b"]:
        try:
            server._remote_target({"remote_host": bad, "remote_bin": "/bin/true"})
            check(f"injection rejected: {bad!r}", False)
        except ValueError:
            check(f"injection rejected: {bad!r}", True)

    # ── local mode untouched ──
    install(FakeTarget())
    check("local: no remote target when host empty", server._remote_target({"remote_host": ""}) is None)

    print("\n" + ("ALL PASS" if FAIL == 0 else f"{FAIL} FAILURE(S)"))
    return 1 if FAIL else 0


sys.exit(asyncio.run(main()))
