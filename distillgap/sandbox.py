"""Operator-owned execution of untrusted candidate code (references, validator and target programs).

Isolation (macOS, used for the pilot): base interpreter with -I -S -B, scrubbed
environment, fresh empty working directory, sandbox-exec profile that denies
network, all file writes, process forking and reads under /Users and /Volumes
(so repository secrets such as server/.env are unreadable), CPU/file/fd rlimits,
a wall-clock kill and an RSS watchdog. On Linux the same rlimits apply plus
RLIMIT_AS; there is no filesystem sandbox, which is documented as weaker.
The child never receives expected outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

CHILD_SOURCE = (Path(__file__).with_name("sandbox_child.py")).read_text()
SANDBOX_VERSION = "distillgap_sandbox_v1"
MAX_RSS_BYTES = 1_500_000_000

MAC_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (literal "/dev/null"))
(deny process-fork)
(deny file-read* (subpath "/Users") (subpath "/Volumes"))
"""


def base_python() -> str:
    """The interpreter that runs untrusted code: the real binary, never a venv shim."""
    override = os.environ.get("DISTILLGAP_SANDBOX_PYTHON")
    if override:
        return override
    return os.path.realpath(getattr(sys, "_base_executable", sys.executable))


@dataclass
class ExecResult:
    status: str                     # ok | load_error | killed_timeout | killed_memory | harness_error
    load_error: str | None = None
    calls: list = field(default_factory=list)   # index-aligned per input; None if never reached
    wall_seconds: float = 0.0
    detail: str | None = None

    def to_json(self):
        return {"status": self.status, "load_error": self.load_error, "calls": self.calls,
                "wall_seconds": round(self.wall_seconds, 4), "detail": self.detail}


def _limits(cpu_seconds: int):
    def apply():
        os.setsid()
        for name, value in (("RLIMIT_CPU", (cpu_seconds, cpu_seconds + 1)),
                            ("RLIMIT_FSIZE", (1 << 20, 1 << 20)),
                            ("RLIMIT_NOFILE", (64, 64)),
                            ("RLIMIT_CORE", (0, 0))):
            try:
                resource.setrlimit(getattr(resource, name), value)
            except (ValueError, OSError):
                pass
        if platform.system() == "Linux":
            try:
                resource.setrlimit(resource.RLIMIT_AS, (MAX_RSS_BYTES * 2, MAX_RSS_BYTES * 2))
            except (ValueError, OSError):
                pass
    return apply


def _rss_bytes(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=2).stdout.strip()
        return int(out) * 1024 if out else 0
    except (subprocess.SubprocessError, ValueError):
        return 0


def run_function(code: str, entry_point: str, inputs: list, *, per_call_seconds: float = 2.0,
                 load_seconds: float = 5.0, wall_seconds: float | None = None) -> ExecResult:
    sentinel = "@@DG" + uuid.uuid4().hex + "@@"
    if wall_seconds is None:
        wall_seconds = load_seconds + per_call_seconds * max(1, len(inputs)) + 5.0
    cpu_seconds = int(wall_seconds) + 2
    job = json.dumps({"code": code, "entry_point": entry_point, "inputs": inputs,
                      "per_call_seconds": per_call_seconds, "load_seconds": load_seconds,
                      "sentinel": sentinel})
    workdir = tempfile.mkdtemp(prefix="dg-sbx-", dir="/private/tmp" if sys.platform == "darwin" else None)
    argv = [base_python(), "-I", "-S", "-B", "-c", CHILD_SOURCE]
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        argv = ["sandbox-exec", "-p", MAC_PROFILE] + argv
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
    started = time.monotonic()
    killed = None
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=workdir, env=env,
                                preexec_fn=_limits(cpu_seconds))
    except OSError as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        return ExecResult("harness_error", detail=f"spawn failed: {exc}")
    chunks: list[bytes] = []
    errs: list[bytes] = []

    def pump(stream, sink, cap):
        total = 0
        for block in iter(lambda: stream.read(65536), b""):
            if total < cap:
                sink.append(block[: cap - total])
            total += len(block)

    readers = [threading.Thread(target=pump, args=(proc.stdout, chunks, 8 << 20), daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, errs, 64 << 10), daemon=True)]
    for reader in readers:
        reader.start()
    try:
        proc.stdin.write(job.encode())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    last_rss = 0.0
    while proc.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed > wall_seconds:
            killed = "killed_timeout"
        elif elapsed - last_rss > 0.5:
            last_rss = elapsed
            if _rss_bytes(proc.pid) > MAX_RSS_BYTES:
                killed = "killed_memory"
        if killed:
            try:
                os.killpg(proc.pid, 9)
            except OSError:
                proc.kill()
            break
        time.sleep(0.02)
    proc.wait()
    for reader in readers:
        reader.join(timeout=5)
    elapsed = time.monotonic() - started
    shutil.rmtree(workdir, ignore_errors=True)
    text = b"".join(chunks).decode("utf-8", errors="replace")
    events = []
    for line in text.splitlines():
        position = line.find(sentinel)
        if position >= 0:
            try:
                events.append(json.loads(line[position + len(sentinel):]))
            except json.JSONDecodeError:
                pass
    calls: list = [None] * len(inputs)
    load = next((e for e in events if e.get("event") == "load"), None)
    for event in events:
        if event.get("event") == "call" and 0 <= event.get("index", -1) < len(inputs):
            calls[event["index"]] = {k: v for k, v in event.items() if k not in ("event", "index")}
    if load is None:
        status = killed or "harness_error"
        detail = b"".join(errs).decode("utf-8", errors="replace")[-800:] or f"exit {proc.returncode}"
        return ExecResult(status, calls=calls, wall_seconds=elapsed, detail=detail)
    if not load.get("ok"):
        return ExecResult("load_error", load_error=load.get("error"), calls=calls, wall_seconds=elapsed)
    done = any(e.get("event") == "done" for e in events)
    if killed or not done:
        # Unreached inputs after a hard kill are attributed to the kill, never silently dropped.
        reason = killed or ("crashed" if proc.returncode else "incomplete")
        for index, call in enumerate(calls):
            if call is None:
                calls[index] = {"ok": False, "cls": "timeout" if reason == "killed_timeout" else "crash",
                                "error": reason}
        return ExecResult(killed or "ok", calls=calls, wall_seconds=elapsed,
                          detail=None if killed else f"child ended early ({reason}, exit {proc.returncode})")
    return ExecResult("ok", calls=calls, wall_seconds=elapsed)


def values_equal(a, b, rel=1e-6, abs_tol=1e-9) -> bool:
    """Canonical-output equality; numeric int/float compare with tolerance, bool stays distinct."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y, rel, abs_tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(values_equal(a[k], b[k], rel, abs_tol) for k in a)
    return a == b
