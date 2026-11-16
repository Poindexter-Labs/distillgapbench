"""Persistent namespace sandbox pool for high-throughput scoring (learning-track RL).

Creating a user+mount+pid namespace per call caps scoring at about 6 calls/s per node: namespace set-up and
teardown serialise in the kernel, so more workers only add queueing (measured 2026-11-13: 596 calls took 94 s with
32 workers and 186 s with 96). Here each worker builds the SAME locked-down namespace as linux_sandbox.py once (no
network; /home, /projects, /scratch, /lus, /var, /run, /opt, /srv, /media, /dev/shm and /tmp hidden behind private
tmpfs; only the node-local interpreter prefix visible). Its PID 1 is a small zygote that, per request:

  * starts the FROZEN distillgap child program in a fresh process (python -I -S -B), in a fresh scratch directory,
    with the frozen rlimits (CPU, file size, open files, core, address space) plus the process cap;
  * enforces the wall clock, killing the child's process group on expiry;
  * then kills EVERY other process left in the namespace (kill(-1) from PID 1), reaps them, and wipes /tmp and
    /dev/shm, so nothing survives from one evaluation to the next;
  * returns the child's stdout/stderr, which the parent parses exactly as distillgap.sandbox.run_function does.

The candidate cannot signal the zygote (a PID-namespace init ignores signals from inside without handlers) and cannot
see the host. If a zygote dies or hangs, the call is infrastructure (`harness_error`) and the worker is replaced.
`install_pool()` routes the unmodified distillgap verifier through the pool (in memory; frozen files untouched).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import select
import signal
import struct
import subprocess
import threading
import time
import uuid

import linux_sandbox as L

POOL_VERSION = "distillgap_ns_pool_v1"

ZYGOTE_SOURCE = r'''
import json, os, resource, shutil, signal, struct, subprocess, sys, tempfile, time
stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
def read_exact(n):
    data = b""
    while len(data) < n:
        chunk = stdin.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data
def read_msg():
    head = read_exact(4)
    if head is None:
        return None
    return json.loads(read_exact(struct.unpack(">I", head)[0]))
def write_msg(obj):
    data = json.dumps(obj).encode()
    stdout.write(struct.pack(">I", len(data)) + data)
    stdout.flush()
def reap():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
def wipe(path, keep=()):
    for entry in os.listdir(path):
        if entry in keep:
            continue
        full = os.path.join(path, entry)
        try:
            if os.path.isdir(full) and not os.path.islink(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                os.unlink(full)
        except OSError:
            pass
child_source = None
write_msg({"ready": True, "pid": os.getpid()})
while True:
    msg = read_msg()
    if msg is None:
        break
    if "child_source" in msg:
        child_source = msg["child_source"]
        write_msg({"ok": True})
        continue
    lim = msg["limits"]
    work = tempfile.mkdtemp(prefix="eval-", dir="/tmp")
    def pre():
        os.setsid()
        for name, value in (("RLIMIT_CPU", (lim["cpu"], lim["cpu"] + 1)), ("RLIMIT_FSIZE", (1 << 20, 1 << 20)),
                            ("RLIMIT_NOFILE", (64, 64)), ("RLIMIT_CORE", (0, 0)), ("RLIMIT_AS", (lim["as"], lim["as"])),
                            ("RLIMIT_NPROC", (lim["nproc"], lim["nproc"]))):
            try:
                resource.setrlimit(getattr(resource, name), value)
            except (ValueError, OSError):
                pass
    started = time.monotonic()
    killed = None
    try:
        proc = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", child_source], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work,
                                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}, preexec_fn=pre)
        try:
            out, err = proc.communicate(msg["job"].encode(), timeout=lim["wall"])
        except subprocess.TimeoutExpired:
            killed = "killed_timeout"
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            out, err = proc.communicate()
        returncode = proc.returncode
    except OSError as exc:
        out, err, returncode = b"", ("spawn failed: %s" % exc).encode(), None
    try:
        os.kill(-1, signal.SIGKILL)  # everything the candidate left behind in this PID namespace
    except OSError:
        pass
    reap()
    wipe("/tmp", keep=("work",))
    wipe("/dev/shm")
    write_msg({"stdout": out[: 8 << 20].decode("utf-8", "replace"), "stderr": err[-(64 << 10):].decode("utf-8", "replace"),
               "returncode": returncode, "killed": killed, "elapsed": time.monotonic() - started})
'''


class Worker:
    def __init__(self, python_path: str, child_source: str):
        argv, mechanism = L.isolation_argv(python_path, ZYGOTE_SOURCE)
        if mechanism != "unshare":
            raise SystemExit("the namespace pool requires the unshare mechanism (zygote must be the namespace's PID 1)")
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, start_new_session=True)
        self.fd = self.proc.stdout.fileno()
        if not (self._read(120) or {}).get("ready"):
            self.close()
            raise RuntimeError("zygote did not start")
        self._write({"child_source": child_source})
        if not (self._read(60) or {}).get("ok"):
            self.close()
            raise RuntimeError("zygote rejected the child program")

    def _write(self, obj):
        data = json.dumps(obj).encode()
        self.proc.stdin.write(struct.pack(">I", len(data)) + data)
        self.proc.stdin.flush()

    def _read_exact(self, n, deadline):
        data = b""
        while len(data) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self.fd, n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        head = self._read_exact(4, deadline)
        if head is None:
            return None
        body = self._read_exact(struct.unpack(">I", head)[0], deadline)
        return None if body is None else json.loads(body)

    def request(self, job: str, limits: dict):
        try:
            self._write({"job": job, "limits": limits})
        except (BrokenPipeError, OSError):
            return None
        return self._read(limits["wall"] + 60)

    def close(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


class Pool:
    def __init__(self, size: int):
        from distillgap import sandbox as S
        self.python = S.base_python()
        self.child = S.CHILD_SOURCE
        self.size, self.created, self.lock = size, 0, threading.Lock()
        self.idle: queue.Queue = queue.Queue()

    def acquire(self):
        try:
            return self.idle.get_nowait()
        except queue.Empty:
            pass
        with self.lock:
            create = self.created < self.size
            if create:
                self.created += 1
        if create:
            try:
                return Worker(self.python, self.child)
            except Exception:
                with self.lock:
                    self.created -= 1
                raise
        return self.idle.get()

    def release(self, worker, healthy: bool):
        if healthy:
            self.idle.put(worker)
        else:
            worker.close()
            with self.lock:
                self.created -= 1


POOL: Pool | None = None


def run_function_pool(code: str, entry_point: str, inputs: list, *, per_call_seconds: float = 2.0,
                      load_seconds: float = 5.0, wall_seconds: float | None = None):
    from distillgap import sandbox as S
    sentinel = "@@DG" + uuid.uuid4().hex + "@@"
    if wall_seconds is None:
        wall_seconds = load_seconds + per_call_seconds * max(1, len(inputs)) + 5.0
    cpu_seconds = int(wall_seconds) + 2
    job = json.dumps({"code": code, "entry_point": entry_point, "inputs": inputs,
                      "per_call_seconds": per_call_seconds, "load_seconds": load_seconds, "sentinel": sentinel})
    limits = {"cpu": cpu_seconds, "as": S.MAX_RSS_BYTES * 2, "nproc": L.user_thread_count() + L.PROCESS_MARGIN,
              "wall": wall_seconds + L.STARTUP_GRACE_SECONDS}
    try:
        worker = POOL.acquire()
    except Exception as exc:  # noqa: BLE001 - infrastructure
        return S.ExecResult("harness_error", detail=f"pool worker unavailable: {exc}")
    response = worker.request(job, limits)
    POOL.release(worker, healthy=response is not None)
    if response is None:
        return S.ExecResult("harness_error", detail="zygote died or hung")
    events = []
    for line in response["stdout"].splitlines():
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
    elapsed = response.get("elapsed", 0.0)
    killed = response.get("killed")
    if load is None:
        return S.ExecResult("harness_error", calls=calls, wall_seconds=elapsed,
                            detail=f"no load event ({killed or 'exited'}, exit {response.get('returncode')}): "
                                   + response.get("stderr", "")[-600:])
    if not load.get("ok"):
        return S.ExecResult("load_error", load_error=load.get("error"), calls=calls, wall_seconds=elapsed)
    done = any(e.get("event") == "done" for e in events)
    if killed or not done:
        reason = killed or ("crashed" if response.get("returncode") else "incomplete")
        for index, call in enumerate(calls):
            if call is None:
                calls[index] = {"ok": False, "cls": "timeout" if reason == "killed_timeout" else "crash", "error": reason}
        return S.ExecResult(killed or "ok", calls=calls, wall_seconds=elapsed,
                            detail=None if killed else f"child ended early ({reason}, exit {response.get('returncode')})")
    return S.ExecResult("ok", calls=calls, wall_seconds=elapsed)


def install_pool(size: int = 32):
    global POOL
    info = L.install()  # node-local interpreter + the per-call runner as the base; then switch to the pool
    POOL = Pool(size)
    from distillgap import sandbox, verifier
    sandbox.run_function = run_function_pool
    verifier.run_function = run_function_pool
    info.update(pool=POOL_VERSION, pool_size=size)
    return info
