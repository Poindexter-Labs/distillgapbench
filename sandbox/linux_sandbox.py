"""Linux (cluster compute node) isolation for the operator verifier, used by the learning-track RL loop.

The frozen v5 verifier (`distillgap/verifier.py`) runs untrusted code through `distillgap.sandbox.run_function`,
whose filesystem isolation is macOS sandbox-exec. This module provides `run_function_bwrap`: a copy of that
function with ONE change - the child is wrapped in Linux namespaces, by bubblewrap where installed and otherwise by
util-linux `unshare` plus a setup script (the cluster's SLES compute nodes have unshare but no bwrap):

  * new user, pid, net, ipc, uts and cgroup namespaces (--unshare-all): no network, no view of other processes;
  * the host root is mounted read-only, then every data/home mount (/home, /projects, /scratch, /lus, /var, /run,
    /tmp, /mnt, /opt, /srv, /media) is hidden behind an empty tmpfs, so hidden tests, training data, adapters and
    credentials are unreadable (reward hacking by reading files is impossible); only the base interpreter's own
    install prefix is re-exposed read-only;
  * a private tmpfs working directory; --die-with-parent and --new-session, so a wall-clock kill removes everything;
  * RLIMIT_NPROC (the frozen sandbox denies fork only on macOS), so a fork bomb is bounded.

The rlimits (CPU, file size, open files, core, address space) and the wall-clock kill are applied exactly as in the
frozen sandbox (they are inherited by the bwrap child). The child program, the sentinel protocol and the result
parsing are unchanged, so verdicts must equal the macOS verifier's; `qualify_sandbox.py` checks that on recorded
completions before any RL run. `install()` swaps the runner into the imported, unmodified distillgap modules
(in memory only; the frozen files are never edited).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

LINUX_SANDBOX_VERSION = "distillgap_linux_ns_v4"
# RLIMIT_NPROC bounds fork bombs (the frozen sandbox blocks fork only on macOS). Linux counts EVERY thread of the
# real uid on the node (other jobs included; 424-841 observed), so the cap is "current count + margin", re-measured
# at spawn time (cached 2 s): a fork bomb gets at most PROCESS_MARGIN extra processes and normal runs are never blocked.
PROCESS_MARGIN = 512
STARTUP_GRACE_SECONDS = 10.0
_NPROC_CACHE = {"at": 0.0, "count": 0}
_NPROC_LOCK = threading.Lock()  # single-flight refresh: many threads must not all rescan /proc at once
HIDDEN_MOUNTS = ["/home", "/projects", "/scratch", "/lus", "/local", "/var", "/run", "/tmp", "/mnt", "/opt", "/srv",
                 "/media"]


UNSHARE_SETUP = r"""set -eu
PREFIX="$1"; PY="$2"; CHILD="$3"
# stage the interpreter on a private tmpfs over /mnt BEFORE anything is hidden (works wherever PREFIX lives)
mount -t tmpfs -o size=1m dg-stage /mnt
mkdir /mnt/prefix
mount --rbind "$PREFIX" /mnt/prefix
for m in /home /projects /scratch /lus /local /var /run /opt /srv /media /dev/shm /tmp; do
  if [ -d "$m" ]; then mount -t tmpfs -o size=64m dg-hide "$m"; fi
done
mkdir -p "$PREFIX"
mount --rbind /mnt/prefix "$PREFIX"
mkdir -p /tmp/work
cd /tmp/work
exec "$PY" -I -S -B -c "$CHILD"
"""


def bwrap_argv(python_path: str) -> list:
    real = os.path.realpath(python_path)
    prefix = str(Path(real).parents[1])  # .../cpython-3.12.x-<platform>/ (bin/python3.12 -> prefix)
    argv = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--ro-bind", "/", "/"]
    for mount in HIDDEN_MOUNTS + ["/dev/shm"]:
        if os.path.isdir(mount):
            argv += ["--tmpfs", mount]
    argv += ["--ro-bind", prefix, prefix, "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/work", "--chdir", "/work"]
    return argv + ["--", real]


def isolation_argv(python_path: str, child_source: str) -> tuple[list, str]:
    """bwrap where installed; otherwise unshare (user, net, pid, ipc, uts, mount namespaces) with the same hiding."""
    real = os.path.realpath(python_path)
    if shutil.which("bwrap"):
        return bwrap_argv(python_path) + ["-I", "-S", "-B", "-c", child_source], "bwrap"
    prefix = str(Path(real).parents[1])
    argv = ["unshare", "--user", "--map-root-user", "--net", "--pid", "--fork", "--kill-child", "--mount-proc",
            "--mount", "--ipc", "--uts", "--", "/bin/sh", "-c", UNSHARE_SETUP, "dg-sandbox", prefix, real, child_source]
    return argv, "unshare"


def user_thread_count() -> int:
    """Threads owned by this real uid on the node (what RLIMIT_NPROC counts). One thread refreshes at a time (every
    5 s); the others use the last value. A full /proc scan per call from dozens of threads serialised on the GIL and
    capped scoring throughput (measured 2026-11-13)."""
    now = time.monotonic()
    if _NPROC_CACHE["count"] and (now - _NPROC_CACHE["at"] < 5.0 or not _NPROC_LOCK.acquire(blocking=False)):
        return _NPROC_CACHE["count"]
    if not _NPROC_CACHE["count"]:
        _NPROC_LOCK.acquire()
    try:
        if _NPROC_CACHE["count"] and time.monotonic() - _NPROC_CACHE["at"] < 5.0:
            return _NPROC_CACHE["count"]
        uid, total = os.getuid(), 0
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/status") as handle:
                    fields = dict(line.split(":", 1) for line in handle if ":" in line)
                if int(fields["Uid"].split()[0]) == uid:
                    total += int(fields["Threads"])
            except (OSError, KeyError, ValueError):
                continue
        _NPROC_CACHE.update(at=time.monotonic(), count=total)
        return total
    finally:
        _NPROC_LOCK.release()


def _limits_with_nproc(base):
    cap = user_thread_count() + PROCESS_MARGIN  # measured in the parent, before fork

    def apply():
        import resource
        base()
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (cap, cap))
        except (ValueError, OSError):
            pass
    return apply


def run_function_bwrap(code: str, entry_point: str, inputs: list, *, per_call_seconds: float = 2.0,
                       load_seconds: float = 5.0, wall_seconds: float | None = None):
    from distillgap import sandbox as S  # the frozen module: child source, limits, result type, RSS probe
    sentinel = "@@DG" + uuid.uuid4().hex + "@@"
    if wall_seconds is None:
        wall_seconds = load_seconds + per_call_seconds * max(1, len(inputs)) + 5.0
    cpu_seconds = int(wall_seconds) + 2
    wall_seconds += STARTUP_GRACE_SECONDS  # outer safety net only; the child's own load/per-call timers are unchanged
    job = json.dumps({"code": code, "entry_point": entry_point, "inputs": inputs,
                      "per_call_seconds": per_call_seconds, "load_seconds": load_seconds,
                      "sentinel": sentinel})
    workdir = tempfile.mkdtemp(prefix="dg-sbx-")  # host cwd for the bwrap process only; the child sees /work
    argv, _ = isolation_argv(S.base_python(), S.CHILD_SOURCE)
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
    started = time.monotonic()
    killed = None
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=workdir, env=env, preexec_fn=_limits_with_nproc(S._limits(cpu_seconds)))
    except OSError as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        return S.ExecResult("harness_error", detail=f"spawn failed: {exc}")
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
    while proc.poll() is None:
        if time.monotonic() - started > wall_seconds:
            killed = "killed_timeout"
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
        stderr = b"".join(errs).decode("utf-8", errors="replace")
        # address-space exhaustion surfaces as MemoryError / failed allocation under RLIMIT_AS (no RSS watchdog
        # can see a process inside the pid namespace); classify it as the frozen sandbox would
        # No load event at all: the interpreter never got as far as running the candidate (whose module load is timed
        # by the child itself). Under concurrency this is infrastructure (slow start-up), never a model failure.
        return S.ExecResult("harness_error", calls=calls, wall_seconds=elapsed,
                            detail=f"no load event ({killed or 'exited'}, exit {proc.returncode}): " + (stderr[-600:] or ""))
    if not load.get("ok"):
        return S.ExecResult("load_error", load_error=load.get("error"), calls=calls, wall_seconds=elapsed)
    done = any(e.get("event") == "done" for e in events)
    if killed or not done:
        reason = killed or ("crashed" if proc.returncode else "incomplete")
        for index, call in enumerate(calls):
            if call is None:
                calls[index] = {"ok": False, "cls": "timeout" if reason == "killed_timeout" else "crash", "error": reason}
        return S.ExecResult(killed or "ok", calls=calls, wall_seconds=elapsed,
                            detail=None if killed else f"child ended early ({reason}, exit {proc.returncode})")
    return S.ExecResult("ok", calls=calls, wall_seconds=elapsed)


def localize_interpreter(python_path: str | None = None) -> str:
    """Copy the sandbox interpreter's install prefix to node-local storage once per process and point the frozen
    sandbox at it (DISTILLGAP_SANDBOX_PYTHON). Starting many sandboxed interpreters from the shared Lustre filesystem
    stalls on metadata under concurrency (observed: spurious start-up kills at 96 parallel sandboxes)."""
    from distillgap import sandbox as S
    real = os.path.realpath(python_path or S.base_python())
    prefix = Path(real).parents[1]
    # Release: never under /tmp or /dev/shm, which the sandbox pool wipes after every evaluation (the study's TMPDIR
    # was node-local storage elsewhere). DISTILLGAP_INTERPRETER_DIR overrides; /var/tmp is hidden from sandboxed code.
    root = os.environ.get("DISTILLGAP_INTERPRETER_DIR") or os.environ.get("TMPDIR") or "/var/tmp"
    if os.path.realpath(root).startswith(("/tmp", "/dev/shm")) or not os.access(root, os.W_OK):
        root = "/var/tmp"
    local_root = Path(root) / f"dg-python-{os.getpid()}"
    target = local_root / prefix.name
    if not target.exists():
        local_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-a", str(prefix), str(target)], check=True)
    local_python = str(target / Path(real).relative_to(prefix))
    os.environ["DISTILLGAP_SANDBOX_PYTHON"] = local_python
    return local_python


def install(local_interpreter: bool = True):
    """Route the (unmodified, imported) distillgap verifier through bubblewrap. Linux only; refuses otherwise."""
    if not (shutil.which("bwrap") or shutil.which("unshare")):
        raise SystemExit("neither bwrap nor unshare found: refusing to run untrusted code without isolation")
    from distillgap import sandbox, verifier
    interpreter = localize_interpreter() if local_interpreter else sandbox.base_python()
    sandbox.run_function = run_function_bwrap
    verifier.run_function = run_function_bwrap
    return {"interpreter": interpreter, "linux_sandbox": LINUX_SANDBOX_VERSION, "mechanism": "bwrap" if shutil.which("bwrap") else "unshare",
            "hidden_mounts": [m for m in HIDDEN_MOUNTS + ["/dev/shm"] if os.path.isdir(m)]}
