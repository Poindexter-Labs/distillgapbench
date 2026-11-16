"""Check that researcher-written code can run in the namespace sandbox here (container or node), and that it cannot
see the network, other processes, credentials, or the code, keys and results under /home and /opt.

Exit 0 when isolation works. A real run refuses to start without it (scripts/run_grid.py --sandbox ns_pool)."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sandbox")]

PROBE = '''
def probe(x):
    import json, os, socket
    seen = {m: sorted(os.listdir(m)) if os.path.isdir(m) else None for m in ("/home", "/opt")}
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2).close()
        net = "open"
    except OSError:
        net = "blocked"
    pids = len([p for p in os.listdir("/proc") if p.isdigit()])
    keys = sorted(k for k in os.environ if "KEY" in k or "TOKEN" in k)
    return json.dumps({"value": x * 2, "seen": seen, "net": net, "pids": pids, "env_keys": keys})
'''

HINT = """The sandbox needs unprivileged user namespaces and its own /proc (see docs/CONTAINERS.md).
  Docker: --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt systempaths=unconfined
  Podman: --security-opt unmask=ALL
  Host:   sysctl user.max_user_namespaces must be > 0 (and kernel.apparmor_restrict_unprivileged_userns=0 on Ubuntu)"""


def main():
    if not (shutil.which("bwrap") or shutil.which("unshare")):
        sys.exit("FAIL: neither bwrap nor unshare is installed\n" + HINT)
    import ns_pool  # the persistent namespace pool that scripts/run_grid.py --sandbox ns_pool uses
    from distillgap import sandbox
    info = ns_pool.install_pool(size=2)
    # several calls in a row: the pool cleans up between evaluations, and a later call must still work
    for _ in range(3):
        result = sandbox.run_function(PROBE, "probe", [[21]], per_call_seconds=10.0, load_seconds=10.0)
        if result.status != "ok":
            break
    if result.status != "ok":
        print(json.dumps(result.to_json(), indent=1)[:2000])
        import linux_sandbox  # one namespace per call: reports the underlying unshare error

        linux_sandbox.install()
        direct = sandbox.run_function(PROBE, "probe", [[21]], per_call_seconds=10.0, load_seconds=10.0)
        print("direct namespace call:", direct.status, (direct.detail or "")[:600])
        sys.exit(f"FAIL: sandboxed call ended with status {result.status}\n" + HINT)
    call = result.calls[0] if result.calls else None
    if not (isinstance(call, dict) and call.get("ok")):
        sys.exit(f"FAIL: probe raised inside the sandbox: {call}")
    report = json.loads(call["value"])
    problems = []
    if report["value"] != 42:
        problems.append("wrong return value")
    if report["net"] != "blocked":
        problems.append("network reachable")
    if report["pids"] > 5:
        problems.append(f"{report['pids']} processes visible")
    if report["env_keys"]:
        problems.append("credential variables visible: " + ", ".join(report["env_keys"]))
    for mount, entries in report["seen"].items():
        if entries:
            visible = [e for e in entries if e not in ("python",)]  # the staged interpreter's own prefix may show
            if visible:
                problems.append(f"{mount} not hidden: {visible[:5]}")
    print(json.dumps({"sandbox": {k: info.get(k) for k in ("pool", "mechanism", "linux_sandbox")}, **report},
                     sort_keys=True))
    if problems:
        sys.exit("FAIL: " + "; ".join(problems))
    print("sandbox OK")


if __name__ == "__main__":
    main()
