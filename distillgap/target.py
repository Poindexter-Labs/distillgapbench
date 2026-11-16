"""Target interfaces. Requests carry only rendered prompts, sample counts and seeds.

Backends:
  FixtureTarget      deterministic local fixtures for tests (never used for reported results)
  QueueTarget        file-queue client; transport = local directory or the shared SSH pool
An unavailable/erroring backend raises TargetUnavailable: infrastructure-incomplete
evidence, never a target failure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import threading
import time
import uuid

PROFILE_PATH = Path(__file__).resolve().parents[1] / "cluster/target_profile.json"


class TargetUnavailable(RuntimeError):
    pass


def profile_sha256(path=PROFILE_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile(path=PROFILE_PATH) -> dict:
    return json.loads(Path(path).read_text())


class FixtureTarget:
    """behaviour(prompt, n, seed) -> list[str] completions. For local tests only."""

    def __init__(self, behaviour, fail_after: int | None = None):
        self.behaviour, self.fail_after, self.requests = behaviour, fail_after, 0
        self.identity = {"backend": "fixture", "reportable": False}

    def generate(self, items: list, phase: str) -> dict:
        self.requests += 1
        if self.fail_after is not None and self.requests > self.fail_after:
            raise TargetUnavailable("fixture target outage")
        out = {}
        for item in items:
            completions = self.behaviour(item["prompt"], item["n"], item["seed"])
            out[item["item_id"]] = {"completions": completions, "new_tokens": [len(c) for c in completions],
                                    "finish": ["fixture"] * len(completions)}
        return out


class LocalDirTransport:
    """Queue in a local directory (local MPS smoke tests of target_server.py)."""

    def __init__(self, root):
        self.root = Path(root)

    def write(self, relative: str, text: str):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)

    def read(self, relative: str) -> str | None:
        path = self.root / relative
        return path.read_text() if path.exists() else None


class SshPoolTransport:
    """Shared-pool transport. Writes are content-addressed by a fresh rid and therefore
    idempotent; an ambiguous write is reconciled with a read before any resubmission."""

    def __init__(self, remote_root: str):
        import sys
        from .teacher import REPO
        compute = REPO / "training/rl_env_post_train_bench/harness_repair_20261112/orchestration/compute"
        if str(compute) not in sys.path:
            sys.path.insert(0, str(compute))
        import bridge_tool  # shared SSH control pool; only remote()/remote_readonly() are used
        self.bridge = bridge_tool
        self.root = remote_root.rstrip("/")
        self.lock = threading.Lock()

    def write(self, relative: str, text: str):
        path = f"{self.root}/{relative}"
        directory = path.rsplit("/", 1)[0]
        temporary = f"{directory}/.{path.rsplit('/', 1)[1]}.tmp"
        command = (f"umask 077; mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(temporary)} "
                   f"&& mv -f {shlex.quote(temporary)} {shlex.quote(path)}")
        with self.lock:
            for attempt in range(3):
                try:
                    self.bridge.remote(command, stdin=text, timeout=90)
                    return
                except Exception:  # noqa: BLE001 - reconcile before deciding
                    present = self.bridge.remote_readonly(
                        f"test -f {shlex.quote(path)} && echo PRESENT || echo ABSENT", timeout=60).strip()
                    if present == "PRESENT":
                        return
                    if attempt == 2:
                        raise

    def read(self, relative: str) -> str | None:
        path = f"{self.root}/{relative}"
        with self.lock:
            out = self.bridge.remote_readonly(
                f"if test -f {shlex.quote(path)}; then cat {shlex.quote(path)}; else printf __ABSENT__; fi",
                timeout=90)
        return None if out == "__ABSENT__" else out


class QueueTarget:
    def __init__(self, transport, queue_name: str, *, poll_seconds: float = 5.0,
                 timeout_seconds: float = 3600, profile_path=PROFILE_PATH):
        self.transport, self.queue = transport, queue_name.strip("/")
        self.poll, self.timeout = poll_seconds, timeout_seconds
        self.profile_sha = profile_sha256(profile_path)
        ready = self.transport.read(f"{self.queue}/ready.json")
        if ready is None:
            raise TargetUnavailable("target server is not ready (no ready.json)")
        self.ready = json.loads(ready)
        if self.ready.get("profile_sha256") != self.profile_sha:
            raise TargetUnavailable("target server profile differs from the frozen local profile")
        self.identity = {"backend": "queue", "queue": self.queue, "profile_sha256": self.profile_sha,
                         "hardware": self.ready.get("hardware"), "observed_files": self.ready.get("observed_files"),
                         "reportable": True}

    def generate(self, items: list, phase: str) -> dict:
        rid = f"{phase}-{uuid.uuid4().hex}"
        request = {"rid": rid, "profile_sha256": self.profile_sha,
                   "items": [{k: item[k] for k in ("item_id", "prompt", "n", "seed")} for item in items]}
        try:
            self.transport.write(f"{self.queue}/requests/{rid}.json", json.dumps(request))
        except Exception as exc:  # noqa: BLE001 - transport failure is infrastructure, never a target outcome
            raise TargetUnavailable(f"request submission failed for {rid}: {str(exc)[:300]}") from exc
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            time.sleep(self.poll)
            try:
                text = self.transport.read(f"{self.queue}/responses/{rid}.json")
            except Exception:  # noqa: BLE001 - transient pooled read; retried under the same original deadline
                text = None
            if text is None:
                continue
            response = json.loads(text)
            if "error" in response:
                raise TargetUnavailable("target server error: " + response["error"])
            if response.get("profile_sha256") != self.profile_sha:
                raise TargetUnavailable("response profile mismatch")
            by_id = {item["item_id"]: item for item in response["items"]}
            if set(by_id) != {item["item_id"] for item in items}:
                raise TargetUnavailable("response item set differs from request")
            return {k: {"completions": v["completions"], "new_tokens": v["new_tokens"], "finish": v["finish"],
                        "seed": v.get("seed"), "rid": rid} for k, v in by_id.items()}
        raise TargetUnavailable(f"no target response for {rid} within {self.timeout}s")


class LocalVllmTarget:
    """v1: the frozen student served by vLLM (OpenAI-compatible /v1/completions) on the SAME SovAI node as the arm
    controller, so student queries never cross the shared SSH pool. Decoding mirrors the frozen v5 profiles
    (temperature 0.6, top_p 0.95, no top-k, n samples, per-item seed); each item carries its environment's stop
    strings and max_tokens. Same generate() contract as QueueTarget; any transport/server error raises
    TargetUnavailable (infrastructure, never a target outcome)."""

    def __init__(self, base_url: str, served_model: str, student_id: str, *, timeout_seconds: float = 900,
                 concurrency: int = 32, default_stop: list | None = None, default_max_tokens: int | None = None):
        import requests
        self.requests, self.base, self.model = requests, base_url.rstrip("/"), served_model
        self.timeout, self.concurrency = timeout_seconds, concurrency
        self.default_stop, self.default_max_tokens = default_stop, default_max_tokens  # coding items carry neither
        try:
            models = requests.get(self.base + "/v1/models", timeout=30).json()
        except Exception as exc:  # noqa: BLE001
            raise TargetUnavailable(f"student server not reachable: {exc}") from exc
        served = [m.get("id") for m in models.get("data", [])]
        if served_model not in served:
            raise TargetUnavailable(f"student server serves {served}, expected {served_model}")
        self.identity = {"backend": "local_vllm", "student_id": student_id, "served_model": served_model,
                         "decoding": {"temperature": 0.6, "top_p": 0.95, "top_k": -1}, "reportable": True}

    def _one(self, item):
        body = {"model": self.model, "prompt": item["prompt"], "n": item["n"], "temperature": 0.6, "top_p": 0.95,
                "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": item.get("max_tokens") or self.default_max_tokens,
                "stop": item.get("stop") or self.default_stop, "seed": item["seed"]}
        try:
            r = self.requests.post(self.base + "/v1/completions", json=body, timeout=self.timeout)
            r.raise_for_status()
            choices = sorted(r.json()["choices"], key=lambda c: c["index"])
        except Exception as exc:  # noqa: BLE001
            raise TargetUnavailable(f"student generation failed for {item['item_id']}: {str(exc)[:300]}") from exc
        if len(choices) != item["n"]:
            raise TargetUnavailable(f"student returned {len(choices)} samples for {item['item_id']}")
        return item["item_id"], {"completions": [c["text"] for c in choices], "finish": [c.get("finish_reason") for c in choices],
                                 "new_tokens": [None] * len(choices), "seed": item["seed"],
                                 "prompt_tokens": (r.json().get("usage") or {}).get("prompt_tokens")}

    def generate(self, items: list, phase: str) -> dict:
        from concurrent.futures import ThreadPoolExecutor
        rid = f"{phase}-{uuid.uuid4().hex}"
        with ThreadPoolExecutor(self.concurrency) as pool:
            results = dict(pool.map(self._one, items))
        return {k: {**v, "rid": rid} for k, v in results.items()}
