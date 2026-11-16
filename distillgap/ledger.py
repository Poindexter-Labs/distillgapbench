"""Append-only, hash-chained evidence log plus a content-addressed blob store.

Every record carries seq, unix time and the SHA-256 of the previous line, so a
dropped or edited record breaks the chain. No credentials are ever written:
request bodies contain prompts only; route identity stores endpoint hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid


def _dumps(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class EvidenceLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "blobs").mkdir(exist_ok=True)
        self.path = self.directory / "events.jsonl"
        self.lock = threading.Lock()
        self.seq, self.prev = 0, "0" * 64
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.seq = json.loads(line)["seq"]
                    self.prev = hashlib.sha256(line.encode()).hexdigest()

    def append(self, kind: str, **fields) -> dict:
        with self.lock:
            self.seq += 1
            record = {"seq": self.seq, "unix": time.time(), "kind": kind, "prev_sha256": self.prev, **fields}
            line = _dumps(record)
            with self.path.open("a") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.prev = hashlib.sha256(line.encode()).hexdigest()
            return record

    def put_blob(self, value) -> str:
        text = _dumps(value)
        digest = hashlib.sha256(text.encode()).hexdigest()
        path = self.directory / "blobs" / f"{digest}.json"
        if not path.exists():
            # Unique temporary name: concurrent writers of identical content must not race.
            temporary = path.with_name(f".{digest}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(text)
            temporary.replace(path)
        return digest

    def get_blob(self, digest: str):
        return json.loads((self.directory / "blobs" / f"{digest}.json").read_text())


def verify_chain(path) -> dict:
    prev, count = "0" * 64, 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["prev_sha256"] != prev or record["seq"] != count + 1:
            return {"valid": False, "broken_at_seq": record.get("seq"), "records": count}
        prev = hashlib.sha256(line.encode()).hexdigest()
        count += 1
    return {"valid": True, "records": count, "head_sha256": prev}


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_directory(directory, manifest_name="FROZEN_MANIFEST.json") -> dict:
    """Hash every file (excluding the manifest itself) and write an immutable manifest."""
    directory = Path(directory)
    files = {str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != manifest_name}
    manifest = {"frozen_unix": time.time(), "files": files,
                "events_chain": verify_chain(directory / "events.jsonl") if (directory / "events.jsonl").exists() else None}
    path = directory / manifest_name
    if path.exists():
        raise FileExistsError("Directory already frozen: " + str(path))
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    os.chmod(path, 0o444)
    return manifest


def check_frozen(directory, manifest_name="FROZEN_MANIFEST.json") -> list:
    directory = Path(directory)
    manifest = json.loads((directory / manifest_name).read_text())
    problems = []
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.exists():
            problems.append("missing:" + name)
        elif file_sha256(path) != digest:
            problems.append("changed:" + name)
    return problems
