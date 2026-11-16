"""Stdlib-only child harness executed inside the OS sandbox.

Protocol: one JSON job on stdin:
  {"code": str, "entry_point": str, "inputs": [[arg, ...], ...],
   "per_call_seconds": float, "load_seconds": float, "sentinel": str}
Each event is written to the ORIGINAL stdout as one line prefixed by the
sentinel, flushed immediately, so results that completed before a hard kill
survive. Candidate code never receives expected outputs; comparison happens
outside this process. This file must not import anything outside the stdlib.
"""
import copy
import io
import json
import math
import os
import signal
import sys


class _Deadline(BaseException):
    """Raised by the interval timer; BaseException so `except Exception` cannot swallow it."""


def _alarm(_signum, _frame):
    raise _Deadline()


class _Bounded(io.TextIOBase):
    def __init__(self, limit=65536):
        self.limit, self.size = limit, 0

    def write(self, text):
        self.size += len(text)
        return len(text)


def canon(value, depth=0):
    """Type-tagged canonical JSON form. bool is never conflated with int."""
    if depth > 64:
        raise ValueError("output nesting too deep")
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (list, tuple)):
        return [canon(v, depth + 1) for v in value]
    if isinstance(value, dict):
        items = [[canon(k, depth + 1), canon(v, depth + 1)] for k, v in value.items()]
        items.sort(key=lambda kv: json.dumps(kv[0], sort_keys=True))
        return {"__dict__": items}
    if isinstance(value, (set, frozenset)):
        items = [canon(v, depth + 1) for v in value]
        items.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return {"__set__": items}
    raise TypeError("unsupported output type " + type(value).__name__)


def main():
    job = json.loads(sys.stdin.read())
    sentinel = job["sentinel"]
    out = os.fdopen(os.dup(1), "w")
    sink = _Bounded()
    sys.stdout = sink
    sys.stderr = _Bounded()
    sys.setrecursionlimit(3000)
    signal.signal(signal.SIGALRM, _alarm)

    def emit(event):
        out.write(sentinel + json.dumps(event, sort_keys=True) + "\n")
        out.flush()

    namespace = {"__name__": "__candidate__", "__builtins__": __builtins__}
    try:
        signal.setitimer(signal.ITIMER_REAL, float(job["load_seconds"]))
        exec(compile(job["code"], "<candidate>", "exec"), namespace)
        signal.setitimer(signal.ITIMER_REAL, 0)
    except _Deadline:
        emit({"event": "load", "ok": False, "error": "TimeoutError: module load exceeded limit"})
        return
    except BaseException as exc:  # noqa: BLE001 - classify every candidate failure
        signal.setitimer(signal.ITIMER_REAL, 0)
        emit({"event": "load", "ok": False, "error": (type(exc).__name__ + ": " + str(exc))[:500]})
        return
    function = namespace.get(job["entry_point"])
    if not callable(function):
        emit({"event": "load", "ok": False, "error": "MissingEntryPoint: " + job["entry_point"]})
        return
    emit({"event": "load", "ok": True})
    for index, args in enumerate(job["inputs"]):
        record = {"event": "call", "index": index}
        try:
            arguments = copy.deepcopy(args)
            signal.setitimer(signal.ITIMER_REAL, float(job["per_call_seconds"]))
            result = function(*arguments)
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                value = canon(result)
                text = json.dumps(value, sort_keys=True)
                if len(text) > 65536:
                    record.update(ok=False, cls="output_too_large", error="output exceeds 64 KiB")
                else:
                    shown = repr(result)
                    record.update(ok=True, value=value,
                                  repr=shown if len(shown) <= 400 else shown[:397] + "...")
            except (TypeError, ValueError, RecursionError) as exc:
                record.update(ok=False, cls="unserializable_output", error=str(exc)[:300])
        except _Deadline:
            record.update(ok=False, cls="timeout", error="call exceeded per-call limit")
        except RecursionError as exc:
            signal.setitimer(signal.ITIMER_REAL, 0)
            record.update(ok=False, cls="exception", error=("RecursionError: " + str(exc))[:300])
        except BaseException as exc:  # noqa: BLE001
            signal.setitimer(signal.ITIMER_REAL, 0)
            record.update(ok=False, cls="exception", error=(type(exc).__name__ + ": " + str(exc))[:300])
        emit(record)
    emit({"event": "done", "stdout_chars": sink.size})


if __name__ == "__main__":
    main()
