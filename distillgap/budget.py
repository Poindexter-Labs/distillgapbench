"""Matched, conservative budgets for one arm or one evaluation phase.

Paid model calls reserve their worst case (input upper bound, full per-call output
cap, worst-case dollars) before sending. Outcomes:
  settled      - provider usage known; the actual usage is charged
  not_billed   - definitive pre-processing rejection (HTTP 4xx incl. 429); call counted, 0 tokens
  ambiguous    - timeout/connection/5xx/missing usage: the FULL reservation is charged and the
                 call is never replayed automatically
The first binding limit stops new work; queue wait is recorded outside the clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time
import uuid

from .vendor.rlenv_budget import PRICES, Prices, Usage  # noqa: F401  (vendored, hash-pinned)


class BudgetStop(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelLimits:
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_output_per_call: int
    max_usd: float

    def validate(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{key} must be positive")
        return self


class ModelMeter:
    def __init__(self, name: str, limits: ModelLimits, prices: Prices):
        self.name, self.limits, self.prices = name, limits.validate(), prices
        self.lock = threading.Lock()
        self.calls = self.input_tokens = self.output_tokens = self.reasoning_tokens = 0
        self.usd = 0.0
        self.outstanding: dict[str, dict] = {}
        self.outcomes = {"settled": 0, "not_billed": 0, "ambiguous": 0, "rate_limited_retries": 0,
                         "infrastructure_retries": 0}

    def _committed(self):
        res = self.outstanding.values()
        return (self.input_tokens + sum(r["input_tokens"] for r in res),
                self.output_tokens + sum(r["output_tokens"] for r in res),
                self.usd + sum(r["usd"] for r in res))

    def reserve(self, input_upper_bound: int) -> dict:
        with self.lock:
            if self.calls >= self.limits.max_calls:
                raise BudgetStop(f"{self.name}: maximum calls reached")
            used_in, used_out, used_usd = self._committed()
            if used_in + input_upper_bound > self.limits.max_input_tokens:
                raise BudgetStop(f"{self.name}: input-token budget cannot cover reservation")
            output = min(self.limits.max_output_per_call, self.limits.max_output_tokens - used_out)
            if output < 1024:
                raise BudgetStop(f"{self.name}: output-token budget exhausted")
            usd = self.prices.upper_cost(input_upper_bound, output)
            if used_usd + usd > self.limits.max_usd + 1e-9:
                raise BudgetStop(f"{self.name}: dollar budget cannot cover worst-case call")
            self.calls += 1
            reservation = {"id": uuid.uuid4().hex, "meter": self.name, "call_number": self.calls,
                           "input_tokens": input_upper_bound, "output_tokens": output, "usd": usd}
            self.outstanding[reservation["id"]] = reservation
            return dict(reservation)

    def settle(self, reservation_id: str, usage: Usage) -> dict:
        with self.lock:
            reservation = self.outstanding.pop(reservation_id)
            cost = self.prices.cost(usage)
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.reasoning_tokens += usage.reasoning_tokens
            self.usd += cost
            self.outcomes["settled"] += 1
            exceeded = (usage.input_tokens > reservation["input_tokens"]
                        or usage.output_tokens > reservation["output_tokens"])
            return {"usd": cost, "exceeded_reservation": exceeded}

    def not_billed(self, reservation_id: str):
        with self.lock:
            self.outstanding.pop(reservation_id)
            self.outcomes["not_billed"] += 1

    def ambiguous(self, reservation_id: str) -> dict:
        with self.lock:
            reservation = self.outstanding.pop(reservation_id)
            self.input_tokens += reservation["input_tokens"]
            self.output_tokens += reservation["output_tokens"]
            self.usd += reservation["usd"]
            self.outcomes["ambiguous"] += 1
            return reservation

    def snapshot(self) -> dict:
        with self.lock:
            return {"calls": self.calls, "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens, "reasoning_tokens": self.reasoning_tokens,
                    "estimated_usd": round(self.usd, 6), "outcomes": dict(self.outcomes),
                    "unresolved_reservations": list(self.outstanding.values()),
                    "limits": asdict(self.limits), "price_basis": self.prices.description}


class CountMeter:
    def __init__(self, name: str, limit: int):
        self.name, self.limit, self.used = name, int(limit), 0
        self.lock = threading.Lock()

    def take(self, n: int = 1):
        with self.lock:
            if self.used + n > self.limit:
                raise BudgetStop(f"{self.name}: limit {self.limit} reached ({self.used} used)")
            self.used += n

    def remaining(self) -> int:
        return self.limit - self.used

    def snapshot(self):
        return {"used": self.used, "limit": self.limit}


class Clock:
    """Execution clock that starts when the arm starts (not at queue submission)."""

    def __init__(self, wall_seconds: float, now=time.monotonic):
        self.wall_seconds, self.now = float(wall_seconds), now
        self.started = now()
        self.started_unix = time.time()

    def elapsed(self):
        return self.now() - self.started

    def remaining(self):
        return max(0.0, self.wall_seconds - self.elapsed())

    def check(self, reserve_seconds: float = 0.0):
        if self.remaining() <= reserve_seconds:
            raise BudgetStop("wall-clock deadline reached")

    def snapshot(self):
        return {"wall_seconds": self.wall_seconds, "elapsed_seconds": round(self.elapsed(), 3),
                "started_unix": self.started_unix}


def input_upper_bound(*texts: str) -> int:
    """Conservative token upper bound: one token per two UTF-8 bytes plus framing slack."""
    return sum(len(t.encode("utf-8")) for t in texts) // 2 + 1024
