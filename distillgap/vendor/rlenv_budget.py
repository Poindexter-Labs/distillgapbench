"""Raw provider accounting. Product billing multipliers are never imported."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time


class BudgetStop(RuntimeError):
    pass


class InvalidUsage(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    wall_seconds: int = 7200
    input_tokens: int = 1_000_000
    output_tokens: int = 250_000
    max_output_per_call: int = 16_000
    estimated_usd: float = 50.0
    max_calls: int = 300

    def __post_init__(self):
        for name in ("wall_seconds", "input_tokens", "output_tokens", "max_output_per_call", "max_calls"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.estimated_usd) or self.estimated_usd <= 0:
            raise ValueError("estimated_usd must be finite and positive")
        if self.max_output_per_call > 32_000:
            raise ValueError("Supported per-call maximum is 32,000 total output tokens; use the prospectively declared common cap")


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in asdict(self).values()):
            raise InvalidUsage("Provider token counts must be nonnegative integers")
        if self.cached_input_tokens + self.cache_write_tokens > self.input_tokens:
            raise InvalidUsage("Cached and written input exceeds total input")
        if self.cache_write_1h_tokens > self.cache_write_tokens or self.reasoning_tokens > self.output_tokens:
            raise InvalidUsage("Provider usage details exceed their total")

    @classmethod
    def from_provider(cls, provider: str, raw: dict):
        if not isinstance(raw, dict) or any(k not in raw for k in ("input_tokens", "output_tokens")):
            raise InvalidUsage("Response has no complete raw provider usage")
        if provider == "azure":
            inputs = raw.get("input_tokens_details") or {}
            outputs = raw.get("output_tokens_details") or {}
            return cls(raw["input_tokens"], raw["output_tokens"],
                       inputs.get("cached_tokens", 0), inputs.get("cache_write_tokens", 0),
                       0, outputs.get("reasoning_tokens", 0))
        if provider == "aws":
            read = raw.get("cache_read_input_tokens") or 0
            write = raw.get("cache_creation_input_tokens") or 0
            details = raw.get("output_tokens_details") or {}
            # Anthropic input_tokens excludes cache reads and cache writes;
            # output_tokens already INCLUDES thinking. Never add thinking twice.
            return cls(raw["input_tokens"] + read + write, raw["output_tokens"], read, write,
                       (raw.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0),
                       details.get("thinking_tokens", 0))
        raise ValueError("Unknown provider")


@dataclass(frozen=True)
class Prices:
    input_per_million: float
    cached_per_million: float
    cache_write_per_million: float
    cache_write_1h_per_million: float
    output_per_million: float
    description: str

    def cost(self, usage: Usage) -> float:
        uncached = usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
        short_writes = usage.cache_write_tokens - usage.cache_write_1h_tokens
        return (uncached * self.input_per_million
                + usage.cached_input_tokens * self.cached_per_million
                + short_writes * self.cache_write_per_million
                + usage.cache_write_1h_tokens * self.cache_write_1h_per_million
                + usage.output_tokens * self.output_per_million) / 1_000_000

    def upper_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Cache policy cannot make a request exceed the reservation.
        rate = max(self.input_per_million, self.cached_per_million,
                   self.cache_write_per_million, self.cache_write_1h_per_million)
        return (input_tokens * rate + output_tokens * self.output_per_million) / 1_000_000


PRICES = {
    "astra": Prices(10.0, 1.0, 12.5, 25.0, 50.0,
                    "Azure Global Standard short-context public rates; estimated, not an invoice"),
    "opus": Prices(4.4, .22, 5.5, 8.8, 22.0,
                   "Opus5.5 Bedrock US geographic profile; public rates including 10% premium"),
}


class Ledger:
    """One in-flight request; unresolved reservations survive failure unchanged."""
    def __init__(self, limits: Limits, prices: Prices, clock=time.monotonic, started=None):
        self.limits, self.prices, self.clock = limits, prices, clock
        self.started = clock() if started is None else started
        self.input_tokens = self.output_tokens = self.reasoning_tokens = self.calls = 0
        self.estimated_usd = 0.0
        self.reservation = None
        self.invalid = False

    def remaining_seconds(self):
        return max(0.0, self.limits.wall_seconds - (self.clock() - self.started))

    def reserve(self, input_upper_bound: int):
        if self.reservation:
            raise BudgetStop("An in-flight or unresolved request reservation exists")
        if self.invalid or self.remaining_seconds() < 2:
            raise BudgetStop("Invalid accounting or research deadline reached")
        if self.calls >= self.limits.max_calls:
            raise BudgetStop("Maximum model calls reached")
        if type(input_upper_bound) is not int or input_upper_bound < 1:
            raise ValueError("Require a positive integer input reservation")
        if self.input_tokens + input_upper_bound > self.limits.input_tokens:
            raise BudgetStop("Insufficient raw input token budget for conservative reservation")
        output_limit = min(self.limits.max_output_per_call, self.limits.output_tokens - self.output_tokens)
        if output_limit < 1:
            raise BudgetStop("Output token budget exhausted")
        reserve_cost = self.prices.upper_cost(input_upper_bound, output_limit)
        if self.estimated_usd + reserve_cost > self.limits.estimated_usd + 1e-12:
            raise BudgetStop("Insufficient dollar budget for worst-case request")
        self.calls += 1
        self.reservation = {"call": self.calls, "input_tokens": input_upper_bound,
                            "output_tokens": output_limit, "estimated_usd": reserve_cost}
        return dict(self.reservation)

    def settle(self, usage: Usage):
        if not self.reservation:
            raise InvalidUsage("Cannot settle an unreserved request")
        reservation = self.reservation
        cost = self.prices.cost(usage)
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.reasoning_tokens += usage.reasoning_tokens
        self.estimated_usd += cost
        self.reservation = None
        if (usage.input_tokens > reservation["input_tokens"]
                or usage.output_tokens > reservation["output_tokens"]
                or cost > reservation["estimated_usd"] + 1e-12):
            self.invalid = True
            raise InvalidUsage("Provider usage exceeded pre-call reservation; trial must be excluded")
        return cost

    def snapshot(self):
        return {"input_tokens_including_cache": self.input_tokens,
                "output_tokens_including_reasoning": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "estimated_provider_usd": self.estimated_usd,
                "provider_calls": self.calls,
                "elapsed_seconds": self.clock() - self.started,
                "remaining_seconds": self.remaining_seconds(),
                "unresolved_reservation": self.reservation,
                "accounting_valid": not self.invalid and self.reservation is None}
