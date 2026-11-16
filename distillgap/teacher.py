"""Operator-owned, allowlisted, metered model broker (teacher and validator routes).

Routes are pinned: Astra = Azure gpt-6-astra (use_azure), Opus = AWS Bedrock
us.anthropic.claude-opus-5-5 in us-east-2. No provider fallback and no server- or
client-side refusal fallback: a refusal is recorded as a refusal, never silently
re-routed to a different model. Credentials are read into a local dict (never
os.environ, never logs, never task payloads); route identity logs endpoint hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import time

from .budget import BudgetStop, ModelMeter, input_upper_bound
from .vendor.rlenv_budget import InvalidUsage, Usage

def _find_repo() -> Path:
    """The repository root, found by its credential-helper file (independent of snapshot depth)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "server" / "claude_aws.py").is_file():
            return parent
    raise RuntimeError("repository root (server/claude_aws.py) not found above " + __file__)


try:
    REPO = _find_repo()
except RuntimeError:  # SovAI controller: no repository checkout; only the v6 routes (routes.py) are used there
    REPO = None
RATE_LIMIT_RETRIES = 10  # v1: 72 concurrent arms per researcher (v5: 6)
# v5: a call that delivered no answer because of infrastructure (outcome "ambiguous": a mid-stream provider error such
# as api_error/overloaded_error inside an HTTP 200, a dropped stream, or an unparseable response) is followed by up to
# INFRASTRUCTURE_RETRIES fresh calls. Each retry is a NEW call with its own reservation, call_id and log entries
# (linked by retry_of) and the failed call stays charged in full; nothing is ever replayed under the same call_id.
# Refusals, truncations, 4xx rejections and budget stops are outcomes, never retried. One rule for every arm.
INFRASTRUCTURE_RETRIES = 2
INFRASTRUCTURE_RETRY_PAUSE_SECONDS = 30
BROKER_VERSION = "distillgap_broker_v3_v1_rate_limit_10"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}


def collect_terminal_response(stream) -> dict:
    """Drain a raw Responses SSE stream; return the terminal response as a plain dict.

    Only the event type and the terminal event's `response` are read, so an event that the
    SDK could not fully type (left as a dict) cannot crash collection. No terminal event, or
    an `error` event, raises: the broker then classifies the call as ambiguous (charged in
    full, never replayed).
    """
    def field(obj, name):
        return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

    counts, final, error = {}, None, None
    try:
        for event in stream:
            kind = field(event, "type")
            counts[kind] = counts.get(kind, 0) + 1
            if kind in TERMINAL_EVENTS:
                final = field(event, "response")
            elif kind == "error":
                error = event if isinstance(event, dict) else getattr(event, "model_dump", lambda **_: str(event))()
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()
    if final is None:
        raise RuntimeError(f"stream ended without a terminal response event; events={counts}; error={str(error)[:300]}")
    raw = final if isinstance(final, dict) else final.model_dump(exclude_none=True)
    raw = dict(raw)
    raw["_stream_event_counts"] = counts
    return raw


class AstraAzureRoute:
    name, provider, model = "astra", "azure", "gpt-6-astra"

    def __init__(self, effort: str = "max", timeout_seconds: float = 1800):
        from openai import OpenAI
        module = _load_module("distillgap_azure_credentials", REPO / "tools/retro_chess/rl_env/azure.py")
        endpoint, key = module.credentials(provider="azure")
        self.effort, self.timeout = effort, timeout_seconds
        self.client = OpenAI(api_key=key, base_url=endpoint, timeout=timeout_seconds, max_retries=0)
        self.identity = {"route": self.name, "provider": "azure", "use_azure": True, "model": self.model,
                         "reasoning_effort": effort, "endpoint_sha256": _sha(endpoint.rstrip("/")),
                         "api": "responses", "streaming": True, "store": False, "fallback": False}

    def send(self, system: str, user: str, max_output: int) -> dict:
        # Streamed: a non-streamed MAX call held the connection silent for >10 min and was dropped
        # (demo-v1, APIConnectionError at 730 s). The stream emits a reasoning item every few seconds
        # (probe: max silence 11.4 s). Raw SSE iteration keeps only the terminal response event: the
        # SDK's snapshot helper crashed on one of six identical concurrent requests (pilot-v1 abort).
        stream = self.client.responses.create(
            model=self.model, instructions=system, input=[{"role": "user", "content": user}],
            reasoning={"effort": self.effort}, max_output_tokens=max_output, store=False, stream=True)
        return collect_terminal_response(stream)

    def parse(self, raw: dict) -> dict:
        text = "\n".join(block.get("text", "") for item in raw.get("output", [])
                         for block in item.get("content", []) or [] if block.get("type") == "output_text")
        status = raw.get("status")
        reason = (raw.get("incomplete_details") or {}).get("reason")
        if status == "completed":
            stop = "complete"
        elif reason == "max_output_tokens":
            stop = "truncated"
        elif reason == "content_filter":
            stop = "refusal"
        else:
            stop = "incomplete_" + str(reason or status)
        return {"text": text, "stop": stop, "usage": Usage.from_provider("azure", raw.get("usage")),
                "provider_model": raw.get("model"), "response_id_sha256": _sha(str(raw.get("id")))}

    def close(self):
        self.client.close()


class OpusAwsRoute:
    name, provider, model = "opus", "aws", "claude-opus-5-5"

    def __init__(self, effort: str = "max", timeout_seconds: float = 1800):
        from dotenv import dotenv_values
        module = _load_module("distillgap_claude_aws", REPO / "server/claude_aws.py")
        env = {k: v for k, v in dotenv_values(REPO / "server/.env").items() if v is not None}
        env.update(os.environ)
        env.update(CLAUDE_BEDROCK_REGION="us-east-2", CLAUDE_BEDROCK_MODEL_PREFIX="us")
        self.effort, self.timeout = effort, timeout_seconds
        self.client = module.create_aws_claude_client(env=env, timeout=timeout_seconds, max_retries=0)
        del env
        self.identity = {"route": self.name, "provider": "aws", "region": "us-east-2", "model": self.model,
                         "wire_model": "us.anthropic.claude-opus-5-5", "thinking": "adaptive",
                         "effort": effort, "streaming": True, "fallback": False, "refusal_fallback": False}

    def send(self, system: str, user: str, max_output: int) -> dict:
        request = {"model": self.model, "system": system, "max_tokens": max_output,
                   "messages": [{"role": "user", "content": user}],
                   "extra_body": {"output_config": {"effort": self.effort}, "thinking": {"type": "adaptive"}}}
        with self.client.messages.stream(**request) as stream:
            message = stream.get_final_message()
        return message.model_dump(exclude_none=True)

    def parse(self, raw: dict) -> dict:
        text = "\n".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        stop_reason = raw.get("stop_reason")
        stop = {"end_turn": "complete", "stop_sequence": "complete", "max_tokens": "truncated",
                "refusal": "refusal"}.get(stop_reason, "incomplete_" + str(stop_reason))
        return {"text": text, "stop": stop, "usage": Usage.from_provider("aws", raw.get("usage")),
                "provider_model": raw.get("model"), "response_id_sha256": _sha(str(raw.get("id")))}

    def close(self):
        self.client.close()


def make_route(name: str, effort: str = "max"):
    return {"astra": AstraAzureRoute, "opus": OpusAwsRoute}[name](effort=effort)


@dataclass
class CallResult:
    call_id: str
    outcome: str          # complete | truncated | refusal | incomplete_* | rejected | ambiguous | budget_stop
    text: str = ""
    usage: dict | None = None
    usd: float = 0.0
    latency_seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == "complete"


def _status_code(exc):
    code = getattr(exc, "status_code", None)
    if code is None and getattr(exc, "response", None) is not None:
        code = getattr(exc.response, "status_code", None)
    return code


class Broker:
    """Metered calls. 429/529 are retried within a call (never processed by the provider). Every other failure is
    final for that call and that call is never replayed; v5: if the call delivered no answer (ambiguous), up to
    INFRASTRUCTURE_RETRIES new, separately metered calls follow (see INFRASTRUCTURE_RETRIES)."""

    def __init__(self, route, meter: ModelMeter, log, sleep=time.sleep, infrastructure_retries=INFRASTRUCTURE_RETRIES):
        self.route, self.meter, self.log, self.sleep = route, meter, log, sleep
        self.infrastructure_retries = infrastructure_retries

    def call(self, purpose: str, system: str, user: str, **context) -> CallResult:
        result = self._call_once(purpose, system, user, **context)
        attempt = 0
        while result.outcome == "ambiguous" and attempt < self.infrastructure_retries:
            attempt += 1
            self.log.append("model_call_infrastructure_retry", failed_call_id=result.call_id, attempt=attempt,
                            purpose=purpose, meter=self.meter.name, error=(result.error or "")[:300])
            self.sleep(INFRASTRUCTURE_RETRY_PAUSE_SECONDS)
            retry = self._call_once(purpose, system, user, retry_of=result.call_id, retry_attempt=attempt, **context)
            if retry.outcome == "budget_stop":
                return result  # the meter cannot fund a retry: the round keeps its infrastructure outcome
            self.meter.outcomes["infrastructure_retries"] += 1
            result = retry
        return result

    def _call_once(self, purpose: str, system: str, user: str, **context) -> CallResult:
        try:
            reservation = self.meter.reserve(input_upper_bound(system, user))
        except BudgetStop as stop:
            self.log.append("model_call_budget_stop", purpose=purpose, meter=self.meter.name, reason=str(stop), **context)
            return CallResult("none", "budget_stop", error=str(stop))
        request_blob = self.log.put_blob({"system": system, "user": user})
        self.log.append("model_call_reserved", call_id=reservation["id"], purpose=purpose,
                        route=self.route.identity, reservation=reservation, request_blob=request_blob, **context)
        started = time.monotonic()
        attempt = 0
        while True:
            try:
                raw = self.route.send(system, user, reservation["output_tokens"])
                break
            except Exception as exc:  # noqa: BLE001 - every provider failure is classified
                code = _status_code(exc)
                message = (type(exc).__name__ + ": " + str(exc))[:600]
                if code in (429, 529) and attempt < RATE_LIMIT_RETRIES:
                    attempt += 1
                    self.meter.outcomes["rate_limited_retries"] += 1
                    self.log.append("model_call_rate_limited", call_id=reservation["id"], status=code, attempt=attempt)
                    self.sleep(min(120, 5 * 2 ** attempt))
                    continue
                latency = time.monotonic() - started
                if code is not None and 400 <= code < 500 and code not in (408, 409):
                    self.meter.not_billed(reservation["id"])
                    self.log.append("model_call_rejected", call_id=reservation["id"], status=code, error=message,
                                    latency_seconds=latency)
                    return CallResult(reservation["id"], "rejected", error=message, latency_seconds=latency)
                charged = self.meter.ambiguous(reservation["id"])
                self.log.append("model_call_ambiguous", call_id=reservation["id"], status=code, error=message,
                                charged_reservation=charged, latency_seconds=latency, replayed=False)
                return CallResult(reservation["id"], "ambiguous", error=message, latency_seconds=latency,
                                  usd=charged["usd"])
        latency = time.monotonic() - started
        response_blob = self.log.put_blob(raw)
        try:
            parsed = self.route.parse(raw)
        except (InvalidUsage, KeyError, TypeError, ValueError) as exc:
            charged = self.meter.ambiguous(reservation["id"])
            self.log.append("model_call_ambiguous", call_id=reservation["id"], error="unparseable usage: " + str(exc)[:300],
                            response_blob=response_blob, charged_reservation=charged, latency_seconds=latency, replayed=False)
            return CallResult(reservation["id"], "ambiguous", error=str(exc), latency_seconds=latency, usd=charged["usd"])
        settled = self.meter.settle(reservation["id"], parsed["usage"])
        usage = parsed["usage"].__dict__
        self.log.append("model_call_settled", call_id=reservation["id"], outcome=parsed["stop"], usage=usage,
                        usd=settled["usd"], exceeded_reservation=settled["exceeded_reservation"],
                        provider_model=parsed["provider_model"], response_blob=response_blob,
                        latency_seconds=latency, rate_limited_retries=attempt)
        return CallResult(reservation["id"], parsed["stop"], parsed["text"], usage, settled["usd"], latency)
