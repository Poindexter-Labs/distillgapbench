"""Pinned researcher routes for DistillGap v6 (12 researchers, 5 provider families). No provider fallback, no
refusal fallback: a refusal is an outcome. Long calls stream so idle connections are never dropped.

Credentials (release): read once by load_credentials() from the file named by DISTILLGAP_ENV_FILE (default: .env in
the working directory, then ~/.distillgap/provider_keys.env), with process environment variables taking precedence.
See .env.example. Values are kept in a local dict and never logged, never sent to student sandboxes or task payloads.
Identity records hash the endpoint.

Every route returns raw = provider response as a plain dict, and parse(raw) -> {"text", "stop", "usage": Usage,
"provider_model", "response_id_sha256"} with stop in {complete, truncated, refusal, incomplete_*}.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import requests

HERE = Path(__file__).resolve().parent
try:  # same class objects as the broker (distillgap.teacher catches this InvalidUsage); identical file to v5's
    from distillgap.vendor.rlenv_budget import InvalidUsage, Usage
except ImportError:  # standalone use (smoke probes)
    _spec = importlib.util.spec_from_file_location("distillgap_v6_usage", HERE.parent / "v5/distillgap/vendor/rlenv_budget.py")
    _budget = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _budget  # dataclasses with postponed annotations look their module up by name
    _spec.loader.exec_module(_budget)
    Usage, InvalidUsage = _budget.Usage, _budget.InvalidUsage
REGISTRY = json.loads((HERE / "researchers_v6.json").read_text())["researchers"]


def _sha(text: str) -> str:
    return hashlib.sha256(str(text).encode()).hexdigest()


def load_credentials() -> dict:
    """Provider credentials for the researcher, validator and judge routes. A missing key is None: only the routes
    you actually call need one (the operator template on output prediction needs none)."""
    import os
    candidates = [os.environ.get("DISTILLGAP_ENV_FILE"), ".env", str(Path.home() / ".distillgap/provider_keys.env")]
    env = {}
    for name in candidates:
        if name and Path(name).is_file():
            from dotenv import dotenv_values
            env = {k: v for k, v in dotenv_values(name).items() if v}
            break
    env.update({k: v for k, v in os.environ.items() if v})
    return {"azure_key": env.get("AZURE_OPENAI_KEY"), "azure_base_url": env.get("DISTILLGAP_AZURE_BASE_URL"),
            "bedrock_token": env.get("AWS_BEARER_TOKEN_BEDROCK_LONG") or env.get("AWS_BEARER_TOKEN_BEDROCK"),
            "gemini_key": env.get("GEMINI_API_KEY"), "meta_key": env.get("META_API_KEY"),
            "openrouter_key": env.get("OPENROUTER_API_KEY")}


class ProviderHTTPError(RuntimeError):
    """A non-200 provider response. Carries status_code so the broker classifies it (429/529 rate limits are retried
    with backoff inside the call; other 4xx are rejections; 5xx are infrastructure). v1 fix: plain RuntimeError hid
    Bedrock's 429s and turned them into lost rounds (void Kimi run, job 8414622)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


class Route:
    provider = "?"

    def __init__(self, spec: dict, creds: dict, timeout: float = 1800):
        self.spec, self.timeout = spec, timeout
        self.name, self.model, self.effort = spec["id"], spec["wire_model"], spec["effort"]
        self.identity = {"route": self.name, "provider": spec["provider"], "model": self.model, "effort": self.effort,
                         "streaming": True, "fallback": False, "refusal_fallback": False}

    def close(self):
        pass


class AzureResponsesRoute(Route):
    """Astra, Sol, GPT-5.6 on Azure (Responses API, raw SSE; keep only the terminal event)."""

    def __init__(self, spec, creds, timeout=1800):
        super().__init__(spec, creds, timeout)
        from openai import OpenAI
        self.client = OpenAI(api_key=creds["azure_key"], base_url=creds["azure_base_url"], timeout=timeout, max_retries=0)
        self.identity.update(use_azure=True, api="responses", store=False,
                             endpoint_sha256=_sha(creds["azure_base_url"].rstrip("/")))

    def send(self, system, user, max_output):
        stream = self.client.responses.create(model=self.model, instructions=system, input=[{"role": "user", "content": user}],
                                              reasoning={"effort": self.effort}, max_output_tokens=max_output,
                                              store=False, stream=True)
        final, counts = None, {}
        try:
            for event in stream:
                kind = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)
                counts[kind] = counts.get(kind, 0) + 1
                if kind in {"response.completed", "response.incomplete", "response.failed"}:
                    final = getattr(event, "response", None) or event.get("response")
        finally:
            stream.close()
        if final is None:
            raise RuntimeError(f"stream ended without a terminal response event; events={counts}")
        raw = final if isinstance(final, dict) else final.model_dump(exclude_none=True)
        return {**raw, "_stream_event_counts": counts}

    def parse(self, raw):
        text = "\n".join(b.get("text", "") for item in raw.get("output", []) for b in (item.get("content") or [])
                         if b.get("type") == "output_text")
        status, reason = raw.get("status"), (raw.get("incomplete_details") or {}).get("reason")
        stop = ("complete" if status == "completed" else "truncated" if reason == "max_output_tokens"
                else "refusal" if reason == "content_filter" else f"incomplete_{reason or status}")
        return {"text": text, "stop": stop, "usage": Usage.from_provider("azure", raw.get("usage")),
                "provider_model": raw.get("model"), "response_id_sha256": _sha(raw.get("id"))}

    def close(self):
        self.client.close()


class BedrockMessagesRoute(Route):
    """Opus 5.5, Opus 4.8, Sonnet 5.5, Fable 5.2 on AWS Bedrock us-east-2 (full wire ids; adaptive thinking + effort)."""

    def __init__(self, spec, creds, timeout=1800):
        super().__init__(spec, creds, timeout)
        if "fable-5-1" in self.model:
            raise ValueError("Fable 5.1 is excluded by the operator handoff")
        import anthropic
        self.client = anthropic.Anthropic(api_key=creds["bedrock_token"], auth_token="", timeout=timeout, max_retries=0,
                                          base_url="https://bedrock-runtime.us-east-2.amazonaws.com/anthropic")
        self.client.auth_token = None  # never inherit ANTHROPIC_AUTH_TOKEN on an AWS request
        self.identity.update(region="us-east-2", thinking="adaptive")

    def send(self, system, user, max_output):
        with self.client.messages.stream(model=self.model, system=system, max_tokens=max_output,
                                         messages=[{"role": "user", "content": user}],
                                         extra_body={"output_config": {"effort": self.effort},
                                                     "thinking": {"type": "adaptive"}}) as stream:
            return stream.get_final_message().model_dump(exclude_none=True)

    def parse(self, raw):
        text = "\n".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        sr = raw.get("stop_reason")
        stop = {"end_turn": "complete", "stop_sequence": "complete", "max_tokens": "truncated",
                "refusal": "refusal"}.get(sr, f"incomplete_{sr}")
        return {"text": text, "stop": stop, "usage": Usage.from_provider("aws", raw.get("usage")),
                "provider_model": raw.get("model"), "response_id_sha256": _sha(raw.get("id"))}

    def close(self):
        self.client.close()


class GeminiRoute(Route):
    """Gemini 4 Argon / Carbon / Hydrogen, native API, streamed generateContent (SSE)."""
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"

    def __init__(self, spec, creds, timeout=1800):
        super().__init__(spec, creds, timeout)
        self.key = creds["gemini_key"]
        self.identity.update(api="streamGenerateContent", thinking_level=self.effort)

    def send(self, system, user, max_output):
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"maxOutputTokens": max_output, "thinkingConfig": {"thinkingLevel": self.effort}}}
        texts, last = [], {}
        with requests.post(self.URL.format(model=self.model), headers={"x-goog-api-key": self.key}, json=body,
                           stream=True, timeout=(30, self.timeout)) as r:
            if r.status_code != 200:
                raise ProviderHTTPError(r.status_code, r.text[:300].replace(self.key, '[redacted]'))
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                last = chunk
                for cand in chunk.get("candidates", [])[:1]:
                    for part in (cand.get("content") or {}).get("parts", []):
                        if not part.get("thought") and "text" in part:
                            texts.append(part["text"])
        cand = (last.get("candidates") or [{}])[0]
        return {"text": "".join(texts), "finishReason": cand.get("finishReason"), "usageMetadata": last.get("usageMetadata"),
                "modelVersion": last.get("modelVersion"), "responseId": last.get("responseId")}

    def parse(self, raw):
        fr = raw.get("finishReason")
        stop = {"STOP": "complete", "MAX_TOKENS": "truncated", "SAFETY": "refusal", "PROHIBITED_CONTENT": "refusal",
                "RECITATION": "refusal", "BLOCKLIST": "refusal", "SPII": "refusal"}.get(fr, f"incomplete_{fr}")
        u = raw.get("usageMetadata") or {}
        if "promptTokenCount" not in u:
            raise InvalidUsage("Gemini response has no usage metadata")
        thoughts = u.get("thoughtsTokenCount", 0)
        usage = Usage(u["promptTokenCount"], u.get("candidatesTokenCount", 0) + thoughts,
                      u.get("cachedContentTokenCount", 0), 0, 0, thoughts)
        return {"text": raw["text"], "stop": stop, "usage": usage, "provider_model": raw.get("modelVersion"),
                "response_id_sha256": _sha(raw.get("responseId"))}


class OpenRouterRoute(Route):
    """Kimi K3.1 Pro via OpenRouter, pinned to Moonshot AI's own endpoint (no fallbacks), streamed."""
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, spec, creds, timeout=1800):
        super().__init__(spec, creds, timeout)
        self.key, self.pin = creds["openrouter_key"], spec["provider_pin"]
        self.identity.update(provider_pin=self.pin)

    def send(self, system, user, max_output):
        body = {"model": self.model, "max_tokens": max_output, "reasoning": {"effort": self.effort}, "provider": self.pin,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": True, "usage": {"include": True}}
        texts, finish, usage, model, upstream, rid = [], None, None, None, None, None
        with requests.post(self.URL, headers={"Authorization": "Bearer " + self.key}, json=body, stream=True,
                           timeout=(30, self.timeout)) as r:
            if r.status_code != 200:
                raise ProviderHTTPError(r.status_code, r.text[:300].replace(self.key, '[redacted]'))
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if "error" in chunk:
                    raise RuntimeError(f"mid-stream error: {str(chunk['error'])[:300]}")
                model, upstream, rid = chunk.get("model", model), chunk.get("provider", upstream), chunk.get("id", rid)
                for ch in chunk.get("choices", []):
                    texts.append((ch.get("delta") or {}).get("content") or "")
                    finish = ch.get("finish_reason") or finish
                usage = chunk.get("usage") or usage
        return {"text": "".join(texts), "finish_reason": finish, "usage": usage, "model": model, "upstream": upstream, "id": rid}

    def parse(self, raw):
        if raw.get("upstream") not in (None, "Moonshot AI"):
            raise RuntimeError(f"pinned route served by {raw.get('upstream')}")
        fr = raw.get("finish_reason")
        stop = {"stop": "complete", "length": "truncated", "content_filter": "refusal"}.get(fr, f"incomplete_{fr}")
        u = raw.get("usage") or {}
        if "prompt_tokens" not in u:
            raise InvalidUsage("OpenRouter response has no usage")
        usage = Usage(u["prompt_tokens"], u["completion_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                      0, 0, (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        return {"text": raw["text"], "stop": stop, "usage": usage, "provider_model": f"{raw.get('model')} via {raw.get('upstream')}",
                "response_id_sha256": _sha(raw.get("id"))}


class BedrockChatRoute(OpenRouterRoute):
    """Non-Anthropic models on AWS Bedrock (us-east-2) through Bedrock's OpenAI-compatible Chat Completions endpoint,
    with the same long-lived Bedrock API key as the Anthropic routes, streamed. Used for Kimi K3.1 (us.moonshotai.kimi-k3.1,
    US inference profile). reasoning_effort is validated by Bedrock (none/minimal/low/medium/high/xhigh/max)."""
    URL = "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1/chat/completions"

    def __init__(self, spec, creds, timeout=1800):
        Route.__init__(self, spec, creds, timeout)
        self.key, self.pin = creds["bedrock_token"], None
        self.identity.update(region="us-east-2", api="bedrock-openai-chat-completions", streaming=True)

    def send(self, system, user, max_output):
        body = {"model": self.model, "max_tokens": max_output, "reasoning_effort": self.effort,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": True, "stream_options": {"include_usage": True}}
        texts, finish, usage, model, rid = [], None, None, None, None
        with requests.post(self.URL, headers={"Authorization": "Bearer " + self.key}, json=body, stream=True,
                           timeout=(30, self.timeout)) as r:
            if r.status_code != 200:
                raise ProviderHTTPError(r.status_code, r.text[:300].replace(self.key, '[redacted]'))
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if "error" in chunk:
                    raise RuntimeError(f"mid-stream error: {str(chunk['error'])[:300]}")
                model, rid = chunk.get("model", model), chunk.get("id", rid)
                for ch in chunk.get("choices", []):
                    texts.append((ch.get("delta") or {}).get("content") or "")
                    finish = ch.get("finish_reason") or finish
                usage = chunk.get("usage") or usage
        return {"text": "".join(texts), "finish_reason": finish, "usage": usage, "model": model, "upstream": None, "id": rid}

    def parse(self, raw):
        if raw.get("model") not in (None, self.model):
            raise RuntimeError(f"Bedrock served {raw.get('model')}, expected {self.model}")
        out = super().parse(raw)
        out["provider_model"] = f"{raw.get('model')} via aws-bedrock us-east-2"
        return out


class MetaRoute(Route):
    """Muse Wonder 1.0, native Meta Responses API in background mode (create once, poll the same id)."""
    BASE = "https://api.meta.ai/v1/responses"

    def __init__(self, spec, creds, timeout=1800):
        super().__init__(spec, creds, timeout)
        self.key = creds["meta_key"]
        self.identity.update(api="responses-background")

    def send(self, system, user, max_output):
        h = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}
        body = {"model": self.model, "instructions": system, "reasoning": {"effort": self.effort},
                "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}],
                "max_output_tokens": max_output, "background": True, "store": True}
        r = requests.post(self.BASE, headers=h, json=body, timeout=(10, 60))
        if r.status_code >= 300:
            raise ProviderHTTPError(r.status_code, r.text[:300].replace(self.key, '[redacted]'))
        j = r.json()
        deadline = time.monotonic() + self.timeout
        while j.get("status") in {"queued", "in_progress"}:
            if time.monotonic() > deadline:
                requests.post(f"{self.BASE}/{j['id']}/cancel", headers=h, timeout=(10, 30))
                raise TimeoutError("Meta background response exceeded the route timeout (cancelled)")
            time.sleep(5)
            j = requests.get(f"{self.BASE}/{j['id']}", headers=h, timeout=(10, 60)).json()
        return j

    def parse(self, raw):
        text = "\n".join(b.get("text", "") for item in raw.get("output", []) for b in (item.get("content") or [])
                         if b.get("type") == "output_text")
        status, reason = raw.get("status"), (raw.get("incomplete_details") or {}).get("reason")
        stop = ("complete" if status == "completed" else "truncated" if reason == "max_output_tokens"
                else "refusal" if reason == "content_filter" else f"incomplete_{reason or status}")
        return {"text": text, "stop": stop, "usage": Usage.from_provider("azure", raw.get("usage")),
                "provider_model": raw.get("model"), "response_id_sha256": _sha(raw.get("id"))}


ROUTES = {"azure": AzureResponsesRoute, "aws_bedrock": BedrockMessagesRoute, "aws_bedrock_openai": BedrockChatRoute, "google_native": GeminiRoute,
          "openrouter": OpenRouterRoute, "meta_native": MetaRoute}


def make_route(researcher_id: str, creds: dict | None = None, effort: str | None = None):
    spec = next(r for r in REGISTRY if r["id"] == researcher_id)
    if effort:
        spec = {**spec, "effort": effort}
    return ROUTES[spec["provider"]](spec, creds or load_credentials())
