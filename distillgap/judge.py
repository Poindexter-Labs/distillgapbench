"""Layer-3 judge for DistillGap v1: the Poindexter auto-review light path, reproduced request-for-request.

Same prompt builders (vendored verbatim: vendor/light_path.py), same schema, same model and settings as
server/phoenix.generate_light_review[_generic] in the default 'legacy' quality mode: Azure gpt-6-astra, Responses
API, reasoning effort "high", no tools, input = [system (joined with " "), user (joined with " ")], Structured
Outputs json_schema "light_review" strict. Used as a black-box correctness/ambiguity reviewer.

VETO rule (v1): ambiguity verdict true. The light path is Poindexter's training-data authoring rubric, so its final
verdict also encodes authoring style ("discard" for problems it finds too easy, "major issues" for solutions without
2-3 reasoning steps). In the smoke run it discarded every easy control while rating every challenge "perfect".
DistillGap controls are meant to be easy, and correctness is enforced by the independent validator, so only
ambiguity vetoes. STRICT_VETO (final verdict "major issues" or "discard", or ambiguous) is the reported sensitivity
variant. Vetoes are recomputed from the stored judge outputs at report time.
A judge call that fails or returns unparseable output is recorded as judge_incomplete (no veto, reported).
"""
from __future__ import annotations

import json
import time

try:  # Poindexter's internal reviewer prompts are not part of the public release; see docs/JUDGE.md
    from .vendor.light_path import LIGHT_REVIEW_SCHEMA, get_light_review, get_light_review_generic
except ImportError:  # run_grid.py --judge none (the default when the module is absent)
    LIGHT_REVIEW_SCHEMA = get_light_review = get_light_review_generic = None

JUDGE_VERSION = "distillgap_v1_light_path_judge_2_ambiguity_veto"
MODEL, EFFORT = "gpt-6-astra", "high"
VETO_VERDICTS = {"major issues", "discard"}


def build_request(problem: str, solution: str, variant: str) -> dict:
    if get_light_review is None:
        raise RuntimeError("distillgap/vendor/light_path.py is not installed; run with --judge none (docs/JUDGE.md)")
    builder = get_light_review if variant == "math" else get_light_review_generic
    system, user = builder(problem, solution)
    return {"model": MODEL, "input": [{"role": "system", "content": " ".join(system)},
                                      {"role": "user", "content": " ".join(user)}],
            "reasoning": {"effort": EFFORT}, "tools": None,
            "text": {"format": {"type": "json_schema", "name": "light_review", "strict": True, "schema": LIGHT_REVIEW_SCHEMA}}}


def verdict(output: dict | None) -> dict:
    if not isinstance(output, dict) or "final verdict" not in output:
        return {"status": "judge_incomplete", "veto": False}
    final = str(output.get("final verdict", "")).strip().lower()
    ambiguous = output.get("ambiguity verdict") is True
    return {"status": "reviewed", "veto": ambiguous, "strict_veto": final in VETO_VERDICTS or ambiguous,
            "final_verdict": final, "ambiguous": ambiguous}


class LightPathJudge:
    def __init__(self, client, meter=None, log=None):
        """client: an openai.OpenAI client on the Azure endpoint (routes.load_credentials)."""
        self.client, self.meter, self.log = client, meter, log

    def review(self, problem: str, solution: str, variant: str, candidate_id: str) -> dict:
        request = build_request(problem, solution, variant)
        start = time.time()
        try:
            resp = self.client.responses.create(**request)
            output = json.loads(resp.output_text)
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
            error = None
        except Exception as exc:  # noqa: BLE001 - recorded as judge_incomplete, never a veto
            output, usage, error = None, None, (type(exc).__name__ + ": " + str(exc))[:300]
        row = {"candidate_id": candidate_id, "judge_version": JUDGE_VERSION, "variant": variant,
               "latency_seconds": round(time.time() - start, 1), "usage": usage, "error": error,
               "output": output, **verdict(output)}
        if self.log is not None:
            self.log.append("judge_review", **row)
        return row
