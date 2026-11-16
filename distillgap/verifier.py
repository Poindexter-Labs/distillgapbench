"""Independent operator verification. The teacher never authors expected outputs:
they are computed here by executing the reference in the sandbox, twice, and the
reference must agree with itself (determinism) and with any claimed public outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json

from .sandbox import run_function, values_equal
from .schema import Task, parse_task, program_from_completion, render_prompt

VERIFIER_VERSION = "distillgap_verifier_v2_complexity_budget"
PER_CALL_SECONDS = 2.0


@dataclass
class LocalCheck:
    valid: bool
    reasons: list = field(default_factory=list)
    prompt: str | None = None
    expected_hidden: list | None = None       # canonical values aligned with task.hidden_inputs
    public_reprs: list | None = None
    detail: dict = field(default_factory=dict)

    def to_json(self):
        return {"valid": self.valid, "reasons": self.reasons, "prompt": self.prompt,
                "expected_hidden": self.expected_hidden, "public_reprs": self.public_reprs,
                "detail": self.detail}


def _claimed_matches(claimed, canonical_value) -> bool:
    # A claimed output is plain JSON; tuples/sets become lists in canonical form.
    def plain(v):
        if isinstance(v, dict) and "__dict__" in v:
            return {json.dumps(k) if not isinstance(k, str) else k: plain(x) for k, x in v["__dict__"]}
        if isinstance(v, dict) and "__set__" in v:
            return [plain(x) for x in v["__set__"]]
        if isinstance(v, list):
            return [plain(x) for x in v]
        return v
    return values_equal(plain(canonical_value), claimed)


def reference_ast_nodes(code: str) -> int:
    import ast
    return sum(1 for _ in ast.walk(ast.parse(code)))


def local_check(raw_task, max_ast_nodes: int | None = None) -> tuple[Task | None, LocalCheck]:
    task, errors = parse_task(raw_task)
    if errors:
        return task, LocalCheck(False, ["schema:" + e for e in errors])
    nodes = reference_ast_nodes(task.reference_solution)
    if max_ast_nodes is not None and nodes > max_ast_nodes:
        return task, LocalCheck(False, ["reference_exceeds_complexity_budget"],
                                detail={"ast_nodes": nodes, "budget": max_ast_nodes})
    inputs = task.public_inputs + task.hidden_inputs
    first = run_function(task.reference_solution, task.entry_point, inputs, per_call_seconds=PER_CALL_SECONDS)
    if first.status != "ok":
        reason = "reference_load_error" if first.status == "load_error" else "reference_" + first.status
        return task, LocalCheck(False, [reason], detail={"error": first.load_error or first.detail})
    failed = [i for i, c in enumerate(first.calls) if not (c and c.get("ok"))]
    if failed:
        call = first.calls[failed[0]] or {}
        return task, LocalCheck(False, ["reference_fails_on_own_input"],
                                detail={"input_index": failed[0], "class": call.get("cls"),
                                        "error": call.get("error"), "n_failed": len(failed)})
    second = run_function(task.reference_solution, task.entry_point, inputs, per_call_seconds=PER_CALL_SECONDS)
    if second.status != "ok" or any(
            not (b and b.get("ok")) or not values_equal(a["value"], b["value"])
            for a, b in zip(first.calls, second.calls)):
        return task, LocalCheck(False, ["reference_nondeterministic"])
    n_public = len(task.public_inputs)
    public_calls, hidden_calls = first.calls[:n_public], first.calls[n_public:]
    reasons = []
    if task.claimed_public_outputs is not None:
        for claimed, call in zip(task.claimed_public_outputs, public_calls):
            if not _claimed_matches(claimed, call["value"]):
                reasons.append("reference_disagrees_with_claimed_public_output")
                break
    hidden_values = [json.dumps(c["value"], sort_keys=True) for c in hidden_calls]
    if len(set(hidden_values)) == 1 and len(hidden_values) >= 3:
        reasons.append("hidden_outputs_constant")
    public_reprs = [c["repr"] for c in public_calls]
    prompt = render_prompt(task, list(zip(task.public_inputs, public_reprs)))
    leak = [s for s in (task.reference_solution.strip(),) if len(s) > 40 and s in prompt]
    if leak:
        reasons.append("prompt_leaks_reference")
    return task, LocalCheck(not reasons, reasons, prompt, [c["value"] for c in hidden_calls], public_reprs,
                            {"reference_wall_seconds": round(first.wall_seconds, 3), "ast_nodes": nodes})


def score_program(program: str, entry_point: str, inputs: list, expected: list) -> dict:
    """Run a complete program (target or validator) and compare against expected canonical values."""
    result = run_function(program, entry_point, inputs, per_call_seconds=PER_CALL_SECONDS)
    if result.status == "load_error":
        return {"passed": False, "outcome": "load_error", "error": result.load_error, "n_pass": 0,
                "n_tests": len(inputs)}
    if result.status == "harness_error":
        return {"passed": False, "outcome": "harness_error", "error": result.detail, "n_pass": 0,
                "n_tests": len(inputs), "infrastructure": True}
    n_pass, first_fail = 0, None
    for index, (call, want) in enumerate(zip(result.calls, expected)):
        ok = bool(call and call.get("ok") and values_equal(call["value"], want))
        n_pass += ok
        if not ok and first_fail is None:
            first_fail = {"index": index, "input": inputs[index],
                          "class": "wrong_answer" if call and call.get("ok") else (call or {}).get("cls", "crash"),
                          "got": (call or {}).get("repr") if call and call.get("ok") else (call or {}).get("error")}
    outcome = "pass" if n_pass == len(inputs) else (first_fail or {}).get("class", "fail")
    return {"passed": n_pass == len(inputs), "outcome": outcome, "n_pass": n_pass,
            "n_tests": len(inputs), "first_fail": first_fail}


def score_completion(prompt: str, completion: str, entry_point: str, inputs: list, expected: list) -> dict:
    program, status = program_from_completion(prompt, completion)
    if status != "ok":
        return {"passed": False, "outcome": status, "n_pass": 0, "n_tests": len(inputs),
                "body": program[len(prompt):][:1500]}
    scored = score_program(program, entry_point, inputs, expected)
    scored["body"] = program[len(prompt):][:1500]
    return scored


def agreement(reference: str, other: str, entry_point: str, inputs: list) -> dict:
    """Differential check of two independent programs on the same inputs.

    Inputs on which the reference raises are discarded (precondition outside the
    reference's domain) and counted; any other mismatch is disagreement.
    """
    ref = run_function(reference, entry_point, inputs, per_call_seconds=PER_CALL_SECONDS)
    oth = run_function(other, entry_point, inputs, per_call_seconds=PER_CALL_SECONDS)
    if ref.status != "ok":
        return {"agree": False, "reason": "reference_" + ref.status, "retained": 0}
    if oth.status == "load_error":
        return {"agree": False, "reason": "other_load_error", "error": oth.load_error, "retained": 0}
    if oth.status == "harness_error":
        return {"agree": False, "reason": "other_harness_error", "retained": 0, "infrastructure": True}
    retained, discarded, mismatch = [], 0, None
    for index, (rc, oc) in enumerate(zip(ref.calls, oth.calls)):
        if not (rc and rc.get("ok")):
            discarded += 1
            continue
        retained.append(index)
        if mismatch is None and not (oc and oc.get("ok") and values_equal(rc["value"], oc["value"])):
            mismatch = {"index": index, "input": inputs[index], "reference": rc.get("repr"),
                        "other": (oc or {}).get("repr") if oc and oc.get("ok") else (oc or {}).get("error")}
    return {"agree": mismatch is None and len(retained) >= 5, "retained": len(retained),
            "retained_indices": retained, "discarded_reference_errors": discarded,
            "mismatch": mismatch,
            "reason": "output_mismatch" if mismatch else (None if len(retained) >= 5 else "too_few_retained_inputs"),
            "expected": [ref.calls[i]["value"] for i in retained]}
