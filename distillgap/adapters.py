"""DistillGap v1 environment adapters: everything the pair arm, the student interface and the final evaluation need
to know about an environment. Coding wraps the v5 logic unchanged; math and code output prediction are new.

Sandbox calls go through `sandbox.run_function` looked up at call time, so the SovAI namespace pool
(learning/rl/ns_pool.install_pool) applies to every execution.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from . import sandbox

ENVS_ROOT = Path(__file__).resolve().parents[1]
if str(ENVS_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVS_ROOT))
from envs import math_env, output_env  # noqa: E402

ADAPTERS_VERSION = "distillgap_v1_adapters_draft2_same_call"
PAIR_SIMILARITY_MIN = 0.4       # math: problem-text word 3-shingle Jaccard (same threshold as v5 descriptions)
CODE_SIMILARITY_MIN = 0.6       # output prediction: difflib ratio over normalised code lines


def _shingles(text: str, k: int = 3) -> set:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _norm_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in str(code).splitlines() if line.strip())


class Check:
    """Local (operator) check result: student prompt rendering plus whatever scoring needs."""

    def __init__(self, valid, reasons, detail=None, student=None, scoring=None, text=None):
        self.valid, self.reasons, self.detail = valid, reasons, detail or {}
        self.student = student or {}   # {"prompt", "stop", "max_tokens"}
        self.scoring = scoring or {}   # env-specific, operator-computed
        self.text = text or ""         # what pair similarity and dedup look at


# ----------------------------------------------------------------------------------------------- math
MATH_SYSTEM = """You are the challenge author in DistillGapBench, a research benchmark that measures how well a strong model can find the exact edge of a specific smaller model's ability.

Your job: write MINIMAL PAIRS of ORIGINAL, self-contained math problems with a single exact numerical answer. Each pair has a CONTROL that the target model solves and a CHALLENGE that is a minimal edit of the control which the target model fails. The pair scores only if the target passes the control (at least 2 of 4 samples) and fails the challenge (0 of 4), and both problems are independently verified.

TARGET MODEL: {target}

EXACT PROMPT FORMAT THE TARGET SEES (operator-rendered; four worked examples precede it):
Problem: <your problem text>
Solution:
The target writes its reasoning and must end with "Final answer: <number>". An answer counts only if the number after "Final answer:" equals your answer exactly (integers, fractions p/q or terminating decimals).

HOW YOUR PROBLEMS ARE CHECKED (by the operator; you will not see the final evaluation):
 1. Your solution program is executed in a sandbox: it must define solve() returning the answer as an int, or as a string "p/q" for a fraction, and the result must equal your stated answer exactly.
 2. An independent strong model from a different provider sees ONLY the problem text and solves it; its answer must equal yours. So the problem must fully determine one exact answer: state units, ranges, inclusivity, rounding (prefer exact answers; never ask to round), and what is being counted.
 3. A reviewer flags ambiguity, typos, or problems that are wrong; flagged problems never count.
 4. The challenge is a minimal edit of the control (most of the wording is shared) with a different answer. A challenge that is a different problem, or that changes only the numbers without changing what makes it hard, rarely crosses the edge; a challenge that adds a genuine reasoning step, a constraint, a case split or a counting subtlety often does.
Ambiguous, trick, unsolvable or wrong problems never count, even if the target fails them. Do not copy or lightly paraphrase problems from existing benchmarks or competitions (GSM8K, MATH, AIME, AMC, textbooks, websites). Every pair you write counts in the denominator.

HARD CONSTRAINTS FOR EACH PROBLEM
- problem: 20-1500 characters of plain text (no LaTeX needed; ^ for powers is fine).
- answer: an exact integer, or a fraction "p/q" in lowest terms, magnitude at most 10^12.
- solution_program: Python 3.12 source defining solve() with no arguments; standard library only; deterministic; no I/O; at most 2 seconds. Brute force is welcome.
- solution: a short written solution (at most 1200 characters) that a reviewer can check.

OUTPUT FORMAT: return ONLY one JSON object, no markdown fences, no commentary:
{{"pairs": [{{"pair_family": "short topic label", "changed_requirement": "what the challenge changes and why it should cross the target's edge", "control": {{"problem": "...", "answer": "...", "solution_program": "def solve():\\n    ...", "solution": "..."}}, "challenge": {{"problem": "...", "answer": "...", "solution_program": "...", "solution": "..."}}, "derived_from": ["pair ids this builds on, if any"], "rationale": "why the target passes the control but fails the challenge (not shown to the target)"}}]}}
"""

MATH_VALIDATOR_SYSTEM = """You are an independent expert mathematician acting as a verifier for a research benchmark. You are given ONLY a problem. Solve it carefully and exactly; check your work (brute-force mentally or by reasoning where possible). If the problem is ambiguous or ill-posed (more than one defensible answer, missing information, contradictory), say so.

Return ONLY one JSON object, no markdown fences:
{"answer": "<exact integer or p/q>", "ambiguous": false, "ambiguity_note": ""}"""


class MathAdapter:
    name = "math"
    feedback_heads = ("problem",)

    def teacher_system(self, target_description: str) -> str:
        return MATH_SYSTEM.format(target=target_description)

    def local_check(self, raw) -> tuple[dict | None, Check]:
        if not isinstance(raw, dict):
            return None, Check(False, ["schema: task is not an object"])
        inst = {"problem": raw.get("problem"), "answer": str(raw.get("answer", "")).strip()}
        base = math_env.check(inst)
        reasons = list(base["reasons"])
        program = raw.get("solution_program")
        detail = {}
        if not isinstance(program, str) or "def solve" not in program:
            reasons.append("solution_program must define solve()")
        if not reasons:
            res = sandbox.run_function(program, "solve", [[]], per_call_seconds=2.0)
            call = (res.calls or [None])[0] if res.status == "ok" else None
            if res.status != "ok" or not call or not call.get("ok"):
                reasons.append(f"solution_program failed: {res.status} {str(res.load_error or (call or {}).get('error', ''))[:200]}")
            else:
                got = math_env.parse_number(str(call["value"]))
                detail["program_answer"] = str(call["value"])[:60]
                if got is None or got != math_env.parse_number(inst["answer"]):
                    reasons.append("solution_program result differs from the stated answer")
        if reasons:
            return None, Check(False, reasons, detail)
        student = math_env.student_prompt(inst)
        return inst, Check(True, [], detail, student=student,
                           scoring={"answer": str(math_env.parse_number(inst["answer"]))}, text=inst["problem"])

    def pair_reasons(self, ctrl, ch_ctrl, chal, ch_chal) -> tuple[list, float]:
        similarity = round(_jaccard(_shingles(ctrl["problem"]), _shingles(chal["problem"])), 4)
        reasons = []
        if similarity < PAIR_SIMILARITY_MIN:
            reasons.append("pair_not_minimal")
        if ch_ctrl.scoring["answer"] == ch_chal.scoring["answer"]:
            reasons.append("pair_identical_behavior")
        return reasons, similarity

    def dedup_key(self, inst, check) -> str:
        return hashlib.sha256(" ".join(re.findall(r"[a-z0-9]+", inst["problem"].lower())).encode()).hexdigest()

    def score(self, scoring, completion) -> dict:
        got = math_env.extract(completion)
        expected = math_env.parse_number(scoring["answer"])
        if got is None:
            return {"passed": False, "outcome": "no_final_answer"}
        return {"passed": got == expected, "outcome": "pass" if got == expected else "wrong_answer", "got": str(got)}

    def example_failure(self, scoring, sample, completion) -> dict:
        return {"target_final_answer": sample.get("got"), "outcome": sample.get("outcome"),
                "solution_tail": str(completion)[-700:]}

    def template_pair(self, rng: random.Random, index: int) -> dict:
        """Operator baseline: the same family and the same random draws, the challenge one difficulty level up
        (same wording, harder parameters or the family's harder variant)."""
        family = math_env.FAMILIES[index % len(math_env.FAMILIES)]
        level = 1 + index % 3
        pair = {}
        for _ in range(50):
            seed = rng.randrange(2 ** 31)
            drawn = {role: family(random.Random(seed), level + (role == "challenge")) for role in ("control", "challenge")}
            if all(a is not None for _, a in drawn.values()) and drawn["control"][1] != drawn["challenge"][1]:
                break
        for role in ("control", "challenge"):
            problem, answer = drawn[role]
            ans = math_env.fmt(answer)
            program = f"def solve():\n    return {ans!r}" if "/" in ans else f"def solve():\n    return {ans}"
            pair[role] = {"problem": problem, "answer": ans, "solution_program": program, "solution": "operator template"}
        return {"pair_family": family.__name__, "changed_requirement": "operator template parameter redraw", **pair}

    def validator_system(self) -> str:
        return MATH_VALIDATOR_SYSTEM

    def validator_message(self, inst) -> str:
        return "PROBLEM:\n" + inst["problem"]

    def validator_agrees(self, scoring, parsed) -> dict:
        got = math_env.parse_number(str(parsed.get("answer", "")))
        return {"agree": got is not None and got == math_env.parse_number(scoring["answer"]), "validator_answer": str(got)}

    def judge_inputs(self, raw) -> tuple[str, str, str]:
        """(problem, author solution, light-path variant) for the Poindexter light path."""
        solution = f"{raw.get('solution', '')}\n\nFinal answer: {raw.get('answer')}"
        return str(raw.get("problem", "")), solution, "math"


# ------------------------------------------------------------------------------------ output prediction
OUTPUT_SYSTEM = """You are the challenge author in DistillGapBench, a research benchmark that measures how well a strong model can find the exact edge of a specific smaller model's ability.

Your job: write MINIMAL PAIRS of ORIGINAL, small Python functions f with one call each. The target model must predict the exact value f returns, without running code. Each pair has a CONTROL whose output the target predicts correctly and a CHALLENGE, a minimal edit of the control, whose output it gets wrong. The pair scores only if the target is right on the control (at least 2 of 4 samples) and wrong on the challenge (0 of 4).

TARGET MODEL: {target}

EXACT PROMPT FORMAT THE TARGET SEES (operator-rendered; three worked examples precede it):
[PYTHON]
<your code>
assert <your call> ==
The target completes the line with a Python literal. It counts as correct only if ast.literal_eval of its answer has exactly the same repr as the true return value.

HOW YOUR TASKS ARE CHECKED: the operator executes your code in a sandbox (twice) and uses that output as the only truth; you never state outputs. The output must be a plain literal (int, float, str, bool, None, or lists/tuples/dicts/sets of these) with a repr of at most 200 characters.
The challenge must be a minimal edit of the control's CODE with the IDENTICAL call (same function name and the same literal arguments): mostly the same code (an added or changed line, a changed condition, an extra step) with a different output. Changing only the arguments does not count. Aim for genuine tracing difficulty (state across iterations, index arithmetic, aliasing and mutation, nested control flow, string/list semantics), not obscure trivia or traps about Python versions. Do not copy benchmark tasks (CRUXEval, HumanEval, textbooks). Every pair counts in the denominator.

HARD CONSTRAINTS FOR EACH TASK
- code: Python 3.12 defining a top-level function f (helper functions and constant assignments allowed), at most 40 lines; no imports, I/O, print, randomness, time, eval/exec/open, classes, lambdas, try/with, global/nonlocal, or names/attributes starting with an underscore; deterministic; at most 2 seconds.
- call: exactly "f(<literal arguments>)" with positional literal arguments only.

OUTPUT FORMAT: return ONLY one JSON object, no markdown fences, no commentary:
{{"pairs": [{{"pair_family": "short topic label", "changed_requirement": "what the challenge changes and why it should cross the target's edge", "control": {{"code": "def f(x):\\n    ...", "call": "f(...)"}}, "challenge": {{"code": "...", "call": "f(...)"}}, "derived_from": ["pair ids this builds on, if any"], "rationale": "why the target gets the control right and the challenge wrong (not shown to the target)"}}]}}
"""


def sandbox_output_executor(code: str, call: str) -> dict:
    """Operator wrapper: run `repr(<call>)` inside the operator sandbox so the exact repr survives."""
    wrapper = code + "\n\n\ndef dg_operator_output():\n    return repr(" + call + ")\n"
    res = sandbox.run_function(wrapper, "dg_operator_output", [[]], per_call_seconds=output_env.TIME_LIMIT)
    call_result = (res.calls or [None])[0] if res.status == "ok" else None
    if res.status != "ok" or not call_result:
        return {"outcome": "timeout" if "timeout" in str(res.status) else "error",
                "detail": str(res.load_error or res.status)[:200]}
    if not call_result.get("ok"):
        return {"outcome": "error", "detail": str(call_result.get("error", ""))[:200]}
    text = call_result["value"]
    try:
        literal = repr(ast.literal_eval(text)) == text
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        literal = False
    return {"outcome": "ok", "repr": text, "literal": literal}


class OutputAdapter:
    name = "output"
    feedback_heads = ("code", "call")

    def teacher_system(self, target_description: str) -> str:
        return OUTPUT_SYSTEM.format(target=target_description)

    def local_check(self, raw, executor=None) -> tuple[dict | None, Check]:
        if not isinstance(raw, dict):
            return None, Check(False, ["schema: task is not an object"])
        inst = {"code": str(raw.get("code", "")), "call": str(raw.get("call", "")).strip()}
        res = output_env.check(inst, executor=executor or sandbox_output_executor)
        if not res["valid"]:
            return None, Check(False, res["reasons"])
        inst["expected_repr"] = res["expected_repr"]
        return inst, Check(True, [], {"expected_repr": res["expected_repr"]}, student=output_env.student_prompt(inst),
                           scoring={"expected_repr": res["expected_repr"]}, text=inst["code"] + "\n" + inst["call"])

    def pair_reasons(self, ctrl, ch_ctrl, chal, ch_chal) -> tuple[list, float]:
        similarity = round(difflib.SequenceMatcher(None, _norm_code(ctrl["code"]), _norm_code(chal["code"])).ratio(), 4)
        reasons = []
        same_call = ast.dump(ast.parse(ctrl["call"], mode="eval")) == ast.dump(ast.parse(chal["call"], mode="eval"))
        if not same_call:
            reasons.append("pair_call_differs")   # the pair must differ in code only (smoke v1: argument-only pairs)
        if similarity < CODE_SIMILARITY_MIN:
            reasons.append("pair_not_minimal")
        if ch_ctrl.scoring["expected_repr"] == ch_chal.scoring["expected_repr"]:
            reasons.append("pair_identical_behavior")
        return reasons, similarity

    def dedup_key(self, inst, check) -> str:
        return hashlib.sha256((_norm_code(inst["code"]) + "\n" + inst["call"].replace(" ", "")).encode()).hexdigest()

    def score(self, scoring, completion) -> dict:
        got = output_env.extract(completion)
        if got is None:
            return {"passed": False, "outcome": "unparseable_literal"}
        ok = got == scoring["expected_repr"]
        return {"passed": ok, "outcome": "pass" if ok else "wrong_value", "got": got[:200]}

    def example_failure(self, scoring, sample, completion) -> dict:
        return {"target_answer": sample.get("got") or str(completion).strip()[:200], "expected": scoring["expected_repr"],
                "outcome": sample.get("outcome")}

    def template_pair(self, rng: random.Random, index: int) -> dict:
        """Operator baseline: a one-step function and the same function with one more step appended."""
        level = 1 + index % 2
        control = output_env.template(rng, level, executor=sandbox_output_executor)
        lines = control["code"].split("\n")
        is_str = control["call"].startswith("f('") or control["call"].startswith('f("')
        for _ in range(50):
            extra = rng.choice(output_env._str_ops(rng) if is_str else output_env._list_ops(rng))
            code = "\n".join(lines[:-1] + [f"    {extra}", lines[-1]])
            res = output_env.check({"code": code, "call": control["call"]}, executor=sandbox_output_executor)
            if res["valid"] and res["expected_repr"] != control["expected_repr"]:
                break
        return {"pair_family": f"template_{'str' if is_str else 'list'}_ops", "changed_requirement": "operator: one extra step",
                "control": {"code": control["code"], "call": control["call"]}, "challenge": {"code": code, "call": control["call"]}}

    # Operator execution is the ground truth: no validator, no judge.
    def validator_system(self):
        return None

    def judge_inputs(self, raw):
        return None


ADAPTERS = {"math": MathAdapter(), "output": OutputAdapter()}
