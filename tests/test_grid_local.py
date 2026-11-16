"""End-to-end fixture test of the v1 grid runner: coding, math and output cells (adaptive + batch + template) through
discovery, final evaluation, judge and report. No provider calls, no network, no GPU (fake routes, fixture student,
fake judge); the operator sandbox runs a few tiny programs.

Run: python -m pytest -q tests   (from the repository root)
"""
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from distillgap.adapters import MATH_VALIDATOR_SYSTEM  # noqa: E402
from distillgap.prompts import VALIDATOR_SYSTEM  # noqa: E402
from distillgap.target import FixtureTarget  # noqa: E402
from distillgap.vendor.rlenv_budget import Usage  # noqa: E402
import run_grid  # noqa: E402

# ---------------------------------------------------------------- coding (v5 fixtures)
REF = "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n"
REF_Y = "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiouy')\n"
HIDDEN = [["yay"], [""], ["aeiou"], ["bcd"], ["AbE"], ["queue"], ["Yy"]]


def coding_task(description, reference):
    return {"family": "strings", "entry_point": "count_vowels", "signature": "def count_vowels(s: str) -> int:",
            "description": description, "reference_solution": reference, "public_inputs": [["Hello"]],
            "claimed_public_outputs": [2], "hidden_inputs": HIDDEN}


CODING_PAIR = {"pair_family": "vowels", "changed_requirement": "y also counts",
               "control": coding_task("Return the number of vowels (a, e, i, o, u; case-insensitive) in s.", REF),
               "challenge": coding_task("Return the number of vowels (a, e, i, o, u and also y; case-insensitive) in s.", REF_Y)}

# ---------------------------------------------------------------- math
MATH_PAIR = {"pair_family": "counting", "changed_requirement": "adds a derived quantity",
             "control": {"problem": "A box has 3 red and 4 blue balls. How many balls are in the box in total?",
                         "answer": "7", "solution_program": "def solve():\n    return 3 + 4", "solution": "3+4=7"},
             "challenge": {"problem": "A box has 3 red and 4 blue balls and twice as many green balls as red balls. "
                                      "How many balls are in the box in total?",
                           "answer": "13", "solution_program": "def solve():\n    return 3 + 4 + 2 * 3",
                           "solution": "3+4+6=13"}}
MATH_BAD = {"pair_family": "bad", "changed_requirement": "wrong key",
            "control": {"problem": "What is 2 plus 2 when you add the two numbers together carefully?", "answer": "4",
                        "solution_program": "def solve():\n    return 4", "solution": "4"},
            "challenge": {"problem": "What is 2 plus 3 when you add the two numbers together carefully?", "answer": "6",
                          "solution_program": "def solve():\n    return 5", "solution": "wrong"}}
MATH_ANSWERS = {MATH_PAIR["control"]["problem"]: "7", MATH_PAIR["challenge"]["problem"]: "13"}

# ---------------------------------------------------------------- output prediction
OUT_CONTROL = "def f(xs):\n    out = []\n    for x in xs:\n        out.append(x * 2)\n    return out"
OUT_CHALLENGE = "def f(xs):\n    out = []\n    for x in xs:\n        out.append(x * 2)\n    out.reverse()\n    return out"
OUTPUT_PAIR = {"pair_family": "lists", "changed_requirement": "reverse at the end",
               "control": {"code": OUT_CONTROL, "call": "f([1, 2, 3])"},
               "challenge": {"code": OUT_CHALLENGE, "call": "f([1, 2, 3])"}}


class FakeRoute:
    identity = {"route": "fake"}

    def __init__(self):
        self.sent = []

    def send(self, system, user, max_output):
        self.sent.append((system, user))
        if system == MATH_VALIDATOR_SYSTEM:
            problem = user.split("PROBLEM:\n", 1)[1]
            return {"text": json.dumps({"answer": MATH_ANSWERS.get(problem, "0"), "ambiguous": False})}
        if system == VALIDATOR_SYSTEM:
            solution = REF_Y if "also y" in user else REF
            return {"text": json.dumps({"solution": solution, "extra_inputs": [["yy"], ["AEIOUY"]], "ambiguous": False})}
        if "math problems" in system:
            return {"text": json.dumps({"pairs": [MATH_BAD if "ROUND 2" in user else MATH_PAIR]})}
        if "Python functions f" in system:
            return {"text": json.dumps({"pairs": [OUTPUT_PAIR]})}
        return {"text": json.dumps({"pairs": [CODING_PAIR]})}

    def parse(self, raw):
        return {"text": raw["text"], "stop": "complete", "usage": Usage(500, 300), "provider_model": "fake",
                "response_id_sha256": "x"}


def student(prompt, n, seed):
    if prompt.rstrip().endswith("Solution:"):            # math: right on the control, wrong on the challenge
        return [" 3 + 4 = 7. Final answer: 7"] * n
    if prompt.rstrip().endswith("=="):                    # output: always the control's value
        return [" [2, 4, 6]"] * n
    return ["    return sum(1 for ch in s.lower() if ch in 'aeiou')\n"] * n   # coding: control yes, challenge no


class FakeJudge:
    def __init__(self):
        self.calls = []

    def review(self, problem, solution, variant, candidate_id):
        self.calls.append((variant, problem[:40]))
        veto = "green" in problem   # ambiguity flag -> veto (the v1 rule); stored output drives the report
        output = {"final verdict": "perfect", "ambiguity verdict": veto}
        return {"candidate_id": candidate_id, "status": "reviewed", "veto": veto, "variant": variant, "usage": None,
                "output": output, "final_verdict": "perfect"}


def make_grid(out):
    grid = object.__new__(run_grid.Grid)
    grid.args = argparse.Namespace(out=str(out), final_concurrency=4, arms_per_researcher=2, judge="light_path")
    grid._lock, grid._arm_slots = __import__("threading").Lock(), {}
    grid.fake_routes, grid.fake_judge = [], FakeJudge()
    grid.route = lambda rid, effort=None: grid.fake_routes.append(FakeRoute()) or grid.fake_routes[-1]
    grid.target = lambda student_id, env: FixtureTarget(student)
    grid.judge = lambda: grid.fake_judge
    return grid


OVERRIDES = {"max_candidates": 2, "per_call_tasks": 1, "max_teacher_calls": 2, "target_query_limit": 32}


def cell(env, kind_list, researcher="astra"):
    return {"cell_id": f"{researcher}__qwen4b__{env}", "researcher": researcher, "student": "qwen4b", "env": env,
            "reps": [1], "arms": kind_list, "arm_overrides": OVERRIDES}


def test_coding_cell(tmp):
    grid = make_grid(tmp)
    report = grid.run_cell(cell("coding", ["adaptive", "batch"]))
    arms = {a["arm_id"]: a for a in report["per_arm"]}
    # each arm: 2 identical pairs; the second is a duplicate challenge -> 1 frontier gap per arm
    assert arms["rep1-adaptive"]["frontier_gaps"] == 1, arms["rep1-adaptive"]
    assert arms["rep1-batch"]["frontier_gaps"] == 1
    assert arms["rep1-adaptive"]["frontier_gaps_judged"] == 1
    assert all(v == "generic" for v, _ in grid.fake_judge.calls)
    second_round = [u for s, u in grid.fake_routes[0].sent][1]
    assert "control_target" in second_round or "control_target" in [u for s, u in grid.fake_routes[1].sent][1]
    # validator for an Azure researcher is Opus (cross-family): the coding validator prompt was used
    assert any(s == VALIDATOR_SYSTEM for r in grid.fake_routes for s, _ in r.sent)


def test_math_cell(tmp):
    grid = make_grid(tmp)
    report = grid.run_cell(cell("math", ["adaptive", "batch"]))
    arms = {a["arm_id"]: a for a in report["per_arm"]}
    ad = arms["rep1-adaptive"]
    assert ad["pairs_attempted"] == 2 and ad["pairs_locally_valid"] == 1, ad
    assert ad["pair_local_failure_reasons"] == {"challenge_invalid": 1}
    assert ad["frontier_gaps"] == 1 and ad["frontier_gaps_judged"] == 0 and ad["judge_vetoed_ids"]
    # adaptive round 2 saw the student's attempts on round 1; batch never did
    adaptive_route = next(r for r in grid.fake_routes if any("Operator feedback" in u for _, u in r.sent))
    feedback = [u for _, u in adaptive_route.sent if "Operator feedback" in u][0]
    assert '"control_target"' in feedback and "target_final_answer" in feedback
    for route in grid.fake_routes:
        for system, user in route.sent:
            if "receive NO information" in user:
                assert "control_target" not in user
    validated = [json.loads(l) for l in (tmp / "astra__qwen4b__math/final/final_results.jsonl").read_text().splitlines()]
    assert sum(1 for r in validated if r.get("validator_answer")) == 4   # both tasks of each arm's would-be gap
    assert {r["validity"] for r in validated} == {"valid"}


def test_output_cell(tmp):
    grid = make_grid(tmp)
    report = grid.run_cell(cell("output", ["adaptive", "batch"]))
    arms = {a["arm_id"]: a for a in report["per_arm"]}
    assert arms["rep1-adaptive"]["pairs_attempted"] == 2
    assert arms["rep1-adaptive"]["frontier_gaps"] == 1, arms["rep1-adaptive"]   # second pair is a duplicate
    assert arms["rep1-adaptive"]["pair_local_failure_reasons"] == {"challenge_duplicate": 1}
    assert arms["rep1-adaptive"]["frontier_gaps_judged"] == 1   # no judge for output prediction
    assert not grid.fake_judge.calls


def test_template_cells(tmp):
    grid = make_grid(tmp)
    for env in ("math", "output"):
        c = {"cell_id": f"template__qwen4b__{env}", "researcher": None, "student": "qwen4b", "env": env, "reps": [1],
             "arms": ["template"], "arm_overrides": {**OVERRIDES, "max_candidates": 3}}
        report = grid.run_cell(c)
        arm = report["per_arm"][0]
        assert arm["pairs_attempted"] == 3 and arm["pairs_locally_valid"] >= 2, (env, arm)


def test_resume_skips_frozen_work(tmp):
    grid = make_grid(tmp)
    c = cell("output", ["adaptive"])
    grid.run_cell(c)
    calls = sum(len(r.sent) for r in grid.fake_routes)
    grid.run_cell(c)
    assert sum(len(r.sent) for r in grid.fake_routes) == calls


def test_output_pairs_must_keep_the_call(tmp):
    from distillgap.adapters import ADAPTERS
    a = ADAPTERS["output"]
    ic, cc = a.local_check({"code": OUT_CONTROL, "call": "f([])"})
    ih, ch = a.local_check({"code": OUT_CONTROL, "call": "f([1, 2, 3])"})
    reasons, _ = a.pair_reasons(ic, cc, ih, ch)
    assert "pair_call_differs" in reasons, reasons
    ih2, ch2 = a.local_check({"code": OUT_CHALLENGE, "call": "f([1,2,3])"})
    ic2, cc2 = a.local_check({"code": OUT_CONTROL, "call": "f([1, 2, 3])"})
    assert a.pair_reasons(ic2, cc2, ih2, ch2)[0] == []


def test_route_http_429_is_retried_as_rate_limit(tmp):
    """v1 bug (void Kimi run 8414622): a 429 raised without status_code became an 'ambiguous' lost round."""
    import routes
    from distillgap.budget import ModelLimits, ModelMeter
    from distillgap.ledger import EvidenceLog
    from distillgap.teacher import Broker

    class Flaky:
        identity = {"route": "flaky"}

        def __init__(self):
            self.n = 0

        def send(self, system, user, max_output):
            self.n += 1
            if self.n <= 3:
                raise routes.ProviderHTTPError(429, "Too many tokens, please wait before trying again.")
            return {"text": "ok"}

        def parse(self, raw):
            return {"text": raw["text"], "stop": "complete", "usage": Usage(10, 5), "provider_model": "x",
                    "response_id_sha256": "x"}

    meter = ModelMeter("m", ModelLimits(max_calls=5, max_input_tokens=10**6, max_output_tokens=10**6,
                                        max_output_per_call=4000, max_usd=10), run_grid.prices_for("kimi31"))
    result = Broker(Flaky(), meter, EvidenceLog(tmp), sleep=lambda s: None).call("x", "s", "u")
    assert result.outcome == "complete" and meter.outcomes["rate_limited_retries"] == 3, (result, meter.outcomes)
    assert meter.outcomes.get("ambiguous", 0) == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                try:
                    fn(Path(d))
                    print("PASS", name)
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    failures += 1
                    print("FAIL", name, type(exc).__name__, exc)
                    traceback.print_exc()
    sys.exit(1 if failures else 0)
