"""Blocksworld planning with a short written reasoning step before the plan (calibration variant of planning_env).

Same instances, validity check, solver and pass rule as planning_env; only the student prompt differs: few-shot
examples show one or two sentences of reasoning, then "Plan: ...". The answer is the last "Plan:" line.
"""
from __future__ import annotations

from . import planning_env as base

NAME = "planning_cot"
LEVELS = base.LEVELS
check, template, render_task = base.check, base.template, base.render_task

REASONING = [
    "B is on A, but A must end up on B. Move B to the table so A is clear, then put A on B.",
    "The goal stack is C on B on A, so A must stand on the table. A is on C, so move A to the table. Then put B on A, "
    "and finally C on B.",
    "A must end up on D, and D on C. A is buried under B and C, so move C to the table, then B to the table. Now put D "
    "on C, then A on D.",
]


def extract(completion):
    lines = [ln.strip() for ln in str(completion).split("\n")]
    plans = [ln[len("Plan:"):] for ln in lines if ln.startswith("Plan:")]
    return base.extract(plans[-1]) if plans else None


def score(inst, completion) -> bool:
    plan = extract(completion)
    if plan is None:
        return False
    state, facts, _ = base.parse(inst)
    for x, y in plan:
        state = base.apply(state, x, y)
        if state is None:
            return False
    return base.satisfied(state, facts)


def student_prompt(inst) -> dict:
    intro = base.INTRO.replace("Write a plan", "Think briefly about which blocks must move first, then write a plan")
    parts = [intro, ""]
    for shot, reasoning in zip(base.SHOTS, REASONING):
        res = check(shot)
        if not res["valid"]:
            raise RuntimeError(f"few-shot example invalid: {shot}")
        parts += [render_task(shot), f"Reasoning: {reasoning}", f"Plan: {res['witness']}", ""]
    parts += [render_task(inst), "Reasoning:"]
    return {"prompt": "\n".join(parts), "stop": ["\nInitial state:", "\n\n"], "max_tokens": 320}
