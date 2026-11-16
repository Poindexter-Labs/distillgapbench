"""Blocksworld planning (DistillGap v6). The researcher writes a start state and a goal; the student answers with a plan.

Instance: {"initial": [["B", "A"], ["C"]],          stacks listed bottom -> top, blocks are single capital letters
           "goal": [["C", "A"], ["A", "table"]]}     facts "X is on Y" (Y a block or "table"); partial goals allowed
Validity is decided by exact breadth-first search (optimal plan length). Student answer: one line,
"move X to Y; move Z to table; ...". Pass rule: every move is legal (X is clear; Y is a clear block or the table;
X is not already on the table when moved to the table) and every goal fact holds after the last move.
"""
from __future__ import annotations

from collections import deque
import random
import re

NAME = "planning"
MIN_BLOCKS, MAX_BLOCKS, MAX_OPTIMAL, MAX_STATES = 2, 8, 20, 500_000
MOVE_RE = re.compile(r"^move\s+([A-Z])\s+to\s+([A-Z]|table)$", re.I)


def canon(stacks):
    return frozenset(tuple(s) for s in stacks if s)


def parse(inst):
    initial, goal = inst.get("initial"), inst.get("goal")
    if not isinstance(initial, list) or not all(isinstance(s, list) and s for s in initial):
        raise ValueError("initial must be a list of non-empty stacks")
    blocks = [b for s in initial for b in s]
    if any(not (isinstance(b, str) and len(b) == 1 and b.isupper()) for b in blocks):
        raise ValueError("blocks must be single capital letters")
    if len(set(blocks)) != len(blocks):
        raise ValueError("a block appears twice")
    if not MIN_BLOCKS <= len(blocks) <= MAX_BLOCKS:
        raise ValueError(f"need {MIN_BLOCKS}-{MAX_BLOCKS} blocks")
    if not isinstance(goal, list) or not goal:
        raise ValueError("goal must be a non-empty list of [X, Y] facts")
    facts = set()
    for fact in goal:
        if not (isinstance(fact, list) and len(fact) == 2):
            raise ValueError("goal facts are [X, Y] pairs")
        x, y = fact
        if x not in blocks or (y != "table" and y not in blocks) or x == y:
            raise ValueError(f"goal fact refers to unknown blocks: {fact}")
        facts.add((x, y))
    return canon(initial), frozenset(facts), sorted(blocks)


def on_relation(state):
    rel = {}
    for s in state:
        rel[s[0]] = "table"
        for below, above in zip(s, s[1:]):
            rel[above] = below
    return rel


def satisfied(state, facts):
    rel = on_relation(state)
    return all(rel.get(x) == y for x, y in facts)


def apply(state, x, y):
    stacks = [list(s) for s in state]
    src = next((s for s in stacks if s[-1] == x), None)
    if src is None:
        return None  # x is not clear (or does not exist)
    if y == "table":
        if len(src) == 1:
            return None  # already on the table
        src.pop()
        stacks.append([x])
    else:
        dst = next((s for s in stacks if s[-1] == y), None)
        if dst is None or y == x:
            return None  # y not clear
        src.pop()
        dst.append(x)
    return canon(stacks)


def successors(state):
    tops = sorted(s[-1] for s in state)  # sorted: string hashing is randomized per process
    for x in tops:
        for y in tops + ["table"]:
            if y != x:
                nxt = apply(state, x, y)
                if nxt is not None:
                    yield f"move {x} to {y}", nxt


def solve(state, facts, max_states=MAX_STATES):
    if satisfied(state, facts):
        return 0, []
    parent = {state: None}
    frontier = deque([state])
    while frontier:
        cur = frontier.popleft()
        for move, nxt in successors(cur):
            if nxt in parent:
                continue
            parent[nxt] = (cur, move)
            if satisfied(nxt, facts):
                plan = []
                key = nxt
                while parent[key] is not None:
                    key, mv = parent[key]
                    plan.append(mv)
                return len(plan), list(reversed(plan))
            if len(parent) > max_states:
                return None, "state limit reached (unverifiable)"
            frontier.append(nxt)
    return None, "goal is unreachable (contradictory facts)"


def check(inst) -> dict:
    try:
        state, facts, blocks = parse(inst)
    except ValueError as exc:
        return {"valid": False, "reasons": [str(exc)]}
    optimal, plan = solve(state, facts)
    if optimal is None:
        return {"valid": False, "reasons": [plan]}
    if optimal == 0:
        return {"valid": False, "reasons": ["goal already satisfied"]}
    if optimal > MAX_OPTIMAL:
        return {"valid": False, "reasons": [f"optimal plan longer than {MAX_OPTIMAL}"]}
    return {"valid": True, "reasons": [], "optimal": optimal, "witness": "; ".join(plan), "blocks": len(blocks)}


def extract(completion):
    line = str(completion).strip().split("\n")[0].strip().rstrip(".")
    steps = [s.strip() for s in line.split(";") if s.strip()]
    if not steps or len(steps) > 60:
        return None
    out = []
    for s in steps:
        m = MOVE_RE.match(s)
        if not m:
            return None
        y = m.group(2)
        out.append((m.group(1).upper(), "table" if y.lower() == "table" else y.upper()))
    return out


def score(inst, completion) -> bool:
    plan = extract(completion)
    if plan is None:
        return False
    state, facts, _ = parse(inst)
    for x, y in plan:
        state = apply(state, x, y)
        if state is None:
            return False
    return satisfied(state, facts)


def describe_state(stacks):
    lines = []
    for s in sorted(stacks, key=lambda s: s[0]):
        lines.append(f"{s[0]} is on the table.")
        for below, above in zip(s, s[1:]):
            lines.append(f"{above} is on {below}.")
    return " ".join(lines)


def describe_goal(facts):
    return " ".join(f"{x} is on the table." if y == "table" else f"{x} is on {y}." for x, y in sorted(facts))


INTRO = ("Blocks world planning. Blocks sit on the table or on other blocks. A block is clear if nothing is on top of "
         "it. One move takes a clear block and puts it on the table or on another clear block. Write a plan that "
         "makes every goal fact true, on one line, as moves separated by semicolons, e.g. \"move A to table; move B to C\".")
SHOTS = [{"initial": [["A", "B"]], "goal": [["A", "B"]]},
         {"initial": [["C", "A"], ["B"]], "goal": [["C", "B"], ["B", "A"]]},
         {"initial": [["A", "B", "C"], ["D"]], "goal": [["D", "C"], ["A", "D"]]}]


def render_task(inst):
    state, facts, _ = parse(inst)
    return f"Initial state: {describe_state(state)}\nGoal: {describe_goal(facts)}"


def student_prompt(inst) -> dict:
    parts = [INTRO, ""]
    for shot in SHOTS:
        res = check(shot)
        if not res["valid"]:
            raise RuntimeError(f"few-shot example invalid: {shot}")
        parts += [render_task(shot), f"Plan: {res['witness']}", ""]
    parts += [render_task(inst), "Plan:"]
    return {"prompt": "\n".join(parts), "stop": ["\n"], "max_tokens": 160}


LEVELS = {1: dict(blocks=3, lo=1, hi=2, facts=(1, 2)), 2: dict(blocks=4, lo=3, hi=4, facts=(2, 3)),
          3: dict(blocks=5, lo=5, hi=6, facts=(2, 4)), 4: dict(blocks=6, lo=7, hi=9, facts=(3, 5)),
          5: dict(blocks=7, lo=10, hi=14, facts=(4, 7))}


def _random_stacks(rng, blocks):
    order = blocks[:]
    rng.shuffle(order)
    stacks = []
    for b in order:
        if stacks and rng.random() < 0.6:
            rng.choice(stacks).append(b)
        else:
            stacks.append([b])
    return stacks


def template(rng: random.Random, level: int, tries: int = 3000) -> dict:
    cfg = LEVELS[level]
    blocks = [chr(ord("A") + i) for i in range(cfg["blocks"])]
    for _ in range(tries):
        initial = _random_stacks(rng, blocks)
        target_rel = on_relation(canon(_random_stacks(rng, blocks)))
        k = rng.randint(*cfg["facts"])
        facts = [[x, target_rel[x]] for x in rng.sample(blocks, min(k, len(blocks)))]
        inst = {"initial": initial, "goal": facts}
        res = check(inst)
        if res["valid"] and cfg["lo"] <= res["optimal"] <= cfg["hi"]:
            return {**inst, "meta": {"optimal": res["optimal"], "blocks": cfg["blocks"], "level": level}}
    raise RuntimeError(f"no planning instance for difficulty {level}")
