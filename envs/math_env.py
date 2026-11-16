"""Exact-answer math (DistillGap v6). The researcher writes a problem with an exact answer; the student reasons and
ends with "Final answer: X".

Instance: {"problem": str, "answer": "<integer or p/q>", "solution_program": "<python printing the answer>"}
In the study the solution program is run in the operator sandbox and must print exactly the claimed answer; a
cross-family validator also solves the problem blind (independence: the author's claim is never the only check).
Template instances are generated with brute-force-computed answers. Student pass rule: the text after the first
"Final answer:" parses to a number exactly equal to the answer (integers, fractions p/q, or exact decimals).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
import math
import random
import re

NAME = "math"
FINAL_RE = re.compile(r"final answer\s*[:=]\s*(.+)", re.I)


def parse_number(text) -> Fraction | None:
    s = str(text).strip()
    m = re.search(r"\\boxed\{([^{}]*)\}", s)
    if m:
        s = m.group(1)
    s = s.split("\n")[0].strip().rstrip(".").replace("$", "").replace(",", "").replace(" ", "")
    s = s.replace("\\dfrac", "\\frac")
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        s = f"{m.group(1)}/{m.group(2)}"
    try:
        if re.fullmatch(r"-?\d+/-?\d+", s):
            p, q = s.split("/")
            return Fraction(int(p), int(q)) if int(q) else None
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def check(inst) -> dict:
    reasons = []
    problem, answer = inst.get("problem"), inst.get("answer")
    if not isinstance(problem, str) or not 20 <= len(problem) <= 1500:
        reasons.append("problem must be 20-1500 characters")
    value = parse_number(answer) if answer is not None else None
    if value is None:
        reasons.append("answer must be an exact integer, fraction p/q or terminating decimal")
    elif abs(value) > 10 ** 12:
        reasons.append("answer magnitude above 10^12")
    return {"valid": not reasons, "reasons": reasons, **({"value": str(value)} if value is not None else {})}


def extract(completion):
    m = FINAL_RE.search(str(completion))
    return parse_number(m.group(1)) if m else None


def score(inst, completion) -> bool:
    got = extract(completion)
    return got is not None and got == parse_number(inst["answer"])


INTRO = "Solve each problem. Reason step by step, then end with \"Final answer:\" followed by the exact number."
SHOTS = [
    ("A box holds 12 pencils. A teacher buys 7 boxes and gives 3 pencils to each of 25 students. How many pencils are left?",
     "The teacher has 7 * 12 = 84 pencils and gives away 25 * 3 = 75. 84 - 75 = 9.", "9"),
    ("What is the remainder when 2^20 is divided by 7?",
     "2^3 = 8 leaves remainder 1 mod 7, so 2^18 = (2^3)^6 leaves 1. Then 2^20 = 2^18 * 4 leaves 4.", "4"),
    ("How many integers from 1 to 100 are divisible by 4 or 6?",
     "Multiples of 4: 25. Multiples of 6: 16. Multiples of both (12): 8. 25 + 16 - 8 = 33.", "33"),
    ("The sum of two numbers is 30 and their difference is 8. What is their product?",
     "The numbers are (30 + 8) / 2 = 19 and (30 - 8) / 2 = 11. 19 * 11 = 209.", "209"),
]


def student_prompt(inst) -> dict:
    parts = [INTRO, ""]
    for problem, solution, answer in SHOTS:
        parts += [f"Problem: {problem}", f"Solution: {solution} Final answer: {answer}", ""]
    parts += [f"Problem: {inst['problem']}", "Solution:"]
    return {"prompt": "\n".join(parts), "stop": ["\nProblem:", "\n\n\n"], "max_tokens": 400}


# ---- Operator template families: (rng, level) -> (problem, answer). Answers computed by brute force. ----
NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goran", "Hana", "Ivo", "Jun"]
ITEMS = [("notebook", "notebooks"), ("pen", "pens"), ("apple", "apples"), ("ticket", "tickets"), ("candle", "candles")]


def fam_shopping(rng, lv):
    (a1, a2), (b1, b2) = rng.sample(ITEMS, 2)
    pa, pb = rng.randint(2, 6 + 4 * lv), rng.randint(2, 6 + 4 * lv)
    x, y = rng.randint(2, 3 + 2 * lv), rng.randint(2, 3 + 2 * lv)
    cost = pa * x + pb * y
    paid = (cost // 10 + rng.randint(1, 3)) * 10
    name = rng.choice(NAMES)
    if lv >= 3:
        disc = rng.choice([10, 20, 25, 50])
        total = Fraction(cost * (100 - disc), 100)
        paid = (math.ceil(total) // 10 + rng.randint(1, 3)) * 10
        return (f"A {a1} costs ${pa} and a {b1} costs ${pb}. {name} buys {x} {a2} and {y} {b2}, gets {disc}% off the "
                f"whole purchase, and pays with ${paid}. How many dollars of change does {name} receive?", paid - total)
    return (f"A {a1} costs ${pa} and a {b1} costs ${pb}. {name} buys {x} {a2} and {y} {b2} and pays with ${paid}. "
            f"How many dollars of change does {name} receive?", paid - cost)


def fam_power_mod(rng, lv):
    a, b, m = rng.randint(2, 3 + 3 * lv), rng.randint(5, 10 ** min(lv, 4) + 10), rng.choice([7, 9, 11, 13, 17, 19, 23, 37, 101][:3 + lv])
    return f"What is the remainder when {a}^{b} is divided by {m}?", pow(a, b, m)


def fam_divisible(rng, lv):
    n = rng.randint(50, 100 * 10 ** min(lv, 3))
    a, b = rng.sample([2, 3, 4, 5, 6, 7, 9, 10, 11, 12], 2)
    if lv >= 4:
        c = rng.choice([x for x in [2, 3, 5, 7, 11, 13] if x not in (a, b)])
        count = sum(1 for k in range(1, n + 1) if (k % a == 0 or k % b == 0) and k % c != 0)
        return f"How many integers from 1 to {n} are divisible by {a} or {b} but not by {c}?", count
    count = sum(1 for k in range(1, n + 1) if k % a == 0 or k % b == 0)
    return f"How many integers from 1 to {n} are divisible by {a} or {b}?", count


def fam_sum_diff(rng, lv):
    x, y = rng.randint(3, 20 * lv), rng.randint(3, 20 * lv)
    if lv >= 3:
        k = rng.randint(2, 4)
        return (f"{rng.choice(NAMES)} is {k} times as old as her nephew. In {y} years she will be {x + 1} years older "
                f"than twice his age then. How old is the nephew now?", _age_solve(k, y, x + 1))
    return f"The sum of two numbers is {x + y} and their difference is {abs(x - y)}. What is their product?", x * y


def _age_solve(k, y, d):
    # k*n + y = 2*(n + y) + d  ->  n = (y + d) / (k - 2); fall back to a different statement when not integral
    for n in range(1, 500):
        if k * n + y == 2 * (n + y) + d:
            return n
    return None


def fam_series(rng, lv):
    k = rng.randint(3, 4 + 2 * lv)
    lo, hi = rng.randint(1, 50), rng.randint(100, 100 * 3 ** min(lv, 4))
    return f"What is the sum of all multiples of {k} between {lo} and {hi}, inclusive?", sum(v for v in range(lo, hi + 1) if v % k == 0)


def fam_digit_sum_count(rng, lv):
    n = rng.choice([100, 500, 1000, 5000, 10000][: 2 + lv // 2 + 1])
    s = rng.randint(3, 9 + 2 * lv)
    return (f"How many positive integers less than {n} have digits that add up to {s}?",
            sum(1 for v in range(1, n) if sum(map(int, str(v))) == s))


def fam_committee(rng, lv):
    men, women = rng.randint(3, 4 + lv), rng.randint(3, 4 + lv)
    k = rng.randint(2, min(3 + lv // 2, men + women - 1))
    need = rng.randint(1, min(k, women))
    people = ["M"] * men + ["W"] * women
    count = sum(1 for c in combinations(range(men + women), k) if sum(people[i] == "W" for i in c) >= need)
    return (f"A club has {men} men and {women} women. In how many ways can a committee of {k} people be chosen "
            f"that includes at least {need} women?", count)


def fam_grid_paths(rng, lv):
    w, h = rng.randint(2, 2 + lv), rng.randint(2, 2 + lv)
    if lv >= 3:
        bx, by = rng.randint(1, w - 1), rng.randint(1, h - 1)
        ways = {}
        for i in range(w + 1):
            for j in range(h + 1):
                if (i, j) == (bx, by):
                    ways[(i, j)] = 0
                elif i == 0 and j == 0:
                    ways[(i, j)] = 1
                else:
                    ways[(i, j)] = ways.get((i - 1, j), 0) + ways.get((i, j - 1), 0)
        return (f"Moving only right or up one unit at a time, how many paths go from (0, 0) to ({w}, {h}) without "
                f"passing through the point ({bx}, {by})?", ways[(w, h)])
    return (f"Moving only right or up one unit at a time, how many paths go from (0, 0) to ({w}, {h})?",
            math.comb(w + h, w))


def fam_dice(rng, lv):
    dice, target = 2 + (lv >= 3), None
    faces = range(1, 7)
    totals = [sum(r) for r in product(faces, repeat=dice)]
    target = rng.randint(dice + 1, 6 * dice - 1)
    p = Fraction(sum(t == target for t in totals), len(totals))
    if lv >= 4:
        p = Fraction(sum(t >= target for t in totals), len(totals))
        return (f"{dice} fair six-sided dice are rolled. What is the probability that the total is at least {target}? "
                "Give the answer as a fraction in lowest terms.", p)
    return (f"{dice} fair six-sided dice are rolled. What is the probability that the total is exactly {target}? "
            "Give the answer as a fraction in lowest terms.", p)


FAMILIES = [fam_shopping, fam_power_mod, fam_divisible, fam_sum_diff, fam_series, fam_digit_sum_count,
            fam_committee, fam_grid_paths, fam_dice]
LEVELS = {lv: {} for lv in range(1, 6)}


def fmt(value):
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def template(rng: random.Random, level: int, tries: int = 100) -> dict:
    for _ in range(tries):
        fam = rng.choice(FAMILIES)
        problem, answer = fam(rng, level)
        if answer is None:
            continue
        inst = {"problem": problem, "answer": fmt(answer)}
        if check(inst)["valid"]:
            return {**inst, "meta": {"family": fam.__name__, "level": level}}
    raise RuntimeError("no math instance")
