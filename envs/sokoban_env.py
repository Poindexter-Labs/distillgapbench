"""Sokoban levels (DistillGap v6). The researcher writes a level; the student answers with one line of moves.

Instance: {"level": "<rows joined by newline>"} using # wall, ' ' floor, . goal, $ box, * box on goal,
@ player, + player on goal. Validity is decided by an exact breadth-first solver (optimal move count), never by
the author's claim. Student pass rule: the answer line is a sequence of U/D/L/R; every move is legal and the
final state has every box on a goal.
"""
from __future__ import annotations

from collections import deque
import random

NAME = "sokoban"
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
CHARS = set("# .$*@+")
MAX_SIDE, MAX_BOXES, MAX_OPTIMAL, MAX_STATES, MAX_ANSWER = 10, 4, 40, 400_000, 200


def parse(level: str):
    rows = [r.rstrip() for r in str(level).strip("\n").split("\n")]
    if not rows or any(set(r) - CHARS for r in rows):
        raise ValueError("unknown characters or empty level")
    h, w = len(rows), max(len(r) for r in rows)
    if h > MAX_SIDE or w > MAX_SIDE:
        raise ValueError(f"level larger than {MAX_SIDE}x{MAX_SIDE}")
    rows = [r.ljust(w, "#") for r in rows]
    walls, goals, boxes, players = set(), set(), set(), []
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch == "#":
                walls.add((i, j))
            if ch in ".*+":
                goals.add((i, j))
            if ch in "$*":
                boxes.add((i, j))
            if ch in "@+":
                players.append((i, j))
    if len(players) != 1:
        raise ValueError("level must contain exactly one player")
    # The playable region must be enclosed: no floor cell reachable from the player may lie on the border.
    seen, todo = {players[0]}, [players[0]]
    while todo:
        i, j = todo.pop()
        if i in (0, h - 1) or j in (0, w - 1):
            raise ValueError("level is not enclosed by walls")
        for di, dj in MOVES.values():
            nxt = (i + di, j + dj)
            if nxt not in walls and nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return {"walls": frozenset(walls), "goals": frozenset(goals), "boxes": frozenset(boxes), "player": players[0],
            "shape": (h, w), "reachable": frozenset(seen)}


def render(walls, goals, boxes, player, shape) -> str:
    h, w = shape
    out = []
    for i in range(h):
        row = ""
        for j in range(w):
            p = (i, j)
            if p in walls:
                row += "#"
            elif p == player:
                row += "+" if p in goals else "@"
            elif p in boxes:
                row += "*" if p in goals else "$"
            else:
                row += "." if p in goals else " "
        out.append(row.rstrip() or "#")
    return "\n".join(out)


def step(walls, boxes, player, move):
    di, dj = MOVES[move]
    nxt = (player[0] + di, player[1] + dj)
    if nxt in walls:
        return None
    if nxt in boxes:
        beyond = (nxt[0] + di, nxt[1] + dj)
        if beyond in walls or beyond in boxes:
            return None
        return (boxes - {nxt}) | {beyond}, nxt
    return boxes, nxt


def live_squares(walls, goals, reachable):
    """Squares from which a lone box can still be pushed onto some goal (reverse 'pull' search)."""
    live, todo = set(goals), list(goals)
    while todo:
        b = todo.pop()
        for di, dj in MOVES.values():
            prev, stand = (b[0] + di, b[1] + dj), (b[0] + 2 * di, b[1] + 2 * dj)
            if prev in reachable and stand in reachable and prev not in live:
                live.add(prev)
                todo.append(prev)
    return live


def solve(state, max_states=MAX_STATES, max_depth=MAX_OPTIMAL):
    """Exact BFS. Returns (optimal_length, witness) or (None, reason)."""
    walls, goals = state["walls"], state["goals"]
    reach = state["reachable"] | {state["player"]}
    live = live_squares(walls, goals, reach | goals | state["boxes"])
    start = (state["boxes"], state["player"])
    if start[0] <= goals:
        return 0, ""
    parent = {start: None}
    frontier = deque([(start, 0)])
    while frontier:
        (boxes, player), depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for move in "UDLR":
            res = step(walls, boxes, player, move)
            if res is None:
                continue
            nb, np_ = res
            if nb != boxes and any(b not in live for b in nb):
                continue
            key = (nb, np_)
            if key in parent:
                continue
            parent[key] = ((boxes, player), move)
            if nb <= goals:
                moves = []
                while parent[key] is not None:
                    key, mv = parent[key]
                    moves.append(mv)
                return len(moves), "".join(reversed(moves))
            if len(parent) > max_states:
                return None, "state limit reached (unverifiable)"
            frontier.append((key, depth + 1))
    return None, f"unsolvable within {max_depth} moves"


def check(inst) -> dict:
    """Independent validity check: well-formed, enclosed, small, solvable, not already solved."""
    reasons = []
    try:
        st = parse(inst.get("level", ""))
    except ValueError as exc:
        return {"valid": False, "reasons": [str(exc)]}
    n_boxes, n_goals = len(st["boxes"]), len(st["goals"])
    if n_boxes == 0 or n_boxes != n_goals:
        reasons.append(f"boxes ({n_boxes}) must be >= 1 and equal goals ({n_goals})")
    if n_boxes > MAX_BOXES:
        reasons.append(f"more than {MAX_BOXES} boxes")
    if reasons:
        return {"valid": False, "reasons": reasons}
    optimal, witness = solve(st)
    if optimal is None:
        return {"valid": False, "reasons": [witness]}
    if optimal == 0:
        return {"valid": False, "reasons": ["already solved"]}
    return {"valid": True, "reasons": [], "optimal": optimal, "witness": witness, "boxes": n_boxes,
            "canonical": render(st["walls"], st["goals"], st["boxes"], st["player"], st["shape"])}


def extract(completion: str) -> str | None:
    line = str(completion).strip().split("\n")[0]
    moves = line.replace(" ", "").replace(",", "").upper()
    if not moves or set(moves) - set("UDLR") or len(moves) > MAX_ANSWER:
        return None
    return moves


def score(inst, completion) -> bool:
    moves = extract(completion)
    if moves is None:
        return False
    st = parse(inst["level"])
    boxes, player = st["boxes"], st["player"]
    for mv in moves:
        res = step(st["walls"], boxes, player, mv)
        if res is None:
            return False
        boxes, player = res
    return boxes <= st["goals"]


INTRO = ("Sokoban puzzles. Legend: # wall, @ player, $ box, . goal, * box already on a goal, + player standing on a goal. "
         "Moves: U = up, D = down, L = left, R = right. Walking into a box pushes it one square if the square behind it "
         "is free. Push every box onto a goal. Answer with the full move sequence on one line.")
# Few-shot puzzles; answers (None) are filled in with the solver's optimal witness.
SHOTS = [("#####\n#@$.#\n#####", "R"),
         ("######\n#    #\n# #$ #\n# .@ #\n######", None),
         ("#######\n#.    #\n# $## #\n#  @  #\n#######", None)]


def _shots():
    out = []
    for level, answer in SHOTS:
        res = check({"level": level})
        if not res["valid"]:
            raise RuntimeError("few-shot example invalid: " + level)
        out.append((res["canonical"], answer or res["witness"]))
    return out


FEW_SHOT = None


def student_prompt(inst) -> dict:
    global FEW_SHOT
    if FEW_SHOT is None:
        FEW_SHOT = _shots()
    parts = [INTRO, ""]
    for level, answer in FEW_SHOT:
        parts += ["Puzzle:", level, f"Solution: {answer}", ""]
    canonical = check(inst).get("canonical") or str(inst["level"]).strip("\n")
    parts += ["Puzzle:", canonical, "Solution:"]
    return {"prompt": "\n".join(parts), "stop": ["\n"], "max_tokens": 96}


# Operator template generator (calibration and the template arm). Levels widen the optimal-length band.
LEVELS = {1: dict(boxes=1, size=(5, 6), lo=1, hi=4), 2: dict(boxes=1, size=(6, 7), lo=5, hi=9),
          3: dict(boxes=1, size=(7, 7), lo=10, hi=18), 4: dict(boxes=2, size=(7, 7), lo=6, hi=14),
          5: dict(boxes=2, size=(8, 8), lo=15, hi=30)}


def template(rng: random.Random, level: int, tries: int = 4000) -> dict:
    cfg = LEVELS[level]
    h, w = cfg["size"]
    for _ in range(tries):
        cells = [(i, j) for i in range(1, h - 1) for j in range(1, w - 1)]
        walls = {(i, j) for i in range(h) for j in range(w) if i in (0, h - 1) or j in (0, w - 1)}
        walls |= {c for c in cells if rng.random() < 0.12}
        free = [c for c in cells if c not in walls]
        if len(free) < 2 * cfg["boxes"] + 2:
            continue
        picks = rng.sample(free, 2 * cfg["boxes"] + 1)
        goals, boxes, player = set(picks[:cfg["boxes"]]), set(picks[cfg["boxes"]:2 * cfg["boxes"]]), picks[-1]
        text = render(frozenset(walls), frozenset(goals), frozenset(boxes), player, (h, w))
        try:
            res = check({"level": text})
        except ValueError:
            continue
        if res["valid"] and cfg["lo"] <= res["optimal"] <= cfg["hi"]:
            return {"level": res["canonical"], "meta": {"optimal": res["optimal"], "boxes": res["boxes"], "level": level}}
    raise RuntimeError(f"no sokoban level found for difficulty {level}")
