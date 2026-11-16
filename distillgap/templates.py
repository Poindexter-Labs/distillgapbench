"""Operator-authored parametric task families (no teacher involvement).

Uses: (1) public development calibration of the target profile (CALIBRATION families),
(2) the secondary template/random control arm (CONTROL families). The two sets are
disjoint by family. Every task flows through exactly the same verifier, target and final
evaluation as teacher tasks. Parameters and inputs come from an operator seed only.
"""
from __future__ import annotations

import random


def _ints(rng, n, lo, hi):
    return [rng.randint(lo, hi) for _ in range(n)]


def _word(rng, lo=1, hi=8, alphabet="abcdefghij"):
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(lo, hi)))


# ---------------------------------------------------------------- calibration families
def fam_rotate(rng):
    direction = rng.choice(["left", "right"])
    sign = "" if direction == "left" else "-"
    code = f"""def rotate_items(xs, k):
    if not xs:
        return []
    k = ({sign}k) % len(xs)
    return xs[k:] + xs[:k]
"""
    desc = (f"Rotate the list xs to the {direction} by k positions (k >= 0 may exceed len(xs)) and return "
            f"the new list. An empty list returns an empty list.")
    gen = lambda r: [_ints(r, r.randint(0, 8), -9, 9), r.randint(0, 20)]
    return "rotate_items", "def rotate_items(xs: List[int], k: int) -> List[int]:", desc, code, gen


def fam_rle(rng):
    min_run = rng.choice([1, 2])
    code = f"""def compress_runs(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        n = j - i
        out.append(s[i] + str(n) if n >= {min_run} else s[i] * n)
        i = j
    return "".join(out)
"""
    desc = (f"Compress the string s by replacing every maximal run of identical characters with the "
            f"character followed by the run length, but only for runs of length at least {min_run}; "
            f"shorter runs are copied unchanged. Return the compressed string.")
    gen = lambda r: ["".join(r.choice("aab") * r.randint(1, 4) for _ in range(r.randint(0, 6)))]
    return "compress_runs", "def compress_runs(s: str) -> str:", desc, code, gen


def fam_digit_base(rng):
    base = rng.choice([2, 3, 5, 7, 9])
    code = f"""def digit_sum_base(n):
    total = 0
    while n:
        total += n % {base}
        n //= {base}
    return total
"""
    desc = f"Return the sum of the digits of the non-negative integer n when written in base {base}."
    gen = lambda r: [r.choice([0, r.randint(1, 50), r.randint(50, 10 ** 6)])]
    return "digit_sum_base", "def digit_sum_base(n: int) -> int:", desc, code, gen


def fam_brackets(rng):
    pairs = rng.choice(["()[]", "(){}", "()<>", "[]{}"])
    code = f"""def is_balanced(s):
    pairs = {{{pairs[1]!r}: {pairs[0]!r}, {pairs[3]!r}: {pairs[2]!r}}}
    stack = []
    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack
"""
    desc = (f"Return True if the brackets {pairs[0]}{pairs[1]} and {pairs[2]}{pairs[3]} in s are balanced and "
            f"properly nested; all other characters are ignored. The empty string is balanced.")
    def gen(r):
        if r.random() < 0.5:
            out = ""
            for _ in range(r.randint(0, 4)):
                o = r.choice([0, 2])
                out = r.choice([pairs[o] + out + pairs[o + 1], out + pairs[o] + pairs[o + 1]])
            return [out]
        return ["".join(r.choice(pairs + "ab") for _ in range(r.randint(0, 10)))]
    return "is_balanced", "def is_balanced(s: str) -> bool:", desc, code, gen


def fam_kth_distinct(rng):
    which = rng.choice(["smallest", "largest"])
    order = "sorted(set(xs))" if which == "smallest" else "sorted(set(xs), reverse=True)"
    code = f"""def kth_distinct(xs, k):
    values = {order}
    return values[k - 1] if 1 <= k <= len(values) else -1
"""
    desc = (f"Return the k-th {which} DISTINCT value in xs (k is 1-based). If there are fewer than k distinct "
            f"values, or k < 1, return -1.")
    gen = lambda r: [_ints(r, r.randint(0, 9), -5, 5), r.randint(0, 6)]
    return "kth_distinct", "def kth_distinct(xs: List[int], k: int) -> int:", desc, code, gen


def fam_merge_intervals(rng):
    touching = rng.choice([True, False])
    cmp = "<=" if touching else "<"
    code = f"""def merge_spans(spans):
    spans = sorted(spans)
    out = []
    for a, b in spans:
        if out and a {cmp} out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out
"""
    rule = "Spans that merely touch (end of one equals start of the next) are merged." if touching else \
        "Spans that merely touch (end of one equals start of the next) are NOT merged."
    desc = (f"spans is a list of [start, end] pairs with start <= end. Merge overlapping spans and return the "
            f"merged spans sorted by start. {rule}")
    def gen(r):
        out = []
        for _ in range(r.randint(0, 6)):
            a = r.randint(0, 15)
            out.append([a, a + r.randint(0, 4)])
        return [out]
    return "merge_spans", "def merge_spans(spans: List[List[int]]) -> List[List[int]]:", desc, code, gen


def fam_caesar(rng):
    shift = rng.randint(1, 25)
    code = f"""def shift_letters(s):
    out = []
    for ch in s:
        if 'a' <= ch <= 'z':
            out.append(chr((ord(ch) - 97 + {shift}) % 26 + 97))
        elif 'A' <= ch <= 'Z':
            out.append(chr((ord(ch) - 65 + {shift}) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)
"""
    desc = (f"Shift every ASCII letter in s forward by {shift} positions in the alphabet, wrapping around and "
            f"preserving case. All other characters are unchanged.")
    gen = lambda r: ["".join(r.choice("abxyzABZ -!9") for _ in range(r.randint(0, 12)))]
    return "shift_letters", "def shift_letters(s: str) -> str:", desc, code, gen


def fam_longest_run(rng):
    rel = rng.choice([("strictly increasing", "<"), ("non-decreasing", "<="), ("strictly decreasing", ">")])
    code = f"""def longest_run(xs):
    if not xs:
        return 0
    best = cur = 1
    for a, b in zip(xs, xs[1:]):
        cur = cur + 1 if a {rel[1]} b else 1
        best = max(best, cur)
    return best
"""
    desc = f"Return the length of the longest contiguous {rel[0]} run in xs (0 for an empty list)."
    gen = lambda r: [_ints(r, r.randint(0, 10), 0, 5)]
    return "longest_run", "def longest_run(xs: List[int]) -> int:", desc, code, gen


# ---------------------------------------------------------------- control families
def fam_topk_words(rng):
    k = rng.randint(1, 3)
    code = f"""def top_words(text):
    counts = {{}}
    for w in text.split():
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:{k}]]
"""
    desc = (f"Split text on whitespace into words and return the {k} most frequent words. Order by decreasing "
            f"count, breaking ties alphabetically. Return fewer if there are fewer distinct words.")
    gen = lambda r: [" ".join(r.choice(["ab", "ba", "c", "dd", "e"]) for _ in range(r.randint(0, 10)))]
    return "top_words", "def top_words(text: str) -> List[str]:", desc, code, gen


def fam_recurrence(rng):
    a, b, m = rng.randint(1, 3), rng.randint(0, 3), rng.choice([7, 10, 13, 97])
    code = f"""def seq_term(n, x0, x1):
    if n == 0:
        return x0 % {m}
    p, q = x0 % {m}, x1 % {m}
    for _ in range(n - 1):
        p, q = q, ({a} * q + {b} * p) % {m}
    return q
"""
    desc = (f"A sequence has x(0) = x0 and x(1) = x1, and x(i) = ({a} * x(i-1) + {b} * x(i-2)) for i >= 2. "
            f"Return x(n) modulo {m} (as a value in 0..{m - 1}). n >= 0.")
    gen = lambda r: [r.randint(0, 30), r.randint(0, 20), r.randint(0, 20)]
    return "seq_term", "def seq_term(n: int, x0: int, x1: int) -> int:", desc, code, gen


def fam_spiral(rng):
    clockwise = rng.choice([True, False])
    code_cw = """def spiral(grid):
    out = []
    g = [row[:] for row in grid]
    while g and g[0]:
        out.extend(g.pop(0))
        g = [list(r) for r in zip(*g)][::-1] if g and g[0] else []
    return out
"""
    if clockwise:
        desc = ("grid is a rectangular list of rows. Return its elements in clockwise spiral order starting at the "
                "top-left corner and moving right first. An empty grid returns [].")
        code = code_cw
    else:
        desc = ("grid is a rectangular list of rows. Return its elements in counter-clockwise spiral order starting "
                "at the top-left corner and moving DOWN first. An empty grid returns [].")
        code = """def spiral(grid):
    if not grid or not grid[0]:
        return []
    out = []
    top, bottom, left, right = 0, len(grid) - 1, 0, len(grid[0]) - 1
    while top <= bottom and left <= right:
        for i in range(top, bottom + 1):
            out.append(grid[i][left])
        left += 1
        if left > right:
            break
        for j in range(left, right + 1):
            out.append(grid[bottom][j])
        bottom -= 1
        if top > bottom:
            break
        for i in range(bottom, top - 1, -1):
            out.append(grid[i][right])
        right -= 1
        if left > right:
            break
        for j in range(right, left - 1, -1):
            out.append(grid[top][j])
        top += 1
    return out
"""
    def gen(r):
        rows, cols = r.randint(0, 4), r.randint(1, 4)
        return [[_ints(r, cols, 0, 9) for _ in range(rows)]]
    return "spiral", "def spiral(grid: List[List[int]]) -> List[int]:", desc, code, gen


def fam_islands(rng):
    diag = rng.choice([True, False])
    steps = "[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]" if diag else "[(-1,0),(1,0),(0,-1),(0,1)]"
    code = f"""def count_islands(grid):
    seen = set()
    n = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1 and (i, j) not in seen:
                n += 1
                stack = [(i, j)]
                seen.add((i, j))
                while stack:
                    a, b = stack.pop()
                    for da, db in {steps}:
                        x, y = a + da, b + db
                        if 0 <= x < len(grid) and 0 <= y < len(grid[x]) and grid[x][y] == 1 and (x, y) not in seen:
                            seen.add((x, y))
                            stack.append((x, y))
    return n
"""
    conn = "8-directionally (including diagonals)" if diag else "4-directionally (up, down, left, right only)"
    desc = f"grid is a rectangular 0/1 matrix. Count the islands: maximal groups of 1-cells connected {conn}."
    def gen(r):
        rows, cols = r.randint(0, 5), r.randint(1, 5)
        return [[[r.choice([0, 0, 1]) for _ in range(cols)] for _ in range(rows)]]
    return "count_islands", "def count_islands(grid: List[List[int]]) -> int:", desc, code, gen


def fam_grid_paths(rng):
    code = """def count_paths(grid):
    if not grid or not grid[0] or grid[0][0] == 1:
        return 0
    rows, cols = len(grid), len(grid[0])
    ways = [[0] * cols for _ in range(rows)]
    ways[0][0] = 1
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                ways[i][j] = 0
            elif i or j:
                ways[i][j] = (ways[i - 1][j] if i else 0) + (ways[i][j - 1] if j else 0)
    return ways[-1][-1]
"""
    desc = ("grid is a rectangular 0/1 matrix where 1 marks a blocked cell. Count the paths from the top-left to the "
            "bottom-right cell that move only right or down through unblocked cells. Return 0 if either corner is "
            "blocked or the grid is empty.")
    def gen(r):
        rows, cols = r.randint(1, 5), r.randint(1, 5)
        return [[[1 if r.random() < 0.2 else 0 for _ in range(cols)] for _ in range(rows)]]
    return "count_paths", "def count_paths(grid: List[List[int]]) -> int:", desc, code, gen


def fam_base_string(rng):
    base = rng.choice([3, 5, 7, 12, 16])
    code = f"""def to_base(n):
    digits = "0123456789abcdef"
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    out = []
    while n:
        out.append(digits[n % {base}])
        n //= {base}
    return sign + "".join(reversed(out))
"""
    desc = (f"Return the base-{base} representation of the integer n as a string using digits 0-9 then lowercase "
            f"letters, with a leading '-' for negative numbers and no leading zeros ('0' for zero).")
    gen = lambda r: [r.choice([0, r.randint(-300, 300), r.randint(-10 ** 6, 10 ** 6)])]
    return "to_base", "def to_base(n: int) -> str:", desc, code, gen


def fam_collatz(rng):
    mult, add = rng.choice([3, 5]), rng.choice([1, 3])
    code = f"""def steps_to_one(n):
    steps = 0
    while n != 1 and steps < 1000:
        n = n // 2 if n % 2 == 0 else {mult} * n + {add}
        steps += 1
    return steps if n == 1 else -1
"""
    desc = (f"Starting from the positive integer n, repeatedly replace n by n // 2 if n is even, otherwise by "
            f"{mult} * n + {add}. Return the number of replacements needed to reach 1, or -1 if 1 is not reached "
            f"within 1000 replacements.")
    gen = lambda r: [r.randint(1, 60)]
    return "steps_to_one", "def steps_to_one(n: int) -> int:", desc, code, gen


def fam_word_wrap(rng):
    width = rng.randint(6, 14)
    code = f"""def wrap_words(text):
    lines, cur = [], ""
    for w in text.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= {width}:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
"""
    desc = (f"Greedily wrap the whitespace-separated words of text into lines of at most {width} characters, words "
            f"separated by single spaces. A word longer than {width} goes on its own line unbroken. Return the "
            f"list of lines (empty list for no words).")
    gen = lambda r: [" ".join(_word(r, 1, 9) for _ in range(r.randint(0, 8)))]
    return "wrap_words", "def wrap_words(text: str) -> List[str]:", desc, code, gen


CALIBRATION = [fam_rotate, fam_rle, fam_digit_base, fam_brackets, fam_kth_distinct, fam_merge_intervals,
               fam_caesar, fam_longest_run]
CONTROL = [fam_topk_words, fam_recurrence, fam_spiral, fam_islands, fam_grid_paths, fam_base_string,
           fam_collatz, fam_word_wrap]


def make_task(family_fn, seed: int, n_hidden: int = 12, n_public: int = 2) -> dict:
    rng = random.Random(seed)
    entry, signature, description, code, gen = family_fn(rng)
    seen, inputs = set(), []
    for _ in range(400):
        args = gen(rng)
        key = repr(args)
        if key not in seen:
            seen.add(key)
            inputs.append(args)
        if len(inputs) >= n_hidden + n_public:
            break
    return {"family": "template:" + family_fn.__name__[4:], "entry_point": entry, "signature": signature,
            "description": description, "public_inputs": inputs[:n_public], "hidden_inputs": inputs[n_public:],
            "reference_solution": code, "derived_from": [], "rationale": "operator template",
            "template_seed": seed}


def _ast_nodes(code):
    import ast
    return sum(1 for _ in ast.walk(ast.parse(code)))


def generate(families, count: int, seed: int, max_ast_nodes: int | None = None) -> list:
    """Operator template draws. Under a complexity budget, over-budget variants are redrawn
    (up to 50 seeds) and families with no in-budget variant are dropped (documented)."""
    rng = random.Random(seed)
    if max_ast_nodes is not None:
        families = [fn for fn in families
                    if any(_ast_nodes(make_task(fn, s)["reference_solution"]) <= max_ast_nodes for s in range(50))]
    order = [families[i % len(families)] for i in range(count)]
    tasks = []
    for fn in order:
        for _ in range(50):
            task = make_task(fn, rng.randrange(2 ** 31))
            if max_ast_nodes is None or _ast_nodes(task["reference_solution"]) <= max_ast_nodes:
                break
        tasks.append(task)
    return tasks
