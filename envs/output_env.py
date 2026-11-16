"""Code output prediction (DistillGap v6). The researcher writes a small Python function f and a call; the student
completes `assert f(...) == ` with the exact literal output (CRUXEval-style, no execution by the student).

Instance: {"code": "def f(...):\n    ...", "call": "f(<literal arguments>)"}
Validity is decided by the operator executing the code (in the study: inside the operator sandbox; the executor is
injectable), never by the author's claim: static whitelist (no imports, I/O, dunder access, eval/exec, randomness or
time), the call evaluates deterministically (two runs agree) within the time limit, and the result is a plain
literal whose repr round-trips through ast.literal_eval. The operator-computed repr is stored as expected_repr.
Student pass rule: the answer line parses with ast.literal_eval and its repr equals expected_repr.
"""
from __future__ import annotations

import ast
import json
import random
import subprocess
import sys

NAME = "output"
MAX_LINES, MAX_REPR, TIME_LIMIT = 40, 200, 2.0
BANNED_NAMES = {"open", "exec", "eval", "compile", "input", "print", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "breakpoint", "help", "exit", "quit", "memoryview", "id", "hash", "object", "type"}
RUNNER = r"""
import ast, json, sys
src, call = json.loads(sys.stdin.read())
ns = {"__builtins__": __builtins__}
exec(compile(src, "<task>", "exec"), ns)
value = eval(compile(call, "<call>", "eval"), ns)
text = repr(value)
ok = ast.literal_eval(text) == value
print(json.dumps({"repr": text, "literal": ok}))
"""


def static_errors(code: str, call: str) -> list:
    errors = []
    if len(code.splitlines()) > MAX_LINES:
        errors.append(f"code longer than {MAX_LINES} lines")
    try:
        tree = ast.parse(code)
        call_tree = ast.parse(call, mode="eval")
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg}"]
    if not any(isinstance(n, ast.FunctionDef) and n.name == "f" for n in tree.body):
        errors.append("code must define f")
    if any(not isinstance(n, (ast.FunctionDef, ast.Assign)) for n in tree.body):
        errors.append("top level may only hold function definitions and constant assignments")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda, ast.AsyncFunctionDef,
                             ast.Await, ast.Yield, ast.YieldFrom, ast.ClassDef, ast.With, ast.Try)):
            errors.append(f"disallowed construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            errors.append(f"disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errors.append(f"disallowed attribute: {node.attr}")
    body = call_tree.body
    if not (isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and body.func.id == "f" and not body.keywords):
        errors.append("call must be f(<literal arguments>)")
    else:
        for arg in body.args:
            try:
                ast.literal_eval(arg)
            except ValueError:
                errors.append("call arguments must be literals")
    return sorted(set(errors))


def subprocess_executor(code: str, call: str) -> dict:
    """Default executor (templates, tests). The study passes the operator sandbox executor instead."""
    try:
        res = subprocess.run([sys.executable, "-I", "-S", "-c", RUNNER], input=json.dumps([code, call]),
                             capture_output=True, text=True, timeout=TIME_LIMIT)
    except subprocess.TimeoutExpired:
        return {"outcome": "timeout"}
    if res.returncode != 0:
        return {"outcome": "error", "detail": res.stderr.strip().splitlines()[-1:] or [""]}
    return {"outcome": "ok", **json.loads(res.stdout)}


def check(inst, executor=subprocess_executor) -> dict:
    code, call = inst.get("code", ""), inst.get("call", "")
    errors = static_errors(code, call)
    if errors:
        return {"valid": False, "reasons": errors}
    runs = [executor(code, call) for _ in range(2)]
    if any(r["outcome"] != "ok" for r in runs):
        return {"valid": False, "reasons": [f"execution {runs[0]['outcome']}: {runs[0].get('detail', '')}"]}
    if runs[0]["repr"] != runs[1]["repr"]:
        return {"valid": False, "reasons": ["nondeterministic output"]}
    if not runs[0]["literal"]:
        return {"valid": False, "reasons": ["output is not a plain literal"]}
    if len(runs[0]["repr"]) > MAX_REPR:
        return {"valid": False, "reasons": [f"output repr longer than {MAX_REPR} characters"]}
    return {"valid": True, "reasons": [], "expected_repr": runs[0]["repr"]}


def extract(completion):
    line = str(completion).strip().split("\n")[0].strip()
    if line.endswith("[/PYTHON]"):
        line = line[: -len("[/PYTHON]")].strip()
    try:
        return repr(ast.literal_eval(line))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def score(inst, completion) -> bool:
    got = extract(completion)
    return got is not None and got == inst["expected_repr"]


INTRO = ("Each block gives a Python function f and an assertion with an input to f. Complete the assertion with the "
         "exact literal value that f returns for that input.")
SHOTS = [("def f(s):\n    return s[::-1].upper()", 'f("abc")'),
         ("def f(xs):\n    out = []\n    for v in xs:\n        if v % 2 == 0:\n            out.append(v * 3)\n    return out",
          "f([1, 2, 3, 4])"),
         ("def f(text, k):\n    words = text.split()\n    return len([w for w in words if len(w) > k])",
          'f("the quick brown fox jumps", 4)')]


def student_prompt(inst) -> dict:
    parts = [INTRO, ""]
    for code, call in SHOTS:
        res = check({"code": code, "call": call})
        if not res["valid"]:
            raise RuntimeError(f"few-shot example invalid: {res['reasons']}")
        parts += ["[PYTHON]", code, f"assert {call} == {res['expected_repr']}", "[/PYTHON]", ""]
    parts += ["[PYTHON]", inst["code"], f"assert {inst['call']} =="]
    return {"prompt": "\n".join(parts), "stop": ["\n", "[/PYTHON]"], "max_tokens": 64}


# ---- Operator template generator: f chains `level` primitive steps over a string or an integer list. ----
LETTERS = "abcdefghijklmnop"


def _str_ops(rng):
    a, b = rng.sample(LETTERS[:8], 2)
    i = rng.randint(1, 3)
    k = rng.randint(2, 3)
    return [f"x = x[::-1]", f"x = x.upper()", f"x = x.swapcase()", f"x = x.replace({a!r}, {b!r})",
            f"x = x[{i}:] + x[:{i}]", f"x = ''.join(c for c in x if c not in {a + b!r})", "x = ''.join(sorted(x))",
            f"x = x.strip({a!r})", f"x = {b!r}.join(x.split({a!r}))", f"x = ''.join(x[j] for j in range(0, len(x), {k}))",
            f"x = ''.join(c * 2 if c in {a + b!r} else c for c in x)", f"x = x[:{i + 3}]"]


def _list_ops(rng):
    k = rng.randint(2, 4)
    i = rng.randint(1, 3)
    return [f"x = [v * {k} for v in x]", f"x = [v for v in x if v % {k} != 0]", "x = sorted(x)", "x = x[::-1]",
            f"x = x[{i}:] + x[:{i}]", "x = [x[j] + x[j + 1] for j in range(len(x) - 1)]", "x = list(dict.fromkeys(x))",
            "x = [v - min(x) for v in x] if x else x", f"x = [v + {k} if v % 2 == 0 else v - 1 for v in x]",
            "acc = 0\n    out = []\n    for v in x:\n        acc += v\n        out.append(acc)\n    x = out",
            f"x = [v // {k} for v in x]", "x = [v for j, v in enumerate(x) if j % 2 == 0]"]


LIST_RETURNS = ["return x", "return sum(x)", "return len(x)", "return max(x) if x else -1", "return x[-1] if x else None"]
LEVELS = {lv: {"steps": lv} for lv in range(1, 6)}


def template(rng: random.Random, level: int, tries: int = 200, executor=subprocess_executor) -> dict:
    for _ in range(tries):
        if rng.random() < 0.5:
            ops = [rng.choice(_str_ops(rng)) for _ in range(level)]
            arg = repr("".join(rng.choice(LETTERS[:8]) for _ in range(rng.randint(5, 6 + 2 * level))))
            ret = "return x"
        else:
            ops = [rng.choice(_list_ops(rng)) for _ in range(level)]
            arg = repr([rng.randint(-5 if level > 2 else 0, 9 + 3 * level) for _ in range(rng.randint(4, 5 + level))])
            ret = rng.choice(LIST_RETURNS) if level > 1 else "return x"
        code = "def f(x):\n" + "".join(f"    {op}\n" for op in ops) + f"    {ret}"
        inst = {"code": code, "call": f"f({arg})"}
        res = check(inst, executor)
        if res["valid"] and res["expected_repr"] not in ("''", "[]", "None"):
            return {**inst, "expected_repr": res["expected_repr"], "meta": {"steps": level, "level": level}}
    raise RuntimeError("no output-prediction instance")
