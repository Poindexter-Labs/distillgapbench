"""Code-task schema, static validation and the frozen target-visible prompt rendering.

A candidate's hidden material (reference solution, hidden inputs, rationale,
claimed outputs) never enters `render_prompt`; tests assert this isolation.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re

SCHEMA_VERSION = "distillgap_code_task_v1"
PROMPT_TEMPLATE_VERSION = "humaneval_docstring_v1"

LIMITS = {
    "description_chars": 2000,
    "family_chars": 80,
    "public_inputs": (1, 3),
    "hidden_inputs": (6, 20),
    "input_json_bytes": 4096,
    "reference_chars": 8000,
    "rationale_chars": 2000,
    "signature_chars": 300,
}

PROMPT_HEADER = (
    "from typing import *\n"
    "import bisect\n"
    "import collections\n"
    "import functools\n"
    "import heapq\n"
    "import itertools\n"
    "import math\n"
    "import re\n"
    "import string\n"
    "\n\n"
)

ALLOWED_PARAM_NAMES = {"int", "float", "str", "bool", "None", "List", "Dict", "Optional",
                       "list", "dict", "Any", "Union"}
FORBIDDEN_IMPORTS = {"random", "time", "datetime", "os", "sys", "subprocess", "socket",
                     "threading", "multiprocessing", "secrets", "uuid", "shutil", "pathlib",
                     "io", "signal", "ctypes", "importlib", "builtins", "inspect", "gc",
                     "urllib", "http", "requests", "pickle", "tempfile", "asyncio", "resource"}
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,59}")


@dataclass
class Task:
    family: str
    entry_point: str
    signature: str
    description: str
    public_inputs: list
    hidden_inputs: list
    reference_solution: str
    claimed_public_outputs: list | None = None
    derived_from: list = field(default_factory=list)
    rationale: str = ""
    params: list = field(default_factory=list)

    def to_json(self):
        return asdict(self)


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _annotation_names(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} | {
        "None" for n in ast.walk(node) if isinstance(n, ast.Constant) and n.value is None}


def parse_signature(signature: str, entry_point: str):
    """Return (param_names, errors). Signature must be exactly one annotated def line."""
    errors = []
    if not isinstance(signature, str) or not signature.strip().startswith("def "):
        return [], ["signature_not_def"]
    if "\n" in signature.strip() or len(signature) > LIMITS["signature_chars"]:
        return [], ["signature_not_single_line"]
    try:
        module = ast.parse(signature.strip().rstrip(":") + ":\n    pass\n")
    except SyntaxError:
        return [], ["signature_syntax_error"]
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return [], ["signature_not_single_function"]
    fn = module.body[0]
    if fn.name != entry_point:
        errors.append("signature_name_mismatch")
    a = fn.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults or a.kw_defaults:
        errors.append("signature_has_defaults_or_varargs")
    if fn.decorator_list:
        errors.append("signature_has_decorators")
    if fn.returns is None:
        errors.append("signature_missing_return_annotation")
    params = [arg.arg for arg in a.args]
    if not params:
        errors.append("signature_has_no_parameters")
    for arg in a.args:
        if arg.annotation is None:
            errors.append("parameter_missing_annotation")
            continue
        bad = _annotation_names(arg.annotation) - ALLOWED_PARAM_NAMES
        if bad:
            errors.append("parameter_type_not_allowed:" + ",".join(sorted(bad)))
    if len(set(params)) != len(params):
        errors.append("duplicate_parameter")
    return params, errors


def _json_input_ok(args, n_params):
    if not isinstance(args, list) or len(args) != n_params:
        return "input_arity_mismatch"
    try:
        text = json.dumps(args, allow_nan=False)
    except (TypeError, ValueError):
        return "input_not_json"
    if len(text.encode()) > LIMITS["input_json_bytes"]:
        return "input_too_large"
    return None


def reference_static_errors(code: str, entry_point: str, params: list) -> list:
    errors = []
    if not isinstance(code, str) or not code.strip():
        return ["reference_missing"]
    if len(code) > LIMITS["reference_chars"]:
        errors.append("reference_too_long")
    try:
        module = ast.parse(code)
    except SyntaxError:
        return errors + ["reference_syntax_error"]
    top = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == entry_point]
    if not top:
        errors.append("reference_missing_entry_point")
    elif [a.arg for a in top[-1].args.args] != params:
        errors.append("reference_parameters_mismatch")
    for node in ast.walk(module):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        if set(names) & FORBIDDEN_IMPORTS:
            errors.append("reference_forbidden_import:" + ",".join(sorted(set(names) & FORBIDDEN_IMPORTS)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
                "open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}:
            errors.append("reference_forbidden_call:" + node.func.id)
    return sorted(set(errors))


def parse_task(raw) -> tuple[Task | None, list]:
    """Static validation of one teacher-supplied task object. Returns (task or None, errors)."""
    if not isinstance(raw, dict):
        return None, ["task_not_object"]
    errors = []
    get = lambda k, t, d=None: raw.get(k, d) if isinstance(raw.get(k, d), t) else None
    family = get("family", str) or ""
    entry_point = get("entry_point", str) or ""
    signature = get("signature", str) or ""
    description = get("description", str) or ""
    reference = get("reference_solution", str) or ""
    public_inputs = raw.get("public_inputs")
    hidden_inputs = raw.get("hidden_inputs")
    rationale = get("rationale", str) or ""
    derived = raw.get("derived_from") if isinstance(raw.get("derived_from"), list) else []
    claimed = raw.get("claimed_public_outputs") if isinstance(raw.get("claimed_public_outputs"), list) else None
    if not IDENTIFIER.fullmatch(entry_point):
        errors.append("entry_point_invalid")
    if not family.strip() or len(family) > LIMITS["family_chars"]:
        errors.append("family_invalid")
    if not description.strip():
        errors.append("description_missing")
    if len(description) > LIMITS["description_chars"]:
        errors.append("description_too_long")
    if '"""' in description or "'''" in description:
        errors.append("description_contains_triple_quotes")
    if len(rationale) > LIMITS["rationale_chars"]:
        rationale = rationale[: LIMITS["rationale_chars"]]
    params, sig_errors = parse_signature(signature, entry_point)
    errors += sig_errors
    lo, hi = LIMITS["public_inputs"]
    if not isinstance(public_inputs, list) or not lo <= len(public_inputs) <= hi:
        errors.append("public_inputs_count")
        public_inputs = public_inputs if isinstance(public_inputs, list) else []
    lo, hi = LIMITS["hidden_inputs"]
    if not isinstance(hidden_inputs, list) or not lo <= len(hidden_inputs) <= hi:
        errors.append("hidden_inputs_count")
        hidden_inputs = hidden_inputs if isinstance(hidden_inputs, list) else []
    if params:
        for args in list(public_inputs) + list(hidden_inputs):
            problem = _json_input_ok(args, len(params))
            if problem:
                errors.append(problem)
                break
    errors += reference_static_errors(reference, entry_point, params) if params else []
    if claimed is not None and len(claimed) != len(public_inputs):
        claimed = None  # an unusable claim is ignored, not fatal
    task = Task(family.strip()[: LIMITS["family_chars"]], entry_point, signature.strip(),
                description.strip(), public_inputs, hidden_inputs, reference, claimed,
                [str(x)[:40] for x in derived][:8], rationale, params)
    return task, sorted(set(errors))


def render_prompt(task: Task, example_reprs: list) -> str:
    """The exact text the target, the validator and the blind teacher see.

    example_reprs: [(args_list, output_repr_string), ...] computed by executing the
    reference on public inputs; the teacher never writes displayed outputs directly.
    """
    signature = task.signature.strip()
    if not signature.endswith(":"):
        signature += ":"
    body = []
    for line in task.description.strip().splitlines():
        body.append(("    " + line).rstrip())
    if example_reprs:
        body.append("")
        for args, shown in example_reprs:
            call = task.entry_point + "(" + ", ".join(repr(a) for a in args) + ")"
            body.append("    >>> " + call)
            for out_line in str(shown).splitlines() or [""]:
                body.append("    " + out_line)
    doc = "\n".join(body)
    prefix = 'r"""' if "\\" in doc else '"""'
    return PROMPT_HEADER + signature + "\n    " + prefix + doc.lstrip() + "\n    \"\"\"\n"


def program_from_completion(prompt: str, completion: str) -> tuple[str, str]:
    """Truncate a raw continuation at the first non-indented line; return (program, status).

    status: ok | empty_completion | syntax_error
    """
    lines = completion.split("\n")
    kept = []
    for index, line in enumerate(lines):
        if index > 0 and line.strip() and not line[0].isspace():
            break
        if line.strip().startswith("```"):
            break
        kept.append(line)
    body = "\n".join(kept).rstrip()
    if not body.strip():
        return prompt, "empty_completion"
    program = prompt + body + "\n"
    try:
        ast.parse(program)
    except SyntaxError:
        return program, "syntax_error"
    return program, "ok"
