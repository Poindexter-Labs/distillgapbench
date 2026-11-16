"""Canonicalization and deduplication of code tasks (frozen thresholds).

Layers, applied in generation order (first occurrence wins, later ones are
retained and marked duplicate_of):
  1. exact: normalized rendered-prompt hash
  2. ast:   alpha-renamed, docstring-stripped reference AST hash
  3. text:  Jaccard of word 3-shingles over identifier-masked descriptions >= TEXT_THRESHOLD
  4. behavior: same arity; each reference run on the other's inputs; all outputs equal on
     >= BEHAVIOR_MIN_INPUTS inputs that both execute.
Limits (documented): semantic near-duplicates with different I/O types, or tasks that
are copies of public benchmark problems, are not detected by these automated checks.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import re

from .sandbox import run_function, values_equal

DEDUP_VERSION = "distillgap_dedup_v1"
TEXT_THRESHOLD = 0.6
CLUSTER_THRESHOLD = 0.3
BEHAVIOR_MIN_INPUTS = 5
BUILTIN_NAMES = frozenset(dir(builtins))


def normalized_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", prompt.lower()).strip().encode()).hexdigest()


class _Renamer(ast.NodeTransformer):
    def __init__(self):
        self.names = {}

    def _canon(self, name):
        if name not in self.names:
            self.names[name] = f"v{len(self.names)}"
        return self.names[name]

    def visit_FunctionDef(self, node):
        node.name = self._canon(node.name)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        node.arg = self._canon(node.arg)
        node.annotation = None
        return node

    def visit_Name(self, node):
        if node.id in BUILTIN_NAMES:
            return node
        node.id = self._canon(node.id)
        return node


def ast_hash(code: str) -> str | None:
    try:
        tree = _Renamer().visit(ast.parse(code))
    except (SyntaxError, RecursionError):
        return None
    return hashlib.sha256(ast.dump(tree, annotate_fields=False).encode()).hexdigest()


def shingles(text: str, entry_point: str = "", params=()) -> set:
    masked = text.lower()
    for name in [entry_point, *params]:
        if name:
            masked = re.sub(r"\b" + re.escape(name.lower()) + r"\b", " ident ", masked)
    words = re.findall(r"[a-z0-9]+", masked)
    return {" ".join(words[i:i + 3]) for i in range(max(0, len(words) - 2))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def behavior_equivalent(task_a, task_b) -> bool:
    if len(task_a.params) != len(task_b.params):
        return False
    inputs = (task_a.hidden_inputs + task_b.hidden_inputs)[:40]
    ra = run_function(task_a.reference_solution, task_a.entry_point, inputs, per_call_seconds=2.0)
    rb = run_function(task_b.reference_solution, task_b.entry_point, inputs, per_call_seconds=2.0)
    if ra.status != "ok" or rb.status != "ok":
        return False
    compared = 0
    for ca, cb in zip(ra.calls, rb.calls):
        if ca and cb and ca.get("ok") and cb.get("ok"):
            if not values_equal(ca["value"], cb["value"]):
                return False
            compared += 1
    return compared >= BEHAVIOR_MIN_INPUTS


class Deduper:
    """Stateful within-scope deduplicator. `check` returns (duplicate_of_id | None, layer | None)."""

    def __init__(self):
        self.seen = []  # (candidate_id, task, prompt_hash, ast_hash, shingles)

    def check(self, candidate_id, task, prompt):
        p_hash = normalized_prompt_hash(prompt)
        a_hash = ast_hash(task.reference_solution)
        sh = shingles(task.description, task.entry_point, task.params)
        result = (None, None)
        for other_id, other, op, oa, osh in self.seen:
            if p_hash == op:
                result = (other_id, "exact"); break
            if a_hash and a_hash == oa:
                result = (other_id, "ast"); break
            if jaccard(sh, osh) >= TEXT_THRESHOLD:
                result = (other_id, "text"); break
        if result[0] is None:
            for other_id, other, _, _, _ in self.seen:
                if behavior_equivalent(task, other):
                    result = (other_id, "behavior"); break
        if result[0] is None:
            self.seen.append((candidate_id, task, p_hash, a_hash, sh))
        return result


def family_clusters(items) -> int:
    """Single-linkage clusters over description shingles (Jaccard >= CLUSTER_THRESHOLD)."""
    items = list(items)
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if jaccard(items[i], items[j]) >= CLUSTER_THRESHOLD:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(items))})
