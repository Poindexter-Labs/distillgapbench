"""Frozen prompts for the teacher (challenge author), the independent validator and the
blind teacher re-solve, plus tolerant JSON extraction. Changing any text here changes
PROMPTS_SHA256 and therefore the protocol identity.
"""
from __future__ import annotations

import hashlib
import json
import re

from .schema import PROMPT_HEADER

PROMPTS_VERSION = "distillgap_prompts_v2_complexity_budget"

TARGET_DESCRIPTION = (
    "Qwen/Qwen3-4B-Base (revision 906bfd4b), a 4-billion-parameter PRETRAINED BASE language "
    "model (not instruction-tuned), with frozen weights. It receives only the rendered prompt "
    "below and continues the text to write the function body. It draws 4 independent samples "
    "(temperature 0.6, top_p 0.95, at most 512 new tokens); the body ends at the first "
    "non-indented line."
)

TEACHER_SYSTEM = f"""You are the challenge author in DistillGapBench, a research benchmark that measures how well a strong model can find genuine capability gaps in a specific smaller model.

Your job: write ORIGINAL, self-contained Python programming tasks that are unambiguous and solvable by a competent programmer (and by you), but that the target model below fails.

TARGET MODEL: {TARGET_DESCRIPTION}

EXACT PROMPT FORMAT THE TARGET SEES (operator-rendered; nothing else is shown):
{PROMPT_HEADER}def ENTRY_POINT(PARAMS) -> RETURN_TYPE:
    \"\"\"DESCRIPTION

    >>> ENTRY_POINT(<public example arguments>)
    <output of YOUR reference solution on those arguments>
    \"\"\"

The operator computes every expected output by executing your reference solution in a sandbox. You never write expected outputs for hidden tests. The target's function is executed on your hidden inputs and must match your reference exactly (floats within 1e-6 relative; tuples compare equal to lists; bool is distinct from int).

HOW YOUR TASKS ARE SCORED (by an independent operator after discovery ends; you will not see this evaluation):
A task counts as a verified capability gap only if ALL of the following hold.
 1. Your reference runs deterministically on every input within the limits.
 2. An independent strong model that sees ONLY the rendered prompt writes its own solution and proposes extra edge-case inputs; its solution must agree with your reference on all of them. Therefore the description must fully determine the output for every valid input: state input constraints, tie-breaking, ordering, edge cases (empty inputs, duplicates, negatives) and the exact output format.
 3. The target fails all 4 fresh samples in an isolated re-evaluation.
 4. It is not a duplicate of another of your tasks (exact, renamed, reworded or behaviourally equivalent tasks are all duplicates).
Ambiguous, underspecified, trick, unsolvable or checker-exploiting tasks never count, even if the target fails them. Misleading examples, hidden conventions and output-format traps are not capability gaps; aim for genuine algorithmic, reasoning, state-tracking or specification-following difficulty that a careful programmer handles reliably. Every task you write is submitted and counted in the denominator; you cannot withdraw tasks.
Do not copy or lightly paraphrase problems from existing benchmarks or sites (HumanEval, MBPP, APPS, LeetCode, Codeforces, Project Euler, Advent of Code, textbooks); write original tasks.

HARD CONSTRAINTS FOR EACH TASK
- signature: one line "def name(a: T1, b: T2) -> R:". Parameter types may use only int, float, str, bool, None, List[...], Dict[str, ...], Optional[...]. No tuples, sets, bytes or callables as parameters. No default values, *args or **kwargs. The return annotation is required (tuples are allowed in return values).
- reference_solution: complete Python 3.12 source defining the entry-point function at top level (helpers allowed). Pure and deterministic; standard library only; no I/O, randomness, time, os/sys/subprocess/network, eval/exec/open. At most 2 seconds per call.
- public_inputs: 1 to 3 example argument lists (shown to the target as doctests).
- hidden_inputs: 6 to 20 argument lists (never shown to the target). Each argument list is a JSON array of positional arguments, at most 4 KB. Cover edge cases.
- claimed_public_outputs: what you believe the outputs on public_inputs are (JSON), used only as a consistency check of your reference.
- description: at most 2000 characters, plain text, no triple quotes, must not contain the solution code.

OUTPUT FORMAT: return ONLY one JSON object, no markdown fences, no commentary:
{{"tasks": [{{"family": "short topic label", "entry_point": "name", "signature": "def name(xs: List[int], k: int) -> int:", "description": "...", "public_inputs": [[...]], "hidden_inputs": [[...], ...], "claimed_public_outputs": [...], "reference_solution": "def name(xs, k):\\n    ...", "derived_from": ["candidate ids this task builds on, if any"], "rationale": "why you expect the target to fail (not shown to the target)"}}]}}
"""

VALIDATOR_SYSTEM = """You are an independent expert Python programmer acting as a verifier for a research benchmark. You are given ONLY a function prompt (signature plus docstring with examples). Nobody else's solution is available to you.

Do three things:
 1. Write a complete, correct Python 3.12 implementation of the function exactly as specified (standard library only, deterministic, no I/O). Include the def line; helpers are allowed.
 2. Propose up to 8 additional test inputs that are VALID according to the specification, emphasising edge cases where an implementation could plausibly go wrong. Each input is a JSON array of positional arguments.
 3. Judge whether the specification is ambiguous: could two careful programmers reasonably return different outputs for some valid input? If so, set "ambiguous" to true and explain briefly.

Return ONLY one JSON object, no markdown fences:
{"solution": "def ...", "extra_inputs": [[...], ...], "ambiguous": false, "ambiguity_note": ""}
"""

BLIND_SOLVE_SYSTEM = """You are an expert Python programmer. Implement the function below exactly as specified (Python 3.12, standard library only, deterministic, no I/O). Include the def line; helpers are allowed.

Return ONLY one JSON object, no markdown fences:
{"solution": "def ..."}
"""


def budget_section(max_ast_nodes: int) -> str:
    return (f"COMPLEXITY BUDGET (enforced by the operator)\n"
            f"- reference_solution may contain at most {max_ast_nodes} AST nodes, counted exactly as "
            f"sum(1 for _ in ast.walk(ast.parse(reference_solution))). Comments are free; docstrings count. "
            f"That is roughly 12-15 short lines: one small function, perhaps one tiny helper.\n"
            f"- A task over budget is invalid and still uses one of your candidate slots.\n"
            f"- The goal is SIMPLE tasks that this target fails: a short, exact specification whose correct "
            f"solution is small, so the failure reveals a specific missing capability rather than sheer "
            f"difficulty. The other scoring rules above are unchanged.\n\n")


def teacher_system(max_ast_nodes: int | None = None) -> str:
    if max_ast_nodes is None:
        return TEACHER_SYSTEM
    return TEACHER_SYSTEM.replace("OUTPUT FORMAT:", budget_section(max_ast_nodes) + "OUTPUT FORMAT:", 1)


def teacher_round_message(*, round_index: int, rounds: int, per_round: int, candidates_used: int,
                          candidates_max: int, history: list, feedback_mode: str) -> str:
    """Round 1 is byte-identical across arms. Later rounds differ only in the history block."""
    lines = [f"ROUND {round_index + 1} of {rounds}. Write exactly {per_round} new tasks.",
             f"Candidate allowance used so far: {candidates_used} of {candidates_max}."]
    if round_index == 0:
        lines.append("This is the first round; no previous tasks exist.")
    elif feedback_mode == "none":
        lines.append("In this condition you receive NO information about the target's attempts or about "
                     "operator verification. Your previously submitted tasks (summaries only) are listed "
                     "so that you avoid duplicates:")
        lines.append(json.dumps(history, indent=1, ensure_ascii=False))
    else:
        lines.append("Operator feedback on your previous tasks (the target's attempts were made with the "
                     "frozen target and the exact rendered prompt; reference outputs come from executing "
                     "your reference). Use it to find genuine gaps; remember that ambiguous or "
                     "duplicate tasks never count:")
        lines.append(json.dumps(history, indent=1, ensure_ascii=False))
    lines.append('Return ONLY the JSON object {"tasks": [...]} with exactly '
                 f"{per_round} tasks.")
    return "\n\n".join(lines)


def validator_message(prompt: str) -> str:
    return "FUNCTION PROMPT:\n\n" + prompt


def extract_json_object(text: str):
    """Return the first decodable top-level JSON object in text, or raise ValueError."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty response text")
    stripped = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found")


def prompts_sha256(max_ast_nodes: int | None = None) -> str:
    material = json.dumps([PROMPTS_VERSION, teacher_system(max_ast_nodes), VALIDATOR_SYSTEM, BLIND_SOLVE_SYSTEM,
                           TARGET_DESCRIPTION], sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# v4: minimal-pair frontier gaps
PAIR_PROMPTS_VERSION = "distillgap_pair_prompts_v2_targets_validity"

PAIR_TEACHER_SYSTEM = TEACHER_SYSTEM.replace(
    "Your job: write ORIGINAL, self-contained Python programming tasks that are unambiguous and solvable by a "
    "competent programmer (and by you), but that the target model below fails.",
    "Your job: find the EDGE of the target's competence. Write ORIGINAL minimal PAIRS of self-contained Python "
    "programming tasks. In each pair, the CONTROL task is one the target model below solves, and the CHALLENGE "
    "task differs from the control by exactly ONE stated requirement that makes the target fail. Both tasks "
    "must be unambiguous and solvable by a competent programmer (and by you).").replace(
    "HOW YOUR TASKS ARE SCORED (by an independent operator after discovery ends; you will not see this evaluation):\n"
    "A task counts as a verified capability gap only if ALL of the following hold.",
    "HOW YOUR PAIRS ARE SCORED (by an independent operator after discovery ends; you will not see this evaluation):\n"
    "A pair counts as a verified FRONTIER GAP only if: the target passes the CONTROL on at least 2 of 4 fresh "
    "samples, AND fails the CHALLENGE on all 4 fresh samples, AND each of the two tasks satisfies all of the "
    "following.").replace(
    "4. It is not a duplicate of another of your tasks (exact, renamed, reworded or behaviourally equivalent "
    "tasks are all duplicates).",
    "4. The challenge is not a duplicate of another of your challenges (exact, renamed, reworded or "
    "behaviourally equivalent tasks are all duplicates).\n"
    "PAIR RULES: control and challenge use the SAME entry_point and the SAME signature. The challenge "
    "description is the control description with one requirement added or changed (the operator requires "
    "substantial wording overlap). The two references must behave differently on at least one input. A pair "
    "whose control the target fails scores nothing, so do not make the control hard: make it a task the target "
    "reliably solves, then change exactly one thing.").replace(
    "Every task you write is submitted and counted in the denominator; you cannot withdraw tasks.",
    "Every pair you write is submitted and counted in the denominator; you cannot withdraw pairs.")
PAIR_TEACHER_SYSTEM = PAIR_TEACHER_SYSTEM[:PAIR_TEACHER_SYSTEM.index("OUTPUT FORMAT:")] + (
    "OUTPUT FORMAT: return ONLY one JSON object, no markdown fences, no commentary:\n"
    '{"pairs": [{"pair_family": "short topic label", "changed_requirement": "the single requirement that differs, '
    'in one sentence", "control": {TASK}, "challenge": {TASK}, "derived_from": ["pair ids this builds on, if any"], '
    '"rationale": "why the target solves the control but fails the challenge (not shown to the target)"}]}\n'
    'where each {TASK} is {"family": "...", "entry_point": "name", "signature": "def name(...) -> ...:", '
    '"description": "...", "public_inputs": [[...]], "hidden_inputs": [[...], ...], "claimed_public_outputs": [...], '
    '"reference_solution": "def name(...):\\n    ..."}\n')


def pair_teacher_system(max_ast_nodes: int | None = None, target_description: str | None = None) -> str:
    """v5: the target paragraph comes from the target profile; Qwen's text is unchanged when it is the target."""
    text = PAIR_TEACHER_SYSTEM
    if target_description is not None:
        if TARGET_DESCRIPTION not in text:
            raise ValueError("target paragraph not found")
        text = text.replace(TARGET_DESCRIPTION, target_description)
    if max_ast_nodes is None:
        return text
    section = budget_section(max_ast_nodes).replace(
        "- reference_solution may", "- each reference_solution (control and challenge) may")
    return text.replace("OUTPUT FORMAT:", section + "OUTPUT FORMAT:", 1)


def pair_round_message(*, round_index: int, rounds: int, per_round: int, pairs_used: int, pairs_max: int,
                       history: list, feedback_mode: str) -> str:
    lines = [f"ROUND {round_index + 1} of {rounds}. Write exactly {per_round} new pairs.",
             f"Pair allowance used so far: {pairs_used} of {pairs_max}."]
    if round_index == 0:
        lines.append("This is the first round; no previous pairs exist.")
    elif feedback_mode == "none":
        lines.append("In this condition you receive NO information about the target's attempts or about operator "
                     "verification. Your previously submitted pairs (summaries only) are listed so that you avoid "
                     "duplicates:")
        lines.append(json.dumps(history, indent=1, ensure_ascii=False))
    elif feedback_mode == "validity":
        lines.append("In this condition you receive operator VALIDITY feedback on your previous pairs (execution checks, "
                     "pair checks, duplicates) but NO information about the target's attempts:")
        lines.append(json.dumps(history, indent=1, ensure_ascii=False))
    else:
        lines.append("Operator feedback on your previous pairs (the target's attempts were made with the frozen target "
                     "and the exact rendered prompts; reference outputs come from executing your references). A pair "
                     "scores only if the target passes the control and fails the challenge:")
        lines.append(json.dumps(history, indent=1, ensure_ascii=False))
    lines.append('Return ONLY the JSON object {"pairs": [...]} with exactly ' f"{per_round} pairs.")
    return "\n\n".join(lines)


def pair_prompts_sha256(max_ast_nodes: int | None = None, target_description: str | None = None) -> str:
    material = json.dumps([PAIR_PROMPTS_VERSION, pair_teacher_system(max_ast_nodes, target_description), VALIDATOR_SYSTEM,
                           BLIND_SOLVE_SYSTEM, target_description or TARGET_DESCRIPTION], sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()
