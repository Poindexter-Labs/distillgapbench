"""DistillGap v1 students (the four PostTrainBench base models) and the per-environment target paragraph that the
researcher prompt shows. Decoding is the frozen v5 rule (temperature 0.6, top_p 0.95, no top-k, 4 samples, per-item
seed); each environment fixes its own stop strings and new-token cap (envs/*.student_prompt, CODING_* below)."""
from __future__ import annotations

import string

STUDENTS = {
    "qwen1p7b": {"repo": "Qwen/Qwen3-1.7B-Base", "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                 "size": "1.7-billion-parameter"},
    "qwen4b": {"repo": "Qwen/Qwen3-4B-Base", "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
               "size": "4-billion-parameter"},
    "smol3b": {"repo": "HuggingFaceTB/SmolLM3-3B-Base", "revision": "d78a42f79198603e614095753484a04c10c2b940",
               "size": "3-billion-parameter"},
    "gemma4b": {"repo": "google/gemma-3-4b-pt", "revision": "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
                "size": "4-billion-parameter"},
}
STUDENT_ORDER = ["qwen1p7b", "qwen4b", "smol3b", "gemma4b"]

# Coding: the v5 rule "halt at EOS or after a newline followed by a non-whitespace character", expressed exactly as
# vLLM stop strings (the program is still truncated post hoc at the first non-indented line, schema.program_from_completion).
CODING_STOP = ["\n" + c for c in string.printable if not c.isspace()]
CODING_MAX_TOKENS = 512

ENV_BEHAVIOUR = {
    "coding": ("It receives only the rendered prompt below and continues the text to write the function body. It draws "
               "4 independent samples (temperature 0.6, top_p 0.95, at most 512 new tokens); the body ends at the first "
               "non-indented line."),
    "math": ("It sees four short worked examples and then your problem, writes a solution, and must end with "
             "\"Final answer: <number>\". It draws 4 independent samples (temperature 0.6, top_p 0.95, at most 400 new "
             "tokens); generation stops at the next \"Problem:\" line or after two blank lines."),
    "output": ("It sees three worked examples and then your code, and completes the assert line with a Python literal. "
               "It draws 4 independent samples (temperature 0.6, top_p 0.95, at most 64 new tokens); generation stops at "
               "the end of that line."),
}


def teacher_description(student_id: str, env: str) -> str:
    s = STUDENTS[student_id]
    return (f"{s['repo']} (revision {s['revision'][:8]}), a {s['size']} PRETRAINED BASE language model (not "
            f"instruction-tuned), with frozen weights. {ENV_BEHAVIOUR[env]}")
