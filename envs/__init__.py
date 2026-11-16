"""DistillGap v6 environments beyond coding. Each module exposes the same contract:

  NAME, LEVELS
  check(inst) -> {"valid", "reasons", ...}      independent validity (exact solver / search / operator execution)
  score(inst, completion) -> bool               student pass/fail for one sampled completion
  extract(completion) -> parsed answer or None  format check
  student_prompt(inst) -> {"prompt", "stop", "max_tokens"}   few-shot completion prompt for base students
  template(rng, level) -> inst                  operator generator (calibration and the template arm)

Calibration 2026-11-13 (docs/CALIBRATION_v6_students.md): sokoban, chess and one-shot planning are at the floor for
the four base students and are kept only for reference; math is calibrated; output and planning_cot under test.
"""
from . import math_env, output_env, planning_cot_env, planning_env, sokoban_env

ENVS = {m.NAME: m for m in (math_env, output_env, planning_cot_env, sokoban_env, planning_env)}
try:  # appendix environment; python-chess is not needed by the main study
    from . import chess_env
    ENVS[chess_env.NAME] = chess_env
except ImportError:
    pass
