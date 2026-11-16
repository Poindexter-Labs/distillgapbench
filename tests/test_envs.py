"""Self-tests for the v6 environments (fast; run with pytest from v6/)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs import ENVS, math_env, planning_env, sokoban_env  # noqa: E402


def test_prompts_build_and_shots_valid():
    for env in ENVS.values():
        inst = env.template(random.Random(1), 1)
        p = env.student_prompt(inst)
        assert p["prompt"].endswith((":", "==")) and p["stop"] and p["max_tokens"] > 0


def test_sokoban_solver_and_scoring():
    level = "#######\n#     #\n# $ . #\n#  @  #\n#######"
    res = sokoban_env.check({"level": level})
    assert res["valid"] and res["optimal"] >= 1
    assert sokoban_env.score({"level": level}, res["witness"] + "\n")
    assert not sokoban_env.score({"level": level}, "UUUU")
    assert not sokoban_env.score({"level": level}, "hello")
    assert not sokoban_env.check({"level": "#####\n#@$ #\n#####"})["valid"]  # no goal
    assert not sokoban_env.check({"level": "#####\n#@$#.\n#####"})["valid"]  # not enclosed / unsolvable
    assert not sokoban_env.check({"level": "####\n#@*#\n####"})["valid"]  # already solved


def test_planning_solver_and_scoring():
    inst = {"initial": [["A", "B"]], "goal": [["A", "B"]]}
    res = planning_env.check(inst)
    assert res["valid"] and res["optimal"] == 2
    assert planning_env.score(inst, "move B to table; move A to B")
    assert not planning_env.score(inst, "move A to B")
    assert not planning_env.check({"initial": [["A"], ["B"]], "goal": [["A", "B"], ["B", "A"]]})["valid"]


def test_math_parsing_and_scoring():
    inst = {"problem": "What is one half of nine, as a fraction?", "answer": "9/2"}
    assert math_env.check(inst)["valid"]
    assert math_env.score(inst, " 9 halves. Final answer: 9/2")
    assert math_env.score(inst, " Final answer: 4.5")
    assert math_env.score(inst, " Final answer: \\frac{9}{2}")
    assert not math_env.score(inst, " Final answer: 4")
    assert not math_env.score(inst, " the answer is 4.5")


def test_templates_valid_and_witness_passes():
    rng = random.Random(7)
    for level in (1, 2, 3):
        s = sokoban_env.template(rng, level)
        assert sokoban_env.score(s, sokoban_env.check(s)["witness"])
        p = planning_env.template(rng, level)
        assert planning_env.score(p, planning_env.check(p)["witness"])
        m = math_env.template(rng, level)
        assert math_env.score(m, f"Final answer: {m['answer']}")


def test_output_prediction_checks_and_scoring():
    from envs import output_env
    inst = {"code": "def f(x):\n    return x[::-1]", "call": "f('abc')"}
    res = output_env.check(inst)
    assert res["valid"] and res["expected_repr"] == "'cba'"
    inst["expected_repr"] = res["expected_repr"]
    assert output_env.score(inst, ' "cba"') and output_env.score(inst, " 'cba' [/PYTHON]")
    assert not output_env.score(inst, " 'abc'") and not output_env.score(inst, " cba")
    assert not output_env.check({"code": "import os\ndef f(x):\n    return 1", "call": "f(1)"})["valid"]
    assert not output_env.check({"code": "def f(x):\n    while True:\n        pass", "call": "f(1)"})["valid"]
    assert not output_env.check({"code": "def f(x):\n    return object()", "call": "f(1)"})["valid"]


def test_planning_witness_is_hash_seed_independent():
    import subprocess
    code = ("import sys; sys.path.insert(0, '.'); from envs import planning_env as p; "
            "print([p.check(s)['witness'] for s in p.SHOTS])")
    here = str(Path(__file__).resolve().parents[1])
    outs = {subprocess.run([sys.executable, "-c", code], cwd=here, capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"}).stdout for seed in (1, 2, 3)}
    assert len(outs) == 1
