"""collect_grid on synthetic cell reports: scores, overall per repetition, paired contrast, exclusion, duplicates.

  .target_venv/bin/python v6/tests/test_collect_grid.py
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import collect_grid as cg  # noqa: E402


def arm(rep, kind, gaps, vetoed=0, strict=0, attempted=16):
    ids = [f"rep{rep}-{kind}-p{i}" for i in range(gaps)]
    return {"arm_id": f"rep{rep}-{kind}", "kind": kind, "frontier_gaps": gaps, "frontier_gap_ids": ids,
            "frontier_gaps_judged": gaps - vetoed, "frontier_gaps_judged_strict": gaps - strict,
            "judge_vetoed_ids": ids[:vetoed], "judge_strict_vetoed_ids": ids[:strict], "pairs_attempted": attempted,
            "pairs_locally_valid": attempted, "teacher_estimated_usd": 1.0, "teacher_call_outcomes": {"complete": 15, "truncated": 1},
            "stop_reason": "candidate_allowance_or_call_limit", "judge_verdicts": {i: {"control": "perfect", "challenge": "perfect"} for i in ids}}


def write_cell(run, researcher, student, env, per_arm):
    cell_id = f"{researcher or 'template'}__{student}__{env}"
    d = Path(run) / cell_id
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({"cell": {"cell_id": cell_id, "researcher": researcher, "student": student,
                                                        "env": env}, "per_arm": per_arm}))
    (d / "cell_status.json").write_text(json.dumps({"state": "done"}))


def build(run):
    for s in cg.STUDENTS:
        for e in cg.ENVS:
            write_cell(run, "good", s, e, [a for r in cg.REPS for a in (arm(r, "adaptive", 8 + r, vetoed=1, strict=2),
                                                                          arm(r, "batch", 4))])
            write_cell(run, "diag", s, e, [arm(1, "adaptive", 16), arm(1, "batch", 0)])
            write_cell(run, None, s, e, [arm(r, "template", 1) for r in cg.REPS])


def test_statistics():
    assert abs(cg.sign_test(9, 1) - 22 / 1024) < 1e-12
    assert cg.sign_test(0, 0) is None
    # all ten differences positive, distinct: W+ = 55, z = (27.5 - 0.5) / sqrt(96.25)
    assert abs(cg.wilcoxon(list(range(1, 11))) - math.erfc(27 / math.sqrt(96.25) / math.sqrt(2))) < 1e-12
    assert cg.wilcoxon([0, 0]) is None
    assert cg.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_leaderboard():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        build(run)
        out = Path(tmp) / "lb.json"
        subprocess.run([sys.executable, str(HERE / "scripts/collect_grid.py"), "--runs", str(run), "--out", str(out),
                        "--exclude", "diag", "--md", str(Path(tmp) / "lb.md")], check=True, capture_output=True)
        lb = json.loads(out.read_text())
        assert "diag" not in lb["overall"] and lb["excluded"] == ["diag"]
        good = lb["overall"]["good"]["adaptive"]
        # judged gaps per rep r: 8 + r - 1 -> per-rep overall (7+r)/16
        assert good["per_rep"] == {"1": 0.5, "2": 0.5625, "3": 0.625} and good["complete"]
        assert abs(good["mean"] - 0.5625) < 1e-9 and abs(good["sd"] - 0.0625) < 1e-9
        assert abs(lb["overall_unjudged"]["good"]["adaptive"]["mean"] - 10 / 16) < 1e-9
        assert abs(lb["overall_strict"]["good"]["adaptive"]["mean"] - 8 / 16) < 1e-9
        c = lb["contrasts"]["good"]
        assert c["n"] == 36 and c["wins"] == 36 and abs(c["mean_difference"] - 5.0) < 1e-9
        assert lb["overall"]["template"]["template"]["mean"] == 1 / 16
        i = lb["integrity"]["good"]
        assert i["arms"] == i["arms_expected"] == 72 and i["arms_short"] == 0 and i["truncation_rate"] == 0.0625
        assert i["judge_vetoes"] == 36 and i["judge_strict_vetoes"] == 72
        assert lb["integrity"]["template"]["arms_expected"] == 36
        assert "| 1 | good |" in (Path(tmp) / "lb.md").read_text()
        # the same arm in two runs is an error (e.g. a void run passed by mistake)
        dup = subprocess.run([sys.executable, str(HERE / "scripts/collect_grid.py"), "--runs", str(run), str(run),
                              "--out", "-"], capture_output=True, text=True)
        assert dup.returncode != 0 and "duplicate arm" in dup.stderr


if __name__ == "__main__":
    test_statistics()
    test_leaderboard()
    print("collect_grid tests passed")
