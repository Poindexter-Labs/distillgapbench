"""Build the v1 leaderboard from cell reports (report.json per cell). Standard library only, so it also runs read-only
on the SovAI login node with `--out -` (JSON on stdout, summary on stderr).

  python scripts/collect_grid.py --runs DIR [DIR ...] --out leaderboard_v1.json [--exclude kimi] [--md table.md]

Score per arm run = frontier gaps after judge vetoes / pairs requested (16). Per (researcher, student, environment):
mean and SD over repetitions, for the adaptive arm (the scored arm) and the batch reference arm. Overall per researcher,
PostTrainBench style: for each repetition, the mean over the 12 student x environment cells; reported as mean and SD
over repetitions. Paired contrast adaptive - batch per researcher on (student, environment, repetition) units, which
share the arm seed: wins/ties/losses, exact sign test, Wilcoxon signed-rank (normal approximation, tie-corrected),
Holm over researchers. Integrity: completeness, short arms, call outcomes (truncation), rejection reasons, validator
disagreements, judge vetoes (ambiguity rule and the strict sensitivity rule).
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

SCHEMA = "distillgap_v1_leaderboard_2"
STUDENTS = ["qwen1p7b", "qwen4b", "smol3b", "gemma4b"]
ENVS = ["coding", "math", "output"]
REPS = [1, 2, 3]


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return (round(statistics.mean(values), 4), round(statistics.stdev(values), 4) if len(values) > 1 else None,
            len(values))


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilcoxon(diffs):
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks, ties, i = [0.0] * n, 0, 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        ties += (j - i + 1) ** 3 - (j - i + 1)
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    if var <= 0:
        return None
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
    return min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2)))


def holm(pvalues):
    keys = sorted((k for k, p in pvalues.items() if p is not None), key=lambda k: pvalues[k])
    out, running = {}, 0.0
    for index, key in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - index) * pvalues[key]))
        out[key] = round(running, 6)
    return out


def load_arms(runs, exclude, pairs_default):
    arms, seen, status, costs = [], {}, Counter(), defaultdict(Counter)
    for run in runs:
        for path in sorted(Path(run).glob("*/cell_status.json")):
            status[json.loads(path.read_text()).get("state", "unknown")] += 1
        for report_path in sorted(Path(run).glob("*/report.json")):
            report = json.loads(report_path.read_text())
            cell = report["cell"]
            researcher = cell["researcher"] or "template"
            if researcher in exclude:
                continue
            costs[researcher]["validator_usd"] += report.get("validator_usd_total") or 0.0
            costs[researcher]["judge_usd"] += report.get("judge_usd_total") or 0.0
            costs[researcher]["teacher_usd"] += sum(a.get("teacher_estimated_usd") or 0.0 for a in report["per_arm"])
            for arm in report["per_arm"]:
                key = (researcher, cell["student"], cell["env"], arm["arm_id"])
                if key in seen:
                    raise SystemExit(f"duplicate arm {key}: {seen[key]} and {report_path}")
                seen[key] = str(report_path)
                requested = (cell.get("arm_overrides") or {}).get("max_candidates", pairs_default)
                gaps = arm.get("frontier_gaps") or 0
                judged = arm.get("frontier_gaps_judged")
                strict = arm.get("frontier_gaps_judged_strict")
                verdicts = Counter()
                for roles in (arm.get("judge_verdicts") or {}).values():
                    for role, v in roles.items():
                        verdicts[f"{role}:{v}"] += 1
                arms.append({
                    "researcher": researcher, "student": cell["student"], "env": cell["env"], "kind": arm["kind"],
                    "rep": int(arm["arm_id"].split("-", 1)[0][3:]), "arm_id": arm["arm_id"], "cell_id": cell["cell_id"],
                    "pairs_requested": requested,
                    "frontier_gaps": gaps,
                    "frontier_gaps_judged": gaps if judged is None else judged,
                    "frontier_gaps_judged_strict": gaps if strict is None else strict,
                    "pairs_attempted": arm.get("pairs_attempted") or 0,
                    "pairs_locally_valid": arm.get("pairs_locally_valid") or 0,
                    "pairs_independently_valid": arm.get("pairs_independently_valid") or 0,
                    "control_final_mean_pass_rate": arm.get("control_final_mean_pass_rate"),
                    "local_reasons": arm.get("pair_local_failure_reasons") or {},
                    "invalid_disagreement": arm.get("invalid_disagreement") or 0,
                    "validator_incomplete": arm.get("validator_incomplete") or 0,
                    "judge_vetoed": len(arm.get("judge_vetoed_ids") or []),
                    "judge_strict_vetoed": len(arm.get("judge_strict_vetoed_ids") or []),
                    "judge_incomplete": arm.get("judge_incomplete") or 0,
                    "judge_verdicts": dict(verdicts),
                    "teacher_usd": arm.get("teacher_estimated_usd") or 0.0, "judge_usd": arm.get("judge_usd") or 0.0,
                    "teacher_calls": arm.get("teacher_calls") or 0, "call_outcomes": arm.get("teacher_call_outcomes") or {},
                    "teacher_output_tokens": arm.get("teacher_output_tokens"),
                    "parse_failures": arm.get("teacher_parse_failures") or 0, "stop_reason": arm.get("stop_reason"),
                    "wall_seconds": arm.get("arm_wall_seconds")})
    for a in arms:
        a["score"] = a["frontier_gaps_judged"] / a["pairs_requested"]
        a["score_unjudged"] = a["frontier_gaps"] / a["pairs_requested"]
        a["score_strict"] = a["frontier_gaps_judged_strict"] / a["pairs_requested"]
    costs = {r: {k: round(v, 2) for k, v in c.items()} for r, c in sorted(costs.items())}
    return arms, dict(status), costs


def cell_table(arms):
    groups = defaultdict(list)
    for a in arms:
        groups[(a["researcher"], a["student"], a["env"], a["kind"])].append(a)
    table = []
    for (researcher, student, env, kind), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["rep"])
        entry = {"researcher": researcher, "student": student, "env": env, "kind": kind,
                 "reps": [r["rep"] for r in rows], "gaps_judged": [r["frontier_gaps_judged"] for r in rows],
                 "teacher_usd_mean": round(statistics.mean(r["teacher_usd"] for r in rows), 2)}
        for name in ("score", "score_unjudged", "score_strict"):
            m, sd, _ = mean_sd([r[name] for r in rows])
            entry[name], entry[name + "_sd"] = m, sd
        table.append(entry)
    return table


def overall(arms, researcher, kind, field="score"):
    rows = [a for a in arms if a["researcher"] == researcher and a["kind"] == kind]
    if not rows:
        return None
    by_rep = defaultdict(list)
    for a in rows:
        by_rep[a["rep"]].append(a)
    expected = len(STUDENTS) * len(ENVS)
    per_rep = {rep: statistics.mean(a[field] for a in v) for rep, v in sorted(by_rep.items())}
    m, sd, n = mean_sd(list(per_rep.values()))
    out = {"mean": m, "sd": sd, "reps": n, "per_rep": {str(k): round(v, 4) for k, v in per_rep.items()},
           "complete": all(len(v) == expected for v in by_rep.values()) and sorted(by_rep) == REPS,
           "arms": len(rows), "by_env": {}}
    for env in ENVS:
        env_rows = [a for a in rows if a["env"] == env]
        if not env_rows:
            continue
        env_rep = defaultdict(list)
        for a in env_rows:
            env_rep[a["rep"]].append(a[field])
        em, esd, en = mean_sd([statistics.mean(v) for v in env_rep.values()])
        out["by_env"][env] = {"mean": em, "sd": esd, "reps": en, "arms": len(env_rows),
                              "complete": len(env_rows) == len(STUDENTS) * len(REPS)}
    return out


def contrast(arms, researcher, field="frontier_gaps_judged"):
    units = defaultdict(dict)
    for a in arms:
        if a["researcher"] == researcher and a["kind"] in ("adaptive", "batch"):
            units[(a["student"], a["env"], a["rep"])][a["kind"]] = a[field]
    paired = {u: v["adaptive"] - v["batch"] for u, v in sorted(units.items()) if len(v) == 2}
    if not paired:
        return None
    diffs = list(paired.values())
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    by_env = {}
    for env in ENVS:
        d = [v for u, v in paired.items() if u[1] == env]
        if d:
            by_env[env] = {"n": len(d), "mean_difference": round(statistics.mean(d), 3),
                           "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d)}
    return {"metric": field, "contrast": "adaptive - batch", "n": len(diffs),
            "mean_difference": round(statistics.mean(diffs), 3),
            "sd_difference": round(statistics.stdev(diffs), 3) if len(diffs) > 1 else None,
            "wins": wins, "ties": len(diffs) - wins - losses, "losses": losses,
            "sign_test_p": sign_test(wins, losses), "wilcoxon_p": wilcoxon(diffs), "by_env": by_env}


def integrity(arms, researcher):
    rows = [a for a in arms if a["researcher"] == researcher]
    reasons, outcomes, stops, verdicts = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        reasons.update(r["local_reasons"])
        outcomes.update(r["call_outcomes"])
        stops[r["stop_reason"]] += 1
        verdicts.update(r["judge_verdicts"])
    calls = sum(outcomes.values())
    kinds = {k for k in (r["kind"] for r in rows)}
    expected = len(STUDENTS) * len(ENVS) * len(REPS) * (1 if kinds == {"template"} else 2)
    return {
        "arms": len(rows), "arms_expected": expected,
        "arms_short": sum(1 for r in rows if r["pairs_attempted"] < r["pairs_requested"]),
        "short_arm_ids": sorted(f"{r['cell_id']}/{r['arm_id']}:{r['pairs_attempted']}" for r in rows
                                if r["pairs_attempted"] < r["pairs_requested"]),
        "stop_reasons": dict(stops),
        "pairs_attempted": sum(r["pairs_attempted"] for r in rows),
        "pairs_locally_valid": sum(r["pairs_locally_valid"] for r in rows),
        "pairs_independently_valid": sum(r["pairs_independently_valid"] for r in rows),
        "local_rejection_reasons": dict(reasons.most_common()),
        "validator_disagreements": sum(r["invalid_disagreement"] for r in rows),
        "validator_incomplete": sum(r["validator_incomplete"] for r in rows),
        "frontier_gaps_unjudged": sum(r["frontier_gaps"] for r in rows),
        "judge_vetoes": sum(r["judge_vetoed"] for r in rows),
        "judge_strict_vetoes": sum(r["judge_strict_vetoed"] for r in rows),
        "judge_incomplete": sum(r["judge_incomplete"] for r in rows),
        "judge_verdicts": dict(verdicts),
        "call_outcomes": dict(outcomes), "calls": calls,
        "truncation_rate": round(outcomes.get("truncated", 0) / calls, 4) if calls else None,
        "parse_failures": sum(r["parse_failures"] for r in rows),
        "teacher_usd": round(sum(r["teacher_usd"] for r in rows), 2),
        "judge_usd": round(sum(r["judge_usd"] for r in rows), 2)}


def markdown(result, pairs):
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}"
    pm = lambda o: "–" if not o else (f"{pct(o['mean'])} ± {pct(o['sd'])}" if o.get("sd") is not None else pct(o["mean"]))
    lines = [f"Frontier-gap rate (% of {pairs} pairs, after judge vetoes), adaptive arm; mean ± SD over repetitions.",
             "", "| # | Researcher | Overall | Coding | Math | Output | Batch overall | Adaptive − batch (gaps/arm) | Wins/ties/losses | Holm p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows = sorted(((r, o) for r, o in result["overall"].items() if "adaptive" in o),
                  key=lambda kv: -(kv[1]["adaptive"]["mean"] or 0))
    for rank, (researcher, o) in enumerate(rows, 1):
        env = o["adaptive"]["by_env"]
        c = result["contrasts"].get(researcher) or {}
        flag = "" if o["adaptive"]["complete"] else " (incomplete)"
        lines.append(f"| {rank} | {researcher}{flag} | {pm(o['adaptive'])} | {pm(env.get('coding'))} | {pm(env.get('math'))} | "
                     f"{pm(env.get('output'))} | {pm(o.get('batch'))} | {c.get('mean_difference', '–')} | "
                     f"{c.get('wins', '–')}/{c.get('ties', '–')}/{c.get('losses', '–')} | "
                     f"{result['contrasts_holm'].get(researcher, '–')} |")
    if "template" in result["overall"]:
        t = result["overall"]["template"]["template"]
        lines.append(f"| – | template (operator) | {pm(t)} | {pm(t['by_env'].get('coding'))} | {pm(t['by_env'].get('math'))} | "
                     f"{pm(t['by_env'].get('output'))} | – | – | – | – |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="output JSON path, or - for stdout")
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--exclude", nargs="*", default=[], help="researcher ids left out (route diagnostics)")
    parser.add_argument("--md", default=None, help="also write the markdown leaderboard table here")
    parser.add_argument("--no-arms", action="store_true", help="omit the per-arm rows (smaller output)")
    args = parser.parse_args()
    arms, status, costs = load_arms(args.runs, set(args.exclude), args.pairs)
    researchers = sorted({a["researcher"] for a in arms})
    result = {"schema": SCHEMA, "score": "frontier gaps after judge vetoes / pairs requested",
              "excluded": sorted(args.exclude), "runs": args.runs, "cell_status": status,
              "overall": {}, "overall_unjudged": {}, "overall_strict": {}, "contrasts": {}, "contrasts_strict": {},
              "costs_usd": costs, "costs_usd_total": {k: round(sum(c.get(k, 0) for c in costs.values()), 2)
                                                      for k in ("teacher_usd", "validator_usd", "judge_usd")},
              "cells": cell_table(arms), "integrity": {r: integrity(arms, r) for r in researchers}}
    for r in researchers:
        for kind in ("adaptive", "batch", "template"):
            for key, field in (("overall", "score"), ("overall_unjudged", "score_unjudged"), ("overall_strict", "score_strict")):
                o = overall(arms, r, kind, field)
                if o:
                    result[key].setdefault(r, {})[kind] = o
        c = contrast(arms, r)
        if c:
            result["contrasts"][r] = c
            result["contrasts_strict"][r] = contrast(arms, r, "frontier_gaps_judged_strict")
    result["contrasts_holm"] = holm({r: c["wilcoxon_p"] for r, c in result["contrasts"].items()})
    diffs = [c["mean_difference"] for c in result["contrasts"].values()]
    up, down = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    result["contrast_across_researchers"] = {"researchers": len(diffs), "adaptive_better": up, "batch_better": down,
                                             "sign_test_p": sign_test(up, down)}
    if not args.no_arms:
        result["arms"] = arms
    text = json.dumps(result, indent=1) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)
    if args.md:
        Path(args.md).write_text(markdown(result, args.pairs))
    log = sys.stderr if args.out == "-" else sys.stdout
    print(markdown(result, args.pairs), file=log)
    for r, i in result["integrity"].items():
        print(f"{r:10s} arms {i['arms']}/{i['arms_expected']} short {i['arms_short']} truncation {i['truncation_rate']} "
              f"vetoes {i['judge_vetoes']}/{i['frontier_gaps_unjudged']} strict {i['judge_strict_vetoes']} "
              f"usd {i['teacher_usd']}", file=log)


if __name__ == "__main__":
    main()
