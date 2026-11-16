"""Exact denominators, yields and paired summaries. No significance tests are computed:
the pilot is exploratory; independent arm runs (not generated tasks) are the units."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import statistics

from .canonical import family_clusters, shingles


def _load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def arm_metrics(arm_dir, final_rows_by_id):
    arm_dir = Path(arm_dir)
    summary = json.loads((arm_dir / "arm_summary.json").read_text())
    cands = _load_jsonl(arm_dir / "candidates.jsonl")
    n = len(cands)
    schema_invalid = sum(1 for c in cands if c["local"] and not c["local"]["valid"]
                         and any(r.startswith("schema:") for r in c["local"]["reasons"]))
    local_invalid = sum(1 for c in cands if c["local"] and not c["local"]["valid"])
    duplicates = sum(1 for c in cands if c["local"] and c["local"]["valid"] and c["duplicate_of"])
    evaluable = [c for c in cands if c["local"] and c["local"]["valid"] and not c["duplicate_of"]]
    disc = [c["discovery"] for c in evaluable if c.get("discovery") and c["discovery"].get("attempted")]
    disc_infra = sum(1 for c in evaluable if (c.get("discovery") or {}).get("infrastructure_incomplete"))
    rows = [final_rows_by_id[c["candidate_id"]] for c in evaluable if c["candidate_id"] in final_rows_by_id]
    valid = [r for r in rows if r.get("validity") == "valid"]
    final_complete = [r for r in valid if isinstance(r.get("target_final"), dict)
                      and not r["target_final"].get("infrastructure_incomplete")]
    gaps = [r for r in rows if r.get("gap")]
    strict_gaps = [r for r in gaps if not r.get("validator_ambiguous_flag")]
    blind = Counter((r.get("teacher_blind_solve") or {}).get("status", "not_run") for r in gaps)
    meter = summary.get("teacher_meter") or {}
    calls = summary.get("teacher_calls") or []
    disc_queries = summary["target_queries"]["used"]
    task_text = {c["candidate_id"]: shingles(c.get("description_head") or "") for c in cands}
    families_gap = family_clusters(task_text[r["candidate_id"]] for r in gaps)
    families_valid = family_clusters(task_text[r["candidate_id"]] for r in valid)
    ratio = lambda a, b: round(a / b, 4) if b else None
    pair_block = {}
    if (arm_dir / "pairs.jsonl").exists():
        from .pairs import CONTROL_PASS_MIN
        pairs = _load_jsonl(arm_dir / "pairs.jsonl")
        row = lambda cid: final_rows_by_id.get(cid, {})
        done = lambda r: isinstance(r.get("target_final"), dict) and not r["target_final"].get("infrastructure_incomplete")
        valid_pairs = [q for q in pairs if row(q["control_id"]).get("validity") == "valid"
                       and row(q["challenge_id"]).get("validity") == "valid"]
        scored = [q for q in valid_pairs if done(row(q["control_id"])) and done(row(q["challenge_id"]))]
        frontier = [q for q in scored if row(q["control_id"])["target_final"]["n_pass"] >= CONTROL_PASS_MIN
                    and row(q["challenge_id"]).get("gap")]
        pair_block = {
            "pairs_attempted": len(pairs),
            "pairs_locally_valid": sum(1 for q in pairs if (q.get("local") or {}).get("valid")),
            "pair_local_failure_reasons": dict(Counter(r for q in pairs for r in (q.get("local") or {}).get("reasons", []))),
            "pairs_independently_valid": len(valid_pairs),
            "pairs_final_scored": len(scored),
            "control_final_mean_pass_rate": round(statistics.mean(row(q["control_id"])["target_final"]["n_pass"] / 4
                                                                  for q in scored), 4) if scored else None,
            "controls_passing_threshold": sum(1 for q in scored if row(q["control_id"])["target_final"]["n_pass"] >= CONTROL_PASS_MIN),
            "challenge_only_gaps": sum(1 for q in valid_pairs if row(q["challenge_id"]).get("gap")),
            "frontier_gaps": len(frontier),
            "frontier_gap_yield_per_pair": ratio(len(frontier), len(pairs)),
            "frontier_gap_ids": [q["pair_id"] for q in frontier],
        }
    return {**pair_block,
        "arm_id": summary["arm_id"], "kind": summary["kind"], "stop_reason": summary["stop_reason"],
        "teacher_calls": len(calls),
        "teacher_call_outcomes": dict(Counter(c["outcome"] for c in calls)),
        "teacher_parse_failures": sum(1 for c in calls if c.get("parse_error")),
        "attempted_candidates": n,
        "schema_invalid": schema_invalid,
        "locally_invalid_total": local_invalid,
        "duplicates_among_locally_valid": duplicates,
        "unique_locally_valid": len(evaluable),
        "discovery_target_attempted": len(disc),
        "discovery_infrastructure_incomplete": disc_infra,
        "discovery_all_fail": sum(1 for d in disc if d["n_pass_samples"] == 0),
        "discovery_mean_pass_rate": round(statistics.mean(d["n_pass_samples"] / len(d["samples"]) for d in disc), 4) if disc else None,
        "final_evaluated": len(rows),
        "independently_valid": len(valid),
        "invalid_disagreement": sum(1 for r in rows if r.get("validity") == "invalid_disagreement"),
        "validator_incomplete": sum(1 for r in rows if r.get("validity") == "validator_incomplete"),
        "validator_flagged_ambiguous_among_valid": sum(1 for r in valid if r.get("validator_ambiguous_flag")),
        "final_target_complete": len(final_complete),
        "final_target_infrastructure_incomplete": len(valid) - len(final_complete),
        "final_target_mean_pass_rate": round(statistics.mean(r["target_final"]["n_pass"] / r["target_final"]["samples"]
                                                            for r in final_complete), 4) if final_complete else None,
        "confirmed_gaps": len(gaps),
        "confirmed_gaps_strict_no_ambiguity_flag": len(strict_gaps),
        "gap_teacher_blind_solve": dict(blind),
        "gap_family_clusters": families_gap,
        "valid_family_clusters": families_valid,
        "gap_yield_per_candidate": ratio(len(gaps), n),
        "gap_yield_per_discovery_target_query": ratio(len(gaps), disc_queries),
        "valid_task_rate": ratio(len(valid), n),
        "conditional_target_failure_rate": ratio(len(gaps), len(final_complete)),
        "discovery_target_queries": disc_queries,
        "teacher_input_tokens": meter.get("input_tokens"),
        "teacher_output_tokens": meter.get("output_tokens"),
        "teacher_reasoning_tokens": meter.get("reasoning_tokens"),
        "teacher_estimated_usd": meter.get("estimated_usd"),
        "teacher_latency_seconds_total": round(sum(c.get("latency_seconds") or 0 for c in calls), 1),
        "arm_wall_seconds": summary["clock"]["elapsed_seconds"],
    }


def paired_summary(per_arm, metric="confirmed_gaps", a="adaptive", b="batch"):
    by_rep = {}
    for row in per_arm:
        rep = row["arm_id"].split("-")[0]
        by_rep.setdefault(rep, {})[row["kind"]] = row.get(metric)
    pairs = [(rep, v[a], v[b]) for rep, v in sorted(by_rep.items()) if a in v and b in v
             and v[a] is not None and v[b] is not None]
    diffs = [x - y for _, x, y in pairs]
    return {"metric": metric, "contrast": f"{a} - {b}",
            "pairs": [{"rep": r, a: x, b: y, "difference": x - y} for r, x, y in pairs],
            "mean_difference": round(statistics.mean(diffs), 4) if diffs else None,
            "sd_difference": round(statistics.stdev(diffs), 4) if len(diffs) > 1 else None,
            "n_pairs": len(diffs),
            "note": "exploratory pilot; no significance test; units are independent arm runs"}


CONTRASTS = [("adaptive", "batch"), ("adaptive", "adaptive_validity"), ("adaptive_validity", "batch")]


def build_report(arm_dirs, final_dirs, out_path):
    final_rows = {}
    final_meta = []
    for directory in final_dirs:
        for row in _load_jsonl(Path(directory) / "final_results.jsonl"):
            final_rows[row["candidate_id"]] = row
        final_meta.append(json.loads((Path(directory) / "final_meta.json").read_text()))
    per_arm = [arm_metrics(d, final_rows) for d in arm_dirs]
    report = {"per_arm": per_arm,
              "paired": [paired_summary(per_arm, m) for m in
                         ("frontier_gaps", "challenge_only_gaps", "confirmed_gaps", "confirmed_gaps_strict_no_ambiguity_flag",
                          "independently_valid", "gap_family_clusters", "teacher_estimated_usd")],
              "contrasts": {f"{a}-{b}": paired_summary(per_arm, "frontier_gaps", a, b) for a, b in CONTRASTS},
              "validator_usd_total": sum((m.get("validator_meter") or {}).get("estimated_usd", 0) for m in final_meta),
              "teacher_blind_usd_total": sum((m.get("teacher_blind_meter") or {}).get("estimated_usd", 0) or 0
                                             for m in final_meta)}
    Path(out_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def markdown_table(per_arm):
    cols = ["arm_id", "attempted_candidates", "locally_invalid_total", "duplicates_among_locally_valid",
            "unique_locally_valid", "independently_valid", "validator_incomplete", "final_target_mean_pass_rate",
            "confirmed_gaps", "confirmed_gaps_strict_no_ambiguity_flag", "gap_family_clusters",
            "teacher_estimated_usd"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in per_arm:
        lines.append("| " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in cols) + " |")
    return "\n".join(lines)
