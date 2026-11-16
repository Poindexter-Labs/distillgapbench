"""DistillGap v1 final evaluation for math and code output prediction, and the light-path judge stage for every
environment. Nothing here is ever returned to a researcher.

Math / output (frozen arms, fresh operator seeds):
  T. every locally valid, non-duplicate task is re-sampled from the frozen student (4 fresh samples) and scored
     against the operator-checked answer (math: the stated answer reproduced by the researcher's solution program;
     output: the repr produced by operator execution).
  V. math only: an independent cross-family validator blind-solves BOTH problems of every pair that would otherwise
     be a frontier gap (control >= 2/4, challenge 0/4); a task is valid only if the validator's exact answer equals
     the operator-checked answer. Output prediction needs no validator (operator execution is the ground truth).
     Validating only would-be frontier pairs leaves the frontier-gap count unchanged (a pair needs both tasks valid)
     and is the preregistered cost rule; validity rates are therefore reported among would-be gaps.
  G. rows follow the v5 final_results format (validity, target_final, gap) so report.arm_metrics applies unchanged.

Judge (coding and math; layer 3): the Poindexter light path reviews both tasks of every frontier gap
(judge.LightPathJudge). A veto on either task removes the pair from the judged frontier-gap count.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from .budget import CountMeter
from .controllers import target_seed, versions
from .final_eval import load_arm
from .ledger import EvidenceLog, freeze_directory
from .pairs import CONTROL_PASS_MIN
from .prompts import extract_json_object

ENV_FINAL_VERSION = "distillgap_v1_env_final_1"


def _load_pairs(arm_dir):
    path = Path(arm_dir) / "pairs.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def _validate(broker, adapter, inst, candidate_id, retries=1):
    attempts = []
    for attempt in range(retries + 1):
        result = broker.call("validator_solve", adapter.validator_system(), adapter.validator_message(inst),
                             candidate_id=candidate_id, attempt=attempt)
        record = {"call_id": result.call_id, "outcome": result.outcome, "usd": result.usd,
                  "latency_seconds": result.latency_seconds, "error": result.error}
        attempts.append(record)
        if result.outcome in ("ambiguous", "budget_stop", "rejected"):
            return None, attempts
        if result.ok:
            try:
                parsed = extract_json_object(result.text)
                if str(parsed.get("answer", "")).strip():
                    return parsed, attempts
                record["parse_error"] = "missing answer"
            except ValueError as exc:
                record["parse_error"] = str(exc)
    return None, attempts


def env_final_evaluate(arm_dirs, out_dir, *, adapter, target, eval_seed: int, validator_broker=None,
                       samples: int = 4, target_query_limit_per_arm: int = 128, concurrency: int = 12, log=None):
    out = Path(out_dir)
    log = log or EvidenceLog(out)
    arms = {}
    for directory in arm_dirs:
        summary, candidates = load_arm(directory)
        arms[summary["arm_id"]] = {"summary": summary, "candidates": candidates, "pairs": _load_pairs(directory),
                                   "dir": str(directory)}
    log.append("final_eval_started", env=adapter.name, version=ENV_FINAL_VERSION, arms=list(arms),
               validator_route=validator_broker.route.identity if validator_broker else None,
               target=getattr(target, "identity", None), eval_seed=eval_seed, versions=versions())
    by_id = {c["candidate_id"]: c for arm in arms.values() for c in arm["candidates"]}
    evaluable = [c for c in by_id.values() if c["local"] and c["local"]["valid"] and not c["duplicate_of"]]
    evaluable_ids = {c["candidate_id"] for c in evaluable}

    # T: fresh final samples, one generate call per arm (target query cap per arm, as in v5).
    def final_target(arm_id):
        meter = CountMeter("final_target_queries", target_query_limit_per_arm)
        items = []
        for c in arms[arm_id]["candidates"]:
            if c["candidate_id"] in evaluable_ids:
                try:
                    meter.take(samples)
                except Exception:  # noqa: BLE001
                    continue
                items.append({"item_id": c["candidate_id"], "prompt": c["prompt"], "stop": c["student_stop"],
                              "max_tokens": c["student_max_tokens"], "n": samples,
                              "seed": target_seed(eval_seed, c["candidate_id"], "final")})
        if not items:
            return arm_id, {}, None
        try:
            return arm_id, target.generate(items, phase="final"), None
        except Exception as exc:  # noqa: BLE001 - infrastructure-incomplete, never a target failure
            return arm_id, {}, (type(exc).__name__ + ": " + str(exc))[:400]

    responses, target_errors = {}, {}
    with ThreadPoolExecutor(max(1, len(arms))) as pool:
        for arm_id, response, error in pool.map(final_target, list(arms)):
            responses.update(response)
            if error:
                target_errors[arm_id] = error
    rows = {}
    for c in evaluable:
        cid = c["candidate_id"]
        row = {"candidate_id": cid, "arm_id": c["arm_id"], "env": adapter.name, "family": None,
               "validity": "valid_operator_execution" if adapter.validator_system() is None else "not_validated"}
        response = responses.get(cid)
        if response is None:
            row.update(target_final="infrastructure_incomplete", target_error=target_errors.get(c["arm_id"]), gap=False)
        else:
            scored = [adapter.score(c["scoring"], completion) for completion in response["completions"]]
            n_pass = sum(s["passed"] for s in scored)
            row.update(target_final={"samples": len(scored), "n_pass": n_pass, "outcomes": [s["outcome"] for s in scored],
                                     "got": [s.get("got") for s in scored], "finish": response.get("finish"),
                                     "seed": response.get("seed"), "infrastructure_incomplete": False},
                       final_completions_blob=log.put_blob(response["completions"]),
                       _all_fail=(n_pass == 0 and len(scored) == samples))
        rows[cid] = row

    # Would-be frontier pairs: both tasks scored, control >= CONTROL_PASS_MIN, challenge all fail.
    def scored_ok(row):
        return row is not None and isinstance(row.get("target_final"), dict)
    candidates_for_validation = []
    for arm in arms.values():
        for pair in arm["pairs"]:
            rc, rh = rows.get(pair["control_id"]), rows.get(pair["challenge_id"])
            if scored_ok(rc) and scored_ok(rh) and rc["target_final"]["n_pass"] >= CONTROL_PASS_MIN and rh["_all_fail"]:
                candidates_for_validation += [pair["control_id"], pair["challenge_id"]]

    if adapter.validator_system() is not None:
        if validator_broker is None:
            raise ValueError(f"{adapter.name} needs a validator broker")

        def validate(cid):
            parsed, attempts = _validate(validator_broker, adapter, by_id[cid]["instance"], cid)
            return cid, parsed, attempts

        with ThreadPoolExecutor(concurrency) as pool:
            for cid, parsed, attempts in pool.map(validate, candidates_for_validation):
                row = rows[cid]
                row["validator_attempts"] = attempts
                if parsed is None:
                    row["validity"] = "validator_incomplete"
                    continue
                verdict = adapter.validator_agrees(by_id[cid]["scoring"], parsed)
                row.update(validator_answer=verdict["validator_answer"],
                           validator_ambiguous_flag=bool(parsed.get("ambiguous")),
                           validator_note=str(parsed.get("ambiguity_note", ""))[:400],
                           validity="valid" if verdict["agree"] else "invalid_disagreement")
    for row in rows.values():
        if row["validity"] == "valid_operator_execution":
            row["validity"] = "valid"
        all_fail = row.pop("_all_fail", False)
        row["gap"] = bool(row["validity"] == "valid" and all_fail)
        log.append("final_candidate", **row)
    ordered = [rows[c["candidate_id"]] for c in evaluable]
    with (out / "final_results.jsonl").open("w") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    meta = {"env": adapter.name, "version": ENV_FINAL_VERSION, "arms": {k: v["dir"] for k, v in arms.items()},
            "eval_seed": eval_seed, "samples": samples, "validated_candidates": len(candidates_for_validation),
            "validator_meter": validator_broker.meter.snapshot() if validator_broker else None,
            "teacher_blind_meter": None, "target_errors": target_errors, "versions": versions(),
            "finished_unix": time.time()}
    (out / "final_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    log.append("final_eval_finished", n_rows=len(ordered))
    freeze_directory(out)
    return ordered, meta


# ------------------------------------------------------------------------------------------------ judge stage
def frontier_pairs(arm_dir, final_rows_by_id):
    out = []
    for pair in _load_pairs(arm_dir):
        rc, rh = final_rows_by_id.get(pair["control_id"], {}), final_rows_by_id.get(pair["challenge_id"], {})
        done = lambda r: isinstance(r.get("target_final"), dict) and not r["target_final"].get("infrastructure_incomplete")
        if (rc.get("validity") == "valid" and rh.get("validity") == "valid" and done(rc) and done(rh)
                and rc["target_final"]["n_pass"] >= CONTROL_PASS_MIN and rh.get("gap")):
            out.append(pair)
    return out


def judge_frontier(arm_dirs, final_dir, out_dir, *, env, judge, adapter=None, concurrency=8):
    """Run the light-path judge on both tasks of every frontier gap. Writes judge_results.jsonl (one row per pair)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_rows = {}
    for line in (Path(final_dir) / "final_results.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            final_rows[row["candidate_id"]] = row
    jobs = []
    for arm_dir in arm_dirs:
        arm_dir = Path(arm_dir)
        candidates = {c["candidate_id"]: c for c in (json.loads(l) for l in (arm_dir / "candidates.jsonl").read_text().splitlines() if l.strip())}
        for pair in frontier_pairs(arm_dir, final_rows):
            for role in ("control", "challenge"):
                cand = candidates[pair[f"{role}_id"]]
                raw = json.loads((arm_dir / "blobs" / (cand["raw_task_blob"] + ".json")).read_text())
                if env == "coding":
                    problem, solution, variant = cand["prompt"], raw.get("reference_solution", ""), "generic"
                else:
                    problem, solution, variant = adapter.judge_inputs(raw)
                jobs.append((pair["pair_id"], cand["arm_id"], role, cand["candidate_id"], problem, solution, variant))

    def run(job):
        pair_id, arm_id, role, cid, problem, solution, variant = job
        return pair_id, arm_id, role, judge.review(problem, solution, variant, cid)

    results = {}
    with ThreadPoolExecutor(concurrency) as pool:
        for pair_id, arm_id, role, review in pool.map(run, jobs):
            results.setdefault(pair_id, {"pair_id": pair_id, "arm_id": arm_id})[role] = review
    rows = []
    for pair_id, row in sorted(results.items()):
        row["veto"] = any((row.get(role) or {}).get("veto") for role in ("control", "challenge"))
        row["incomplete"] = any((row.get(role) or {}).get("status") != "reviewed" for role in ("control", "challenge"))
        rows.append(row)
    with (out / "judge_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows
