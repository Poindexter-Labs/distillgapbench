"""Independent final evaluation of frozen discovery arms. Nothing here is ever returned
to a generator.

For every locally valid, non-duplicate candidate:
  V. validator (cross-family route) blind-solves from the target-visible prompt only and
     proposes up to 8 extra valid inputs; validity = differential agreement of the
     validator's program with the teacher reference on hidden + extra inputs
     (inputs on which the reference raises are discarded and counted; >= 5 retained).
  T. the frozen target is re-queried in a fresh stateless request with NEW operator seeds;
     samples are scored on the retained union inputs against reference outputs.
  G. confirmed gap = valid AND every final sample fails AND no infrastructure gap.
  B. (secondary) the teacher blind-solves each confirmed gap from the prompt alone, scored
     by the same differential agreement.
Validator outcomes that are not a completed parseable answer are infrastructure-
incomplete (one fresh, non-ambiguous retry is allowed for truncated/unparseable/refused
answers; ambiguous calls are never replayed).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from .budget import CountMeter
from .controllers import target_seed, versions
from .ledger import EvidenceLog, check_frozen, freeze_directory
from .prompts import BLIND_SOLVE_SYSTEM, VALIDATOR_SYSTEM, extract_json_object, validator_message
from .schema import LIMITS
from .target import TargetUnavailable
from .verifier import agreement, score_completion

MAX_EXTRA_INPUTS = 8


def load_arm(directory):
    directory = Path(directory)
    problems = check_frozen(directory)
    if problems:
        raise RuntimeError(f"frozen arm {directory} changed: {problems[:5]}")
    candidates = [json.loads(l) for l in (directory / "candidates.jsonl").read_text().splitlines() if l.strip()]
    summary = json.loads((directory / "arm_summary.json").read_text())
    return summary, candidates


def _clean_extras(extras, n_params):
    kept, rejected = [], 0
    for args in extras if isinstance(extras, list) else []:
        try:
            ok = (isinstance(args, list) and len(args) == n_params
                  and len(json.dumps(args, allow_nan=False).encode()) <= LIMITS["input_json_bytes"])
        except (TypeError, ValueError):
            ok = False
        if ok and len(kept) < MAX_EXTRA_INPUTS:
            kept.append(args)
        else:
            rejected += 1
    return kept, rejected


def _solve(broker, system, prompt, purpose, candidate_id, retries=1):
    attempts = []
    for attempt in range(retries + 1):
        result = broker.call(purpose, system, validator_message(prompt), candidate_id=candidate_id, attempt=attempt)
        record = {"call_id": result.call_id, "outcome": result.outcome, "usd": result.usd,
                  "latency_seconds": result.latency_seconds, "error": result.error}
        attempts.append(record)
        if result.outcome in ("ambiguous", "budget_stop", "rejected"):
            return None, attempts            # never replay ambiguous; budget/rejection is final
        if result.ok:
            try:
                parsed = extract_json_object(result.text)
                if isinstance(parsed.get("solution"), str) and parsed["solution"].strip():
                    return parsed, attempts
                record["parse_error"] = "missing solution"
            except ValueError as exc:
                record["parse_error"] = str(exc)
    return None, attempts


def final_evaluate(arm_dirs, out_dir, *, validator_broker, teacher_broker, target, eval_seed: int,
                   samples: int = 4, target_query_limit_per_arm: int = 128, concurrency: int = 6,
                   blind_solve: bool = True, log=None):
    out = Path(out_dir)
    log = log or EvidenceLog(out)  # brokers must share this same log (single hash chain)
    arms = {}
    for directory in arm_dirs:
        summary, candidates = load_arm(directory)
        arms[summary["arm_id"]] = {"summary": summary, "candidates": candidates, "dir": str(directory)}
    log.append("final_eval_started", arms=list(arms), validator_route=validator_broker.route.identity,
               teacher_route=teacher_broker.route.identity if teacher_broker else None,
               target=getattr(target, "identity", None), eval_seed=eval_seed, versions=versions())
    evaluable = [c for arm in arms.values() for c in arm["candidates"]
                 if c["local"] and c["local"]["valid"] and not c["duplicate_of"]]
    evaluable_ids = {c["candidate_id"] for c in evaluable}

    # V and T run concurrently: target generation depends only on the prompt.
    def validate(candidate):
        parsed, attempts = _solve(validator_broker, VALIDATOR_SYSTEM, candidate["prompt"], "validator_solve",
                                  candidate["candidate_id"])
        return candidate["candidate_id"], parsed, attempts

    def final_target(arm_id):
        meter = CountMeter("final_target_queries", target_query_limit_per_arm)
        items = []
        for c in arms[arm_id]["candidates"]:
            if c["candidate_id"] in evaluable_ids:
                try:
                    meter.take(samples)
                except Exception:  # noqa: BLE001 - budget stop is recorded per candidate
                    continue
                items.append({"item_id": c["candidate_id"], "prompt": c["prompt"], "n": samples,
                              "seed": target_seed(eval_seed, c["candidate_id"], "final")})
        if not items:
            return arm_id, {}, None
        try:
            return arm_id, target.generate(items, phase="final"), None
        except Exception as exc:  # noqa: BLE001 - infrastructure-incomplete, never a target failure
            return arm_id, {}, (type(exc).__name__ + ": " + str(exc))[:400]

    with ThreadPoolExecutor(max(1, len(arms))) as target_pool, ThreadPoolExecutor(concurrency) as pool:
        target_futures = [target_pool.submit(final_target, arm_id) for arm_id in arms]
        validations = {cid: (parsed, attempts) for cid, parsed, attempts in pool.map(validate, evaluable)}
        target_results = {}
        target_errors = {}
        for future in target_futures:
            arm_id, response, error = future.result()
            target_results.update(response)
            if error:
                target_errors[arm_id] = error
    log.append("final_generation_done", target_errors=target_errors, n_validations=len(validations))

    rows = []

    def judge(candidate):
        cid = candidate["candidate_id"]
        parsed, attempts = validations[cid]
        task_blob = json.loads((Path(arms[candidate["arm_id"]]["dir"]) / "blobs" / (candidate["raw_task_blob"] + ".json")).read_text())
        reference = task_blob["reference_solution"]
        n_params = len(candidate["hidden_inputs"][0]) if candidate["hidden_inputs"] else 0
        row = {"candidate_id": cid, "arm_id": candidate["arm_id"], "family": candidate["family"],
               "validator_attempts": attempts}
        if parsed is None:
            row.update(validity="validator_incomplete", gap=False)
        else:
            extras, rejected = _clean_extras(parsed.get("extra_inputs"), n_params)
            union = candidate["hidden_inputs"] + extras
            check = agreement(reference, parsed["solution"], candidate["entry_point"], union)
            row.update(validator_ambiguous_flag=bool(parsed.get("ambiguous")),
                       validator_note=str(parsed.get("ambiguity_note", ""))[:400],
                       extra_inputs=len(extras), extra_inputs_rejected_schema=rejected,
                       agreement={k: v for k, v in check.items() if k != "expected"},
                       validator_solution_blob=log.put_blob(parsed["solution"]))
            if check.get("infrastructure"):
                row.update(validity="validator_incomplete", gap=False)
            elif check["agree"]:
                row["validity"] = "valid"
                retained = [union[i] for i in check["retained_indices"]]
                row["final_inputs"] = len(retained)
                response = target_results.get(cid)
                if response is None:
                    row.update(target_final="infrastructure_incomplete",
                               target_error=target_errors.get(candidate["arm_id"]), gap=False)
                else:
                    scored = [score_completion(candidate["prompt"], completion, candidate["entry_point"],
                                               retained, check["expected"])
                              for completion in response["completions"]]
                    infra = any(s.get("infrastructure") for s in scored)
                    n_pass = sum(s["passed"] for s in scored)
                    row.update(target_final={"samples": len(scored), "n_pass": n_pass,
                                             "outcomes": [s["outcome"] for s in scored],
                                             "seed": response.get("seed"), "infrastructure_incomplete": infra},
                               final_completions_blob=log.put_blob(response["completions"]),
                               gap=(not infra and n_pass == 0 and len(scored) == samples))
                    row["_retained"], row["_expected"], row["_reference"] = retained, check["expected"], reference
            else:
                row.update(validity="invalid_disagreement", gap=False)
        return row

    with ThreadPoolExecutor(concurrency) as pool:
        rows = list(pool.map(judge, evaluable))

    if blind_solve and teacher_broker is not None:
        gaps = [r for r in rows if r.get("gap")]
        by_id = {c["candidate_id"]: c for c in evaluable}

        def blind(row):
            parsed, attempts = _solve(teacher_broker, BLIND_SOLVE_SYSTEM, by_id[row["candidate_id"]]["prompt"],
                                      "teacher_blind_solve", row["candidate_id"])
            if parsed is None:
                return row["candidate_id"], {"status": "incomplete", "attempts": attempts}
            check = agreement(row["_reference"], parsed["solution"], by_id[row["candidate_id"]]["entry_point"],
                              row["_retained"])
            return row["candidate_id"], {"status": "solved" if check["agree"] else "not_solved",
                                         "attempts": attempts, "retained": check.get("retained"),
                                         "mismatch": check.get("mismatch"),
                                         "solution_blob": log.put_blob(parsed["solution"])}

        with ThreadPoolExecutor(concurrency) as pool:
            blinds = dict(pool.map(blind, gaps))
        for row in rows:
            if row["candidate_id"] in blinds:
                row["teacher_blind_solve"] = blinds[row["candidate_id"]]

    for row in rows:
        for key in ("_retained", "_expected", "_reference"):
            row.pop(key, None)
        log.append("final_candidate", **row)
    with (out / "final_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    meta = {"arms": {k: v["dir"] for k, v in arms.items()}, "eval_seed": eval_seed, "samples": samples,
            "validator_meter": validator_broker.meter.snapshot(),
            "teacher_blind_meter": teacher_broker.meter.snapshot() if teacher_broker else None,
            "target_errors": target_errors, "versions": versions(), "finished_unix": time.time()}
    (out / "final_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    log.append("final_eval_finished", n_rows=len(rows))
    freeze_directory(out)
    return rows, meta
