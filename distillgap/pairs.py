"""v4 minimal-pair arms: the generator submits (control, challenge) pairs.

A FRONTIER GAP requires: both tasks locally valid, same entry point and signature, description
3-shingle Jaccard >= PAIR_SIMILARITY_MIN (a minimal edit), references that behave differently on at
least one input, a non-duplicate challenge, both tasks independently valid in the final evaluation,
the target passing the control on >= CONTROL_PASS_MIN of 4 fresh final samples and failing the
challenge on all 4. Pair-level failures exclude both tasks from target attempts and final evaluation.
The candidate allowance counts pairs; target queries count every sample of both tasks.
"""
from __future__ import annotations

import hashlib
import json
import time

from . import templates
from .budget import BudgetStop
from .canonical import jaccard, shingles
from .controllers import Arm, versions
from .prompts import extract_json_object, pair_prompts_sha256, pair_round_message, pair_teacher_system
from . import sandbox
from .sandbox import values_equal
from .verifier import local_check

PAIR_VERSION = "distillgap_pairs_v2_targets_validity"
PAIR_SIMILARITY_MIN = 0.4
CONTROL_PASS_MIN = 2
PAIR_FEEDBACK_FIELDS_VALIDITY = ["pair_id", "round", "pair_family", "changed_requirement", "entry_point",
                                  "control_description_head", "challenge_description_head", "operator_status",
                                  "pair_check_detail (last round)"]
PAIR_FEEDBACK_FIELDS = {
    "full": ["pair_id", "round", "pair_family", "changed_requirement", "entry_point",
             "control_description_head", "challenge_description_head", "operator_status",
             "pair_check_detail (last round)", "control_target.{samples,passed,outcomes}",
             "challenge_target.{samples,passed,outcomes}",
             "<role>_target.example_failure.{body<=1200,input,expected,got} (last round)"],
    "none": ["pair_id", "round", "pair_family", "changed_requirement", "entry_point",
             "control_description_head", "challenge_description_head"],
    "validity": PAIR_FEEDBACK_FIELDS_VALIDITY,
}


def behaviour_identical(task_a, task_b) -> bool:
    inputs = (task_a.hidden_inputs + task_b.hidden_inputs)[:40]
    ra = sandbox.run_function(task_a.reference_solution, task_a.entry_point, inputs, per_call_seconds=2.0)
    rb = sandbox.run_function(task_b.reference_solution, task_b.entry_point, inputs, per_call_seconds=2.0)
    compared = 0
    for ca, cb in zip(ra.calls, rb.calls):
        if ca and cb and ca.get("ok") and cb.get("ok"):
            compared += 1
            if not values_equal(ca["value"], cb["value"]):
                return False
    return True  # no input distinguishes them (or none could be compared)


def example_failure(record) -> dict | None:
    disc = record.get("discovery") or {}
    failing = next((s for s in disc.get("samples", []) if not s["passed"]), None)
    if not failing:
        return None
    fail = failing.get("first_fail") or {}
    index = fail.get("index")
    expected = record["expected_hidden"][index] if index is not None else None
    return {"body": failing.get("body", "")[:1200], "input": fail.get("input"),
            "expected": json.dumps(expected)[:300] if index is not None else None,
            "got": str(fail.get("got"))[:300] if fail else failing.get("outcome")}


class PairArm(Arm):
    def __init__(self, cfg, out_dir, target, broker=None):
        super().__init__(cfg, out_dir, target, broker)
        self.pairs: list[dict] = []
        self._ids: dict[str, dict] = {}

    def _by_id(self, cid):
        if cid not in self._ids:
            self._ids = {c["candidate_id"]: c for c in self.candidates}
        return self._ids[cid]

    def _commit_pair(self, raw_pair, round_index, index, call_id):
        raw_pair = raw_pair if isinstance(raw_pair, dict) else {"_not_object": raw_pair}
        pair_id = f"{self.cfg.arm_id}-p{len(self.pairs) + 1:03d}"
        records = {}
        for role in ("control", "challenge"):
            raw = raw_pair.get(role)
            record = self._new_candidate(raw if isinstance(raw, dict) else {"_not_object": raw}, round_index, index, call_id)
            record.update(pair_id=pair_id, role=role)
            records[role] = record
        pair = {"pair_id": pair_id, "round": round_index, "control_id": records["control"]["candidate_id"],
                "challenge_id": records["challenge"]["candidate_id"],
                "pair_family": str(raw_pair.get("pair_family", ""))[:80],
                "changed_requirement": str(raw_pair.get("changed_requirement", ""))[:300],
                "derived_from": [str(x)[:40] for x in raw_pair.get("derived_from", [])][:8]
                if isinstance(raw_pair.get("derived_from"), list) else [],
                "rationale": str(raw_pair.get("rationale", ""))[:2000],
                "raw_pair_blob": self.log.put_blob(raw_pair), "local": None}
        self.pairs.append(pair)
        self.log.append("pair_committed", pair_id=pair_id, control_id=pair["control_id"],
                        challenge_id=pair["challenge_id"], raw_pair_blob=pair["raw_pair_blob"])
        return [records["control"], records["challenge"]]

    def _teacher_round(self, round_index):
        remaining = self.candidates_meter.remaining()
        request_n = min(self.cfg.per_call_tasks, remaining)
        history = self._pair_history(self.cfg.feedback_mode, round_index - 1) if round_index else []
        message = pair_round_message(round_index=round_index, rounds=self.cfg.max_teacher_calls, per_round=request_n,
                                     pairs_used=self.candidates_meter.used, pairs_max=self.cfg.max_candidates,
                                     history=history, feedback_mode=self.cfg.feedback_mode)
        result = self.broker.call("teacher_generate_pairs",
                                  pair_teacher_system(self.cfg.max_reference_ast_nodes, self.cfg.target_description),
                                  message, arm_id=self.cfg.arm_id, round=round_index)
        call = {"round": round_index, "call_id": result.call_id, "outcome": result.outcome, "usd": result.usd,
                "latency_seconds": result.latency_seconds, "usage": result.usage, "error": result.error,
                "tasks_parsed": 0, "parse_error": None}
        self.teacher_calls.append(call)
        if result.outcome == "budget_stop":
            raise BudgetStop("teacher budget: " + str(result.error))
        records = []
        if result.ok:
            try:
                pairs = extract_json_object(result.text).get("pairs")
                if not isinstance(pairs, list):
                    raise ValueError("no pairs list")
            except ValueError as exc:
                call["parse_error"] = str(exc)
                pairs = []
            call["tasks_parsed"] = len(pairs)
            for index, raw_pair in enumerate(pairs):
                if index >= request_n:
                    self.log.append("excess_pair_not_counted", round=round_index, index=index,
                                    raw_pair_blob=self.log.put_blob(raw_pair))
                    continue
                self.candidates_meter.take(1)
                records.extend(self._commit_pair(raw_pair, round_index, index, result.call_id))
        self.log.append("teacher_round_done", **call)
        return records

    def _local_and_dedup(self, records):
        groups = {}
        for record in records:
            groups.setdefault(record["pair_id"], {})[record["role"]] = record
        for pair_id, roles in groups.items():
            pair = next(p for p in self.pairs if p["pair_id"] == pair_id)
            checks = {}
            for role in ("control", "challenge"):
                record = roles[role]
                task, check = local_check(self.log.get_blob(record["raw_task_blob"]),
                                          max_ast_nodes=self.cfg.max_reference_ast_nodes)
                checks[role] = (task, check)
                record["local"] = {"valid": check.valid, "reasons": check.reasons, "detail": check.detail}
                if check.valid:
                    record["prompt"] = check.prompt
                    record["prompt_sha256"] = hashlib.sha256(check.prompt.encode()).hexdigest()
                    record["expected_hidden"] = check.expected_hidden
                    record["hidden_inputs"] = task.hidden_inputs
            (tc, cc), (th, ch) = checks["control"], checks["challenge"]
            reasons, similarity = [], None
            if not cc.valid:
                reasons.append("control_invalid")
            if not ch.valid:
                reasons.append("challenge_invalid")
            if cc.valid and ch.valid:
                if tc.entry_point != th.entry_point or tc.signature.replace(" ", "") != th.signature.replace(" ", ""):
                    reasons.append("pair_signature_mismatch")
                similarity = round(jaccard(shingles(tc.description, tc.entry_point, tc.params),
                                           shingles(th.description, th.entry_point, th.params)), 4)
                if similarity < PAIR_SIMILARITY_MIN:
                    reasons.append("pair_not_minimal")
                if not reasons and behaviour_identical(tc, th):
                    reasons.append("pair_identical_behavior")
            duplicate, layer = None, None
            if not reasons:
                duplicate, layer = self.dedup.check(roles["challenge"]["candidate_id"], th, ch.prompt)
                if duplicate:
                    reasons.append("challenge_duplicate")
            roles["challenge"]["duplicate_of"], roles["challenge"]["duplicate_layer"] = duplicate, layer
            pair["local"] = {"valid": not reasons, "reasons": reasons, "similarity": similarity}
            if reasons:
                for role in ("control", "challenge"):
                    record = roles[role]
                    if record["local"]["valid"]:
                        record["local"] = {"valid": False, "reasons": ["pair:" + ",".join(reasons)],
                                           "detail": record["local"]["detail"]}
            self.log.append("pair_local_check", pair_id=pair_id, local=pair["local"],
                            control_local=roles["control"]["local"], challenge_local=roles["challenge"]["local"],
                            duplicate_of=duplicate, duplicate_layer=layer)

    def _pair_history(self, mode, last_round):
        rows = []
        for pair in self.pairs:
            control, challenge = self._by_id(pair["control_id"]), self._by_id(pair["challenge_id"])
            row = {"pair_id": pair["pair_id"], "round": pair["round"] + 1, "pair_family": pair["pair_family"],
                   "changed_requirement": pair["changed_requirement"], "entry_point": challenge["entry_point"],
                   "control_description_head": control["description_head"],
                   "challenge_description_head": challenge["description_head"]}
            if mode in ("full", "validity"):
                local = pair["local"] or {}
                row["operator_status"] = ("valid pair (operator execution checks; independent validation happens later)"
                                          if local.get("valid") else "invalid pair: " + ", ".join(local.get("reasons", [])))
                if pair["round"] == last_round and not local.get("valid"):
                    row["pair_check_detail"] = json.dumps({
                        "similarity": local.get("similarity"),
                        "control": (control["local"] or {}).get("reasons"), "control_detail": (control["local"] or {}).get("detail"),
                        "challenge": (challenge["local"] or {}).get("reasons"),
                        "challenge_detail": (challenge["local"] or {}).get("detail")})[:900]
                for role, record in (("control", control), ("challenge", challenge)):
                    if mode == "validity":
                        continue  # validity-only condition: never reveal the target's attempts
                    disc = record.get("discovery")
                    if disc and disc.get("attempted"):
                        result = {"samples": len(disc["samples"]), "passed": disc["n_pass_samples"],
                                  "outcomes": [s["outcome"] for s in disc["samples"]]}
                        if pair["round"] == last_round:
                            failure = example_failure(record)
                            if failure:
                                result["example_failure"] = failure
                        row[f"{role}_target"] = result
                    elif disc:
                        row[f"{role}_target"] = "not attempted: " + str(disc.get("reason"))[:120]
                    else:
                        row[f"{role}_target"] = "not attempted (invalid pair)"
            rows.append(row)
        return rows

    def run(self):
        cfg = self.cfg
        self.log.append("arm_started", config=self._config_json(), pair_version=PAIR_VERSION,
                        pair_similarity_min=PAIR_SIMILARITY_MIN, control_pass_min=CONTROL_PASS_MIN,
                        feedback_fields=PAIR_FEEDBACK_FIELDS[cfg.feedback_mode],
                        versions={**versions(cfg.max_reference_ast_nodes),
                                  "pair_prompts_sha256": pair_prompts_sha256(cfg.max_reference_ast_nodes, cfg.target_description)},
                        teacher_route=getattr(getattr(self.broker, "route", None), "identity", None),
                        target=getattr(self.target, "identity", None))
        try:
            if cfg.kind == "template":
                import random
                rng = random.Random(cfg.seed)
                families = templates.CONTROL
                if cfg.max_reference_ast_nodes is not None:  # same family rule as v3's template control
                    families = [f for f in families if any(
                        templates._ast_nodes(templates.make_task(f, s)["reference_solution"]) <= cfg.max_reference_ast_nodes
                        for s in range(50))]
                for index in range(cfg.max_candidates):
                    family = families[index % len(families)]
                    control = templates.generate([family], 1, rng.randrange(2 ** 31), max_ast_nodes=cfg.max_reference_ast_nodes)[0]
                    challenge = templates.generate([family], 1, rng.randrange(2 ** 31), max_ast_nodes=cfg.max_reference_ast_nodes)[0]
                    self.candidates_meter.take(1)
                    self._commit_pair({"pair_family": control["family"], "changed_requirement": "operator template parameter redraw",
                                       "control": control, "challenge": challenge}, 0, index, None)
                self._local_and_dedup(self.candidates)
                self.clock.check()
                self._attempt_target(self.candidates)
            elif cfg.kind == "batch":
                for round_index in range(cfg.max_teacher_calls):
                    if self.candidates_meter.remaining() <= 0:
                        break
                    self.clock.check(reserve_seconds=300)
                    self._teacher_round(round_index)
                self.log.append("batch_precommitted", n_pairs=len(self.pairs))
                self._local_and_dedup(self.candidates)
                self.clock.check()
                self._attempt_target(self.candidates)
            elif cfg.kind in ("adaptive", "adaptive_validity"):
                for round_index in range(cfg.max_teacher_calls):
                    if self.candidates_meter.remaining() <= 0:
                        break
                    self.clock.check(reserve_seconds=300)
                    records = self._teacher_round(round_index)
                    if records:
                        self._local_and_dedup(records)
                        self._attempt_target(records)
            else:
                raise ValueError(cfg.kind)
            self.stop_reason = "candidate_allowance_or_call_limit"
        except BudgetStop as stop:
            self.stop_reason = "budget_stop: " + str(stop)
            pending = [r for r in self.candidates if r["local"] is None]
            if pending:
                self._local_and_dedup(pending)
                if self.clock.remaining() > 0:
                    self._attempt_target(pending)
        except Exception as exc:  # noqa: BLE001 - a crash must never discard committed candidates
            import traceback
            self.stop_reason = "crash: " + (type(exc).__name__ + ": " + str(exc))[:300]
            self.log.append("arm_crashed", error=self.stop_reason, trace=traceback.format_exc()[-3000:])
        return self.finish()

    def _config_json(self):
        from dataclasses import asdict
        return asdict(self.cfg)

    def finish(self):
        with (self.dir / "pairs.jsonl").open("w") as handle:
            for pair in self.pairs:
                handle.write(json.dumps(pair, sort_keys=True) + "\n")
        self.log.append("pairs_written", n_pairs=len(self.pairs), unix=time.time())
        return super().finish()
