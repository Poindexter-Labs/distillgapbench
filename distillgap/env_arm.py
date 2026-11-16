"""DistillGap v1 pair arm for the non-coding environments (math, code output prediction).

Same protocol as the v5 PairArm (pairs.py): identical round structure, candidate allowance, matched adaptive/batch
conditions, feedback modes, crash and budget handling, frozen evidence. Only the environment-specific steps are
delegated to an adapter (adapters.py): the researcher system prompt, the operator check of each task, the minimal-pair
rule, the duplicate key, the student prompt (with its stop strings and token cap), scoring of a student completion,
the example failure shown to adaptive researchers, and the operator template baseline.
"""
from __future__ import annotations

import hashlib
import json
import random

from .budget import BudgetStop
from .controllers import target_seed, versions
from .pairs import CONTROL_PASS_MIN, PAIR_SIMILARITY_MIN, PairArm
from .prompts import pair_round_message

ENV_ARM_VERSION = "distillgap_v1_env_pair_arm_1"
COMPLETION_KEEP_CHARS = 1500


def adapter_prompts_sha256(adapter, target_description) -> str:
    material = json.dumps([ENV_ARM_VERSION, adapter.name, adapter.teacher_system(target_description),
                           adapter.validator_system() or ""], sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


class EnvPairArm(PairArm):
    def __init__(self, cfg, out_dir, target, broker=None, adapter=None):
        super().__init__(cfg, out_dir, target, broker)
        if adapter is None:
            raise ValueError("EnvPairArm needs an adapter")
        self.adapter = adapter
        self.seen_challenges: dict[str, str] = {}

    # ------------------------------------------------------------------ recording
    def _new_candidate(self, raw, round_index, index, call_id):
        record = super()._new_candidate(raw, round_index, index, call_id)
        head = None
        if isinstance(raw, dict):
            head = "\n".join(str(raw.get(k, "")) for k in self.adapter.feedback_heads if raw.get(k) is not None)[:300] or None
        record["description_head"], record["family"], record["entry_point"] = head, None, None
        record["env"] = self.adapter.name
        return record

    # ------------------------------------------------------------------ researcher rounds
    def _teacher_round(self, round_index):
        remaining = self.candidates_meter.remaining()
        request_n = min(self.cfg.per_call_tasks, remaining)
        history = self._pair_history(self.cfg.feedback_mode, round_index - 1) if round_index else []
        message = pair_round_message(round_index=round_index, rounds=self.cfg.max_teacher_calls, per_round=request_n,
                                     pairs_used=self.candidates_meter.used, pairs_max=self.cfg.max_candidates,
                                     history=history, feedback_mode=self.cfg.feedback_mode)
        from .prompts import extract_json_object
        result = self.broker.call("teacher_generate_pairs", self.adapter.teacher_system(self.cfg.target_description),
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

    # ------------------------------------------------------------------ operator checks
    def _local_and_dedup(self, records):
        groups = {}
        for record in records:
            groups.setdefault(record["pair_id"], {})[record["role"]] = record
        for pair_id, roles in groups.items():
            pair = next(p for p in self.pairs if p["pair_id"] == pair_id)
            checks = {}
            for role in ("control", "challenge"):
                record = roles[role]
                try:
                    inst, check = self.adapter.local_check(self.log.get_blob(record["raw_task_blob"]))
                except Exception as exc:  # noqa: BLE001 - a malformed task is invalid, never a crash
                    from .adapters import Check
                    inst, check = None, Check(False, [f"operator_check_error: {type(exc).__name__}: {str(exc)[:200]}"])
                checks[role] = (inst, check)
                record["local"] = {"valid": check.valid, "reasons": check.reasons, "detail": check.detail}
                if check.valid:
                    record["prompt"] = check.student["prompt"]
                    record["prompt_sha256"] = hashlib.sha256(check.student["prompt"].encode()).hexdigest()
                    record["student_stop"], record["student_max_tokens"] = check.student["stop"], check.student["max_tokens"]
                    record["scoring"] = check.scoring
                    record["instance"] = inst
            (ic, cc), (ih, ch) = checks["control"], checks["challenge"]
            reasons, similarity = [], None
            if not cc.valid:
                reasons.append("control_invalid")
            if not ch.valid:
                reasons.append("challenge_invalid")
            if cc.valid and ch.valid:
                try:
                    pair_reasons, similarity = self.adapter.pair_reasons(ic, cc, ih, ch)
                except Exception as exc:  # noqa: BLE001
                    pair_reasons = [f"pair_check_error: {type(exc).__name__}"]
                reasons.extend(pair_reasons)
            duplicate = None
            if not reasons:
                key = self.adapter.dedup_key(ih, ch)
                duplicate = self.seen_challenges.get(key)
                if duplicate:
                    reasons.append("challenge_duplicate")
                else:
                    self.seen_challenges[key] = roles["challenge"]["candidate_id"]
            roles["challenge"]["duplicate_of"] = duplicate
            roles["challenge"]["duplicate_layer"] = "normalised_text" if duplicate else None
            pair["local"] = {"valid": not reasons, "reasons": reasons, "similarity": similarity}
            if reasons:
                for role in ("control", "challenge"):
                    record = roles[role]
                    if record["local"]["valid"]:
                        record["local"] = {"valid": False, "reasons": ["pair:" + ",".join(reasons)],
                                           "detail": record["local"]["detail"]}
            self.log.append("pair_local_check", pair_id=pair_id, local=pair["local"],
                            control_local=roles["control"]["local"], challenge_local=roles["challenge"]["local"],
                            duplicate_of=duplicate, pair_similarity_min=PAIR_SIMILARITY_MIN)

    # ------------------------------------------------------------------ student attempts
    def _attempt_target(self, records):
        eligible = [r for r in records if r["local"]["valid"] and not r["duplicate_of"]]
        items = []
        for record in eligible:
            try:
                self.target_meter.take(self.cfg.samples_per_candidate)
            except BudgetStop as stop:
                record["discovery"] = {"attempted": False, "reason": "target_query_budget", "detail": str(stop)}
                continue
            items.append({"item_id": record["candidate_id"], "prompt": record["prompt"], "stop": record["student_stop"],
                          "max_tokens": record["student_max_tokens"], "n": self.cfg.samples_per_candidate,
                          "seed": target_seed(self.cfg.seed, record["candidate_id"], "discovery")})
        if not items:
            return
        by_id = {r["candidate_id"]: r for r in records}
        self.log.append("target_request", phase="discovery",
                        items=[{"item_id": i["item_id"], "seed": i["seed"], "n": i["n"]} for i in items])
        try:
            response = self.target.generate(items, phase="discovery")
        except Exception as exc:  # noqa: BLE001 - infrastructure, never a target failure
            for item in items:
                by_id[item["item_id"]]["discovery"] = {"attempted": False, "infrastructure_incomplete": True,
                                                       "reason": str(exc)[:300]}
            self.log.append("target_unavailable", phase="discovery", error=str(exc)[:500])
            return
        for item in items:
            record = by_id[item["item_id"]]
            got = response[item["item_id"]]
            samples = []
            for completion, finish in zip(got["completions"], got["finish"]):
                sample = self.adapter.score(record["scoring"], completion)
                sample.update(finish=finish, completion=str(completion)[:COMPLETION_KEEP_CHARS])
                samples.append(sample)
            record["discovery"] = {"attempted": True, "seed": item["seed"], "samples": samples,
                                   "n_pass_samples": sum(s["passed"] for s in samples),
                                   "infrastructure_incomplete": False, "rid": got.get("rid")}
            self.log.append("target_discovery_scored", candidate_id=record["candidate_id"],
                            completions_blob=self.log.put_blob(got["completions"]),
                            n_pass_samples=record["discovery"]["n_pass_samples"],
                            outcomes=[s["outcome"] for s in samples], infrastructure_incomplete=False)

    # ------------------------------------------------------------------ feedback
    def _pair_history(self, mode, last_round):
        rows = []
        for pair in self.pairs:
            control, challenge = self._by_id(pair["control_id"]), self._by_id(pair["challenge_id"])
            row = {"pair_id": pair["pair_id"], "round": pair["round"] + 1, "pair_family": pair["pair_family"],
                   "changed_requirement": pair["changed_requirement"],
                   "control_head": control["description_head"], "challenge_head": challenge["description_head"]}
            if mode in ("full", "validity"):
                local = pair["local"] or {}
                row["operator_status"] = ("valid pair (operator execution checks; independent validation happens later)"
                                          if local.get("valid") else "invalid pair: " + ", ".join(local.get("reasons", [])))
                if pair["round"] == last_round and not local.get("valid"):
                    row["pair_check_detail"] = json.dumps({
                        "similarity": local.get("similarity"),
                        "control": (control["local"] or {}).get("reasons"), "control_detail": (control["local"] or {}).get("detail"),
                        "challenge": (challenge["local"] or {}).get("reasons"),
                        "challenge_detail": (challenge["local"] or {}).get("detail")}, default=str)[:900]
                for role, record in (("control", control), ("challenge", challenge)):
                    if mode == "validity":
                        continue  # validity-only condition: never reveal the target's attempts
                    disc = record.get("discovery")
                    if disc and disc.get("attempted"):
                        result = {"samples": len(disc["samples"]), "passed": disc["n_pass_samples"],
                                  "outcomes": [s["outcome"] for s in disc["samples"]]}
                        if pair["round"] == last_round:
                            failing = next((s for s in disc["samples"] if not s["passed"]), None)
                            if failing:
                                result["example_failure"] = self.adapter.example_failure(
                                    record["scoring"], failing, failing.get("completion", ""))
                        row[f"{role}_target"] = result
                    elif disc:
                        row[f"{role}_target"] = "not attempted: " + str(disc.get("reason"))[:120]
                    else:
                        row[f"{role}_target"] = "not attempted (invalid pair)"
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ run
    def run(self):
        cfg = self.cfg
        self.log.append("arm_started", config=self._config_json(), env=self.adapter.name, env_arm_version=ENV_ARM_VERSION,
                        control_pass_min=CONTROL_PASS_MIN,
                        versions={**versions(cfg.max_reference_ast_nodes),
                                  "env_prompts_sha256": adapter_prompts_sha256(self.adapter, cfg.target_description)},
                        teacher_route=getattr(getattr(self.broker, "route", None), "identity", None),
                        target=getattr(self.target, "identity", None))
        try:
            if cfg.kind == "template":
                rng = random.Random(cfg.seed)
                for index in range(cfg.max_candidates):
                    raw_pair = self.adapter.template_pair(rng, index)
                    self.candidates_meter.take(1)
                    self._commit_pair(raw_pair, 0, index, None)
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
