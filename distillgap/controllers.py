"""Discovery-phase arm controllers: adaptive (feedback), batch (precommitted) and template.

Matched across adaptive/batch: same teacher route and system prompt, identical first
user message, same per-call task request, candidate allowance, teacher call/token/$
caps, target-query allowance, samples per candidate, verifier and wall clock. The only
difference is the history block in later rounds (feedback_mode full vs none) and, for
batch, that every teacher call completes before any target query.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time

from . import templates
from .budget import BudgetStop, Clock, CountMeter, ModelLimits
from .canonical import DEDUP_VERSION, Deduper
from .ledger import EvidenceLog, freeze_directory
from .prompts import PROMPTS_VERSION, extract_json_object, prompts_sha256, teacher_round_message, teacher_system
from .schema import PROMPT_TEMPLATE_VERSION, SCHEMA_VERSION, sha256_json
from .target import TargetUnavailable
from .verifier import VERIFIER_VERSION, local_check, score_completion

FEEDBACK_FIELDS = {
    "full": ["candidate_id", "round", "family", "entry_point", "description_head", "operator_status",
             "reference_error (last round)", "target.samples", "target.passed", "target.outcomes",
             "target.example_failure.body (<=1200 chars, last round)", "target.example_failure.input",
             "target.example_failure.expected", "target.example_failure.got"],
    "none": ["candidate_id", "round", "family", "entry_point", "description_head"],
    "validity": ["candidate_id", "round", "family", "entry_point", "description_head", "operator_status",
                 "reference_error (last round)"],
}


@dataclass
class ArmConfig:
    arm_id: str
    kind: str                      # adaptive | batch | template
    seed: int
    max_candidates: int = 32
    per_call_tasks: int = 4
    max_teacher_calls: int = 16
    samples_per_candidate: int = 4
    target_query_limit: int = 128
    wall_seconds: float = 7200.0
    max_reference_ast_nodes: int | None = None
    pair_mode: bool = False  # v4: candidates are (control, challenge) minimal pairs; see pairs.py
    target_description: str | None = None  # v5: target paragraph for the teacher prompt (from the target profile)
    teacher_limits: dict = field(default_factory=lambda: asdict(ModelLimits(
        max_calls=16, max_input_tokens=1_500_000, max_output_tokens=900_000,
        max_output_per_call=64_000, max_usd=80.0)))

    @property
    def feedback_mode(self):
        return {"adaptive": "full", "adaptive_validity": "validity", "batch": "none", "template": "none"}[self.kind]


def target_seed(arm_seed: int, candidate_id: str, phase: str) -> int:
    return int(hashlib.sha256(f"{arm_seed}:{candidate_id}:{phase}".encode()).hexdigest()[:8], 16)


class Arm:
    def __init__(self, cfg: ArmConfig, out_dir, target, broker=None):
        self.cfg, self.target, self.broker = cfg, target, broker
        self.dir = Path(out_dir)
        self.log = EvidenceLog(self.dir)
        self.clock = Clock(cfg.wall_seconds)
        self.candidates_meter = CountMeter("candidates", cfg.max_candidates)
        self.target_meter = CountMeter("target_queries", cfg.target_query_limit)
        self.dedup = Deduper()
        self.candidates: list[dict] = []
        self.teacher_calls: list[dict] = []
        self.stop_reason = None

    # ------------------------------------------------------------------ recording
    def _new_candidate(self, raw, round_index, index, call_id):
        cid = f"{self.cfg.arm_id}-c{len(self.candidates) + 1:03d}"
        record = {"candidate_id": cid, "arm_id": self.cfg.arm_id, "kind": self.cfg.kind, "round": round_index,
                  "index_in_round": index, "teacher_call_id": call_id, "raw_task_sha256": sha256_json(raw),
                  "raw_task_blob": self.log.put_blob(raw), "created_unix": time.time(),
                  "derived_from": raw.get("derived_from", []) if isinstance(raw, dict) else [],
                  "family": raw.get("family") if isinstance(raw, dict) else None,
                  "entry_point": raw.get("entry_point") if isinstance(raw, dict) else None,
                  "description_head": (raw.get("description") or "")[:200] if isinstance(raw, dict) and isinstance(raw.get("description"), str) else None,
                  "local": None, "duplicate_of": None, "duplicate_layer": None, "discovery": None}
        self.candidates.append(record)
        self.log.append("candidate_committed", candidate_id=cid, round=round_index,
                        raw_task_sha256=record["raw_task_sha256"], raw_task_blob=record["raw_task_blob"])
        return record

    def _local_and_dedup(self, records):
        for record in records:
            raw = self.log.get_blob(record["raw_task_blob"])
            task, check = local_check(raw, max_ast_nodes=self.cfg.max_reference_ast_nodes)
            record["local"] = {"valid": check.valid, "reasons": check.reasons, "detail": check.detail}
            if check.valid:
                record["prompt"] = check.prompt
                record["prompt_sha256"] = hashlib.sha256(check.prompt.encode()).hexdigest()
                record["expected_hidden"] = check.expected_hidden
                record["hidden_inputs"] = task.hidden_inputs
                dup, layer = self.dedup.check(record["candidate_id"], task, check.prompt)
                record["duplicate_of"], record["duplicate_layer"] = dup, layer
            self.log.append("candidate_local_check", candidate_id=record["candidate_id"], local=record["local"],
                            duplicate_of=record["duplicate_of"], duplicate_layer=record["duplicate_layer"],
                            prompt_sha256=record.get("prompt_sha256"))

    def _attempt_target(self, records):
        eligible = [r for r in records if r["local"]["valid"] and not r["duplicate_of"]]
        items = []
        for record in eligible:
            try:
                self.target_meter.take(self.cfg.samples_per_candidate)
            except BudgetStop as stop:
                record["discovery"] = {"attempted": False, "reason": "target_query_budget", "detail": str(stop)}
                continue
            seed = target_seed(self.cfg.seed, record["candidate_id"], "discovery")
            items.append({"item_id": record["candidate_id"], "prompt": record["prompt"],
                          "n": self.cfg.samples_per_candidate, "seed": seed})
        if not items:
            return
        by_id = {r["candidate_id"]: r for r in records}
        self.log.append("target_request", phase="discovery", items=[{"item_id": i["item_id"], "seed": i["seed"],
                                                                     "n": i["n"]} for i in items])
        try:
            response = self.target.generate(items, phase="discovery")
        except Exception as exc:  # noqa: BLE001 - TargetUnavailable or transport: infrastructure, not failure
            for item in items:
                by_id[item["item_id"]]["discovery"] = {"attempted": False, "infrastructure_incomplete": True,
                                                       "reason": str(exc)[:300]}
            self.log.append("target_unavailable", phase="discovery", error=str(exc)[:500])
            return
        jobs = []
        for item in items:
            record = by_id[item["item_id"]]
            for completion in response[item["item_id"]]["completions"]:
                jobs.append((record, completion))
        with ThreadPoolExecutor(6) as pool:
            scores = list(pool.map(lambda job: score_completion(job[0]["prompt"], job[1], job[0]["entry_point"],
                                                               job[0]["hidden_inputs"], job[0]["expected_hidden"]), jobs))
        for item in items:
            record = by_id[item["item_id"]]
            got = response[item["item_id"]]
            samples = [s for (r, _), s in zip(jobs, scores) if r is record]
            for sample, tokens, finish in zip(samples, got["new_tokens"], got["finish"]):
                sample.update(new_tokens=tokens, finish=finish)
            infra = any(s.get("infrastructure") for s in samples)
            record["discovery"] = {"attempted": True, "seed": item["seed"], "samples": samples,
                                   "n_pass_samples": sum(s["passed"] for s in samples),
                                   "infrastructure_incomplete": infra, "rid": got.get("rid")}
            self.log.append("target_discovery_scored", candidate_id=record["candidate_id"],
                            completions_blob=self.log.put_blob(got["completions"]),
                            n_pass_samples=record["discovery"]["n_pass_samples"],
                            outcomes=[s["outcome"] for s in samples], infrastructure_incomplete=infra)

    # ------------------------------------------------------------------ feedback
    def _history(self, mode, last_round):
        rows = []
        for record in self.candidates:
            row = {"candidate_id": record["candidate_id"], "round": record["round"] + 1,
                   "family": record["family"], "entry_point": record["entry_point"],
                   "description_head": record["description_head"]}
            if mode in ("full", "validity"):
                local = record["local"] or {}
                if not local.get("valid"):
                    row["operator_status"] = "invalid: " + ", ".join(local.get("reasons", []))
                    if record["round"] == last_round and local.get("detail"):
                        row["reference_error"] = json.dumps(local["detail"])[:600]
                elif record["duplicate_of"]:
                    row["operator_status"] = f"duplicate of {record['duplicate_of']} ({record['duplicate_layer']})"
                else:
                    row["operator_status"] = "valid (operator execution checks; independent validation happens later)"
                disc = record["discovery"]
                if mode == "validity":
                    disc = None  # validity-only condition: never reveal the target's attempts
                    rows.append(row)
                    continue
                if disc and disc.get("attempted"):
                    target = {"samples": len(disc["samples"]), "passed": disc["n_pass_samples"],
                              "outcomes": [s["outcome"] for s in disc["samples"]]}
                    failing = next((s for s in disc["samples"] if not s["passed"]), None)
                    if record["round"] == last_round and failing:
                        fail = failing.get("first_fail") or {}
                        index = fail.get("index")
                        expected = record["expected_hidden"][index] if index is not None else None
                        target["example_failure"] = {"body": failing.get("body", "")[:1200],
                                                     "input": fail.get("input"),
                                                     "expected": json.dumps(expected)[:300] if index is not None else None,
                                                     "got": str(fail.get("got"))[:300] if fail else failing.get("outcome")}
                    row["target"] = target
                elif disc:
                    row["target"] = "not attempted: " + str(disc.get("reason"))[:120]
                else:
                    row["target"] = "not attempted (invalid or duplicate)"
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ teacher rounds
    def _teacher_round(self, round_index):
        remaining = self.candidates_meter.remaining()
        request_n = min(self.cfg.per_call_tasks, remaining)
        history = self._history(self.cfg.feedback_mode, round_index - 1) if round_index else []
        message = teacher_round_message(round_index=round_index, rounds=self.cfg.max_teacher_calls,
                                        per_round=request_n, candidates_used=self.candidates_meter.used,
                                        candidates_max=self.cfg.max_candidates, history=history,
                                        feedback_mode=self.cfg.feedback_mode)
        result = self.broker.call("teacher_generate", teacher_system(self.cfg.max_reference_ast_nodes), message,
                                  arm_id=self.cfg.arm_id,
                                  round=round_index)
        call = {"round": round_index, "call_id": result.call_id, "outcome": result.outcome,
                "usd": result.usd, "latency_seconds": result.latency_seconds, "usage": result.usage,
                "error": result.error, "tasks_parsed": 0, "parse_error": None}
        self.teacher_calls.append(call)
        if result.outcome == "budget_stop":
            raise BudgetStop("teacher budget: " + str(result.error))
        records = []
        if result.ok:
            try:
                tasks = extract_json_object(result.text).get("tasks")
                if not isinstance(tasks, list):
                    raise ValueError("no tasks list")
            except ValueError as exc:
                call["parse_error"] = str(exc)
                tasks = []
            call["tasks_parsed"] = len(tasks)
            for index, raw in enumerate(tasks):
                if index >= request_n:
                    self.log.append("excess_task_not_counted", round=round_index, index=index,
                                    raw_task_blob=self.log.put_blob(raw))
                    continue
                self.candidates_meter.take(1)
                records.append(self._new_candidate(raw, round_index, index, result.call_id))
        self.log.append("teacher_round_done", **call)
        return records

    # ------------------------------------------------------------------ run
    def run(self):
        cfg = self.cfg
        self.log.append("arm_started", config=asdict(cfg), feedback_fields=FEEDBACK_FIELDS[cfg.feedback_mode],
                        versions=versions(cfg.max_reference_ast_nodes), teacher_route=getattr(getattr(self.broker, "route", None), "identity", None),
                        target=getattr(self.target, "identity", None))
        try:
            if cfg.kind == "template":
                for raw in templates.generate(templates.CONTROL, cfg.max_candidates, cfg.seed,
                                              max_ast_nodes=cfg.max_reference_ast_nodes):
                    self.candidates_meter.take(1)
                    self._new_candidate(raw, 0, len(self.candidates), None)
                self._local_and_dedup(self.candidates)
                self.clock.check()
                self._attempt_target(self.candidates)
            elif cfg.kind == "batch":
                for round_index in range(cfg.max_teacher_calls):
                    if self.candidates_meter.remaining() <= 0:
                        break
                    self.clock.check(reserve_seconds=300)
                    self._teacher_round(round_index)
                self.log.append("batch_precommitted", n_candidates=len(self.candidates))
                self._local_and_dedup(self.candidates)
                self.clock.check()
                self._attempt_target(self.candidates)
            elif cfg.kind == "adaptive":
                for round_index in range(cfg.max_teacher_calls):
                    if self.candidates_meter.remaining() <= 0:
                        break
                    self.clock.check(reserve_seconds=300)
                    records = self._teacher_round(round_index)
                    self._local_and_dedup(records)
                    self._attempt_target(records)
            else:
                raise ValueError(cfg.kind)
            self.stop_reason = "candidate_allowance_or_call_limit"
        except BudgetStop as stop:
            self.stop_reason = "budget_stop: " + str(stop)
            # Candidates already committed still receive local checks and, budget permitting, target attempts.
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

    def finish(self):
        summary = {"arm_id": self.cfg.arm_id, "kind": self.cfg.kind, "stop_reason": self.stop_reason,
                   "clock": self.clock.snapshot(), "candidates": self.candidates_meter.snapshot(),
                   "target_queries": self.target_meter.snapshot(),
                   "teacher_meter": self.broker.meter.snapshot() if self.broker else None,
                   "teacher_calls": self.teacher_calls, "config": asdict(self.cfg),
                   "versions": versions(self.cfg.max_reference_ast_nodes),
                   "feedback_fields": FEEDBACK_FIELDS[self.cfg.feedback_mode]}
        with (self.dir / "candidates.jsonl").open("w") as handle:
            for record in self.candidates:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        (self.dir / "arm_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        self.log.append("arm_finished", stop_reason=self.stop_reason, n_candidates=len(self.candidates))
        manifest = freeze_directory(self.dir)
        return summary, manifest


def versions(max_ast_nodes=None):
    from .teacher import BROKER_VERSION, INFRASTRUCTURE_RETRIES
    return {"broker": BROKER_VERSION, "infrastructure_retries": INFRASTRUCTURE_RETRIES,
            "schema": SCHEMA_VERSION, "prompt_template": PROMPT_TEMPLATE_VERSION, "verifier": VERIFIER_VERSION,
            "dedup": DEDUP_VERSION, "prompts": PROMPTS_VERSION, "prompts_sha256": prompts_sha256(max_ast_nodes),
            "max_reference_ast_nodes": max_ast_nodes}
