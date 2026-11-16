"""Run DistillGapBench v1 cells on one GPU node (the study used one 4x GH200 node per cell list).

A CELL is (researcher, student, environment) with its repetitions and arms; a template cell has researcher null.
Per cell: discovery arms run concurrently (each frozen when done), then the independent final evaluation, then the
light-path judge on frontier gaps (coding and math), then report.json. Cells run concurrently (--max-cells).
Everything is resumable: a frozen arm, a finished final evaluation or judge stage is never redone; an arm directory
left unfrozen by a dead process is moved aside (kept as evidence) and the arm is run again.

  python scripts/run_grid.py --cells cells/v1_leaderboard.json --out runs/X \
      --student-url qwen4b=http://127.0.0.1:8102 ... --sandbox ns_pool

Release changes: the namespace sandbox ships in sandbox/ (--rl-dir defaults to it), credentials come from the
environment (routes.load_credentials), and the light-path judge is optional (--judge; docs/JUDGE.md).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

GRID_VERSION = "distillgap_v1_grid_runner_1"
ARM_DEFAULTS = {"max_candidates": 16, "per_call_tasks": 1, "max_teacher_calls": 20, "samples_per_candidate": 4,
                "target_query_limit": 128, "wall_seconds": 28800, "pair_mode": True,
                "teacher_limits": {"max_calls": 60, "max_input_tokens": 9_000_000, "max_output_tokens": 3_840_000,
                                   "max_output_per_call": 64_000, "max_usd": 300.0}}
CODING_MAX_AST_NODES = 160          # v5 main-study value
VALIDATOR_LIMITS = {"max_calls": 400, "max_input_tokens": 6_000_000, "max_output_tokens": 24_000_000,
                    "max_output_per_call": 64_000, "max_usd": 600.0}
VALIDATOR_EFFORT = "max"            # v5 main-study validator setting
_print_lock = threading.Lock()


def say(*parts):
    with _print_lock:
        print(time.strftime("%H:%M:%S"), *parts, flush=True)


def registry():
    return {r["id"]: r for r in json.loads((HERE / "researchers_v6.json").read_text())["researchers"]}


def prices_for(rid):
    from distillgap.budget import Prices
    spec = registry()[rid]
    p = spec["prices"]
    return Prices(p["input"], p["cached"], p.get("cache_write", p["input"] * 1.25), p.get("cache_write_1h", p["input"] * 2),
                  p["output"], spec.get("price_source", ""))


def validator_for(rid):
    """Cross-family rule: Astra validates Anthropic researchers; Opus 5.5 validates everything else (and templates)."""
    if rid is not None and registry()[rid]["provider"] == "aws_bedrock":
        return "astra"
    return "opus55"


def cell_seed(cell) -> int:
    if cell.get("seed_base") is not None:
        return int(cell["seed_base"])
    key = f"{cell['researcher']}|{cell['student']}|{cell['env']}"
    return 600_000 + int(hashlib.sha256(key.encode()).hexdigest()[:6], 16) % 300_000 * 10


class Grid:
    def __init__(self, args):
        import routes
        self.args, self.routes = args, routes
        self.creds = routes.load_credentials()
        self.urls = dict(item.split("=", 1) for item in args.student_url)
        self._targets, self._lock = {}, threading.Lock()
        self.judge_client = None
        self._arm_slots = {}

    def arm_slot(self, rid):
        """Per-researcher cap on concurrently running arms (an arm's wall clock starts only once it holds a slot)."""
        key = rid or "template"
        with self._lock:
            if key not in self._arm_slots:
                limit = (registry()[rid].get("max_concurrent_arms") if rid else None) or self.args.arms_per_researcher
                self._arm_slots[key] = threading.BoundedSemaphore(limit)
            return self._arm_slots[key]

    def target(self, student, env):
        from distillgap.students import CODING_MAX_TOKENS, CODING_STOP
        from distillgap.target import LocalVllmTarget
        key = (student, env == "coding")
        with self._lock:
            if key not in self._targets:
                coding = env == "coding"
                self._targets[key] = LocalVllmTarget(self.urls[student], student, student, concurrency=16,
                                                     default_stop=CODING_STOP if coding else None,
                                                     default_max_tokens=CODING_MAX_TOKENS if coding else None)
            return self._targets[key]

    def route(self, rid, effort=None):
        return self.routes.make_route(rid, self.creds, effort=effort)

    def judge(self):
        from distillgap.judge import LightPathJudge
        from openai import OpenAI
        with self._lock:
            if self.judge_client is None:
                self.judge_client = OpenAI(api_key=self.creds["azure_key"], base_url=self.creds["azure_base_url"],
                                           timeout=900, max_retries=2)
        return LightPathJudge(self.judge_client)

    # ------------------------------------------------------------------ one cell
    def run_cell(self, cell):
        from distillgap.adapters import ADAPTERS
        from distillgap.budget import ModelLimits, ModelMeter
        from distillgap.controllers import ArmConfig
        from distillgap.env_arm import EnvPairArm
        from distillgap.ledger import EvidenceLog
        from distillgap.pairs import PairArm
        from distillgap.students import teacher_description
        from distillgap.teacher import Broker

        cid, env, student, rid = cell["cell_id"], cell["env"], cell["student"], cell.get("researcher")
        out = Path(self.args.out) / cid
        out.mkdir(parents=True, exist_ok=True)
        stamp = json.dumps(cell, sort_keys=True).encode()
        if (out / "cell.json").exists() and (out / "cell.json").read_bytes() != stamp:
            raise SystemExit(f"{cid}: run directory bound to a different cell config")
        (out / "cell.json").write_bytes(stamp)
        seed_base = cell_seed(cell)
        target = self.target(student, env)
        adapter = ADAPTERS.get(env)
        overrides = {**ARM_DEFAULTS, **cell.get("arm_overrides", {})}
        if env == "coding":
            overrides.setdefault("max_reference_ast_nodes", CODING_MAX_AST_NODES)
        if rid is not None:
            spec = registry()[rid]
            if spec.get("max_output_per_call"):
                overrides["teacher_limits"] = {**overrides["teacher_limits"], "max_output_per_call": spec["max_output_per_call"]}

        # ---------------- discovery
        todo = []
        for rep in cell["reps"]:
            for kind in cell["arms"]:
                arm_id = f"rep{rep}-{kind}"
                arm_dir = out / "arms" / arm_id
                if (arm_dir / "FROZEN_MANIFEST.json").exists():
                    continue
                if arm_dir.exists():
                    arm_dir.rename(arm_dir.with_name(arm_id + f".incomplete-{int(time.time())}"))
                todo.append((rep, kind, arm_id, arm_dir))

        def run_arm(spec):
            rep, kind, arm_id, arm_dir = spec
            with self.arm_slot(rid if kind != "template" else None):
                try:
                    cfg = ArmConfig(arm_id=arm_id, kind=kind, seed=seed_base + 1000 * rep,
                                    target_description=teacher_description(student, env), **overrides)
                    arm = (PairArm(cfg, arm_dir, target) if env == "coding"
                           else EnvPairArm(cfg, arm_dir, target, adapter=adapter))
                    if kind != "template":
                        meter = ModelMeter(f"{arm_id}-teacher", ModelLimits(**cfg.teacher_limits), prices_for(rid))
                        arm.broker = Broker(self.route(rid), meter, arm.log)
                    summary, _ = arm.run()
                    return arm_id, summary["stop_reason"]
                except Exception:  # noqa: BLE001
                    return arm_id, "CRASHED " + traceback.format_exc()[-1500:]

        if todo:
            say(cid, "discovery", [t[2] for t in todo])
            with ThreadPoolExecutor(len(todo)) as pool:
                for arm_id, stop in pool.map(run_arm, todo):
                    say(cid, arm_id, stop)
        arm_dirs = sorted(p for p in (out / "arms").iterdir() if (p / "FROZEN_MANIFEST.json").exists())
        expected = {f"rep{rep}-{kind}" for rep in cell["reps"] for kind in cell["arms"]}
        missing = expected - {p.name for p in arm_dirs}
        if missing:
            raise RuntimeError(f"arms not frozen: {sorted(missing)}")

        # ---------------- final evaluation
        final_dir = out / "final"
        if not (final_dir / "final_meta.json").exists():
            if final_dir.exists():
                final_dir.rename(final_dir.with_name(f"final.incomplete-{int(time.time())}"))
            say(cid, "final evaluation")
            log = EvidenceLog(final_dir)
            vid = validator_for(rid)
            validator = None
            if env in ("coding", "math"):
                validator = Broker(self.route(vid, effort=VALIDATOR_EFFORT),
                                   ModelMeter("validator", ModelLimits(**VALIDATOR_LIMITS), prices_for(vid)), log)
            eval_seed = seed_base + 777_777
            if env == "coding":
                from distillgap.final_eval import final_evaluate
                final_evaluate(arm_dirs, final_dir, log=log, validator_broker=validator, teacher_broker=None,
                               target=target, eval_seed=eval_seed, samples=4, target_query_limit_per_arm=128,
                               concurrency=self.args.final_concurrency, blind_solve=False)
            else:
                from distillgap.env_final import env_final_evaluate
                env_final_evaluate(arm_dirs, final_dir, log=log, adapter=adapter, target=target, eval_seed=eval_seed,
                                   validator_broker=validator, concurrency=self.args.final_concurrency)

        # ---------------- judge (coding, math)
        judge_dir = out / "judge"
        # The operator template row is not judged: its tasks are generated by code with brute-force answers (and still
        # independently validated); the light path reviews researcher-written problems and solutions.
        if (self.args.judge == "light_path" and env in ("coding", "math") and rid is not None
                and not (judge_dir / "judge_results.jsonl").exists()):
            from distillgap.env_final import judge_frontier
            say(cid, "judge")
            judge_frontier(arm_dirs, final_dir, judge_dir, env=env, judge=self.judge(), adapter=adapter)

        # ---------------- report
        report = cell_report(out, arm_dirs, final_dir, judge_dir, cell)
        say(cid, "done", json.dumps({a["arm_id"]: [a.get("frontier_gaps"), a.get("frontier_gaps_judged")]
                                     for a in report["per_arm"]}))
        return report


def judge_usd(review_usage):
    """Light-path judge cost at the Astra registry price (Azure usage fields)."""
    if not review_usage:
        return 0.0
    p = registry()["astra"]["prices"]
    inp = review_usage.get("input_tokens") or 0
    cached = ((review_usage.get("input_tokens_details") or {}).get("cached_tokens")) or 0
    out = review_usage.get("output_tokens") or 0
    return ((inp - cached) * p["input"] + cached * p["cached"] + out * p["output"]) / 1e6


def cell_report(out, arm_dirs, final_dir, judge_dir, cell):
    from distillgap.report import build_report
    report = build_report(arm_dirs, [final_dir], out / "report.json")
    from distillgap.judge import verdict
    judged = {}
    if (judge_dir / "judge_results.jsonl").exists():
        for line in (judge_dir / "judge_results.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                # recompute from the stored light-path outputs with the current veto rule (judge.verdict)
                roles = {r: verdict((row.get(r) or {}).get("output")) for r in ("control", "challenge")}
                row["veto"] = any(v["veto"] for v in roles.values())
                row["strict_veto"] = any(v.get("strict_veto") for v in roles.values())
                row["verdicts"] = {r: v.get("final_verdict") for r, v in roles.items()}
                judged[row["pair_id"]] = row
    judge_total = 0.0
    for arm in report["per_arm"]:
        ids = arm.get("frontier_gap_ids") or []
        vetoed = [i for i in ids if judged.get(i, {}).get("veto")]
        arm["judge_vetoed_ids"] = vetoed
        arm["judge_incomplete"] = sum(1 for i in ids if judged.get(i, {}).get("incomplete"))
        strict = [i for i in ids if judged.get(i, {}).get("strict_veto")]
        arm["judge_strict_vetoed_ids"] = strict
        arm["judge_verdicts"] = {i: judged[i]["verdicts"] for i in ids if i in judged}
        arm["frontier_gaps_judged"] = len(ids) - len(vetoed)
        arm["frontier_gaps_judged_strict"] = len(ids) - len(strict)
        arm["judge_usd"] = round(sum(judge_usd((judged.get(i, {}).get(r) or {}).get("usage")) for i in ids
                                     for r in ("control", "challenge")), 4)
        judge_total += arm["judge_usd"]
    final_meta = json.loads((final_dir / "final_meta.json").read_text())
    report.update(cell=cell, grid_version=GRID_VERSION, judge_usd_total=round(judge_total, 4),
                  validator_usd_total=(final_meta.get("validator_meter") or {}).get("estimated_usd", 0.0))
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--student-url", nargs="+", default=[], help="student=http://host:port")
    parser.add_argument("--sandbox", choices=["ns_pool", "local"], default="ns_pool")
    parser.add_argument("--rl-dir", default=str(HERE / "sandbox"), help="directory with ns_pool.py and linux_sandbox.py")
    parser.add_argument("--judge", choices=["auto", "light_path", "none"], default="auto",
                        help="light-path ambiguity judge on coding/math frontier gaps; auto = light_path if installed")
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--max-cells", type=int, default=24)
    parser.add_argument("--final-concurrency", type=int, default=12)
    parser.add_argument("--arms-per-researcher", type=int, default=72, help="concurrent arms per researcher (default)")
    parser.add_argument("--only", nargs="*", default=None, help="cell ids to run (default: all)")
    args = parser.parse_args()
    if args.judge == "auto":
        from distillgap import judge as _judge
        args.judge = "none" if _judge.get_light_review is None else "light_path"
    say("judge", args.judge)
    if args.sandbox == "ns_pool":
        sys.path.insert(0, str(Path(args.rl_dir).resolve()))
        import ns_pool
        info = ns_pool.install_pool(size=args.pool_size)
        say("sandbox", json.dumps(info))
    cells = json.loads(Path(args.cells).read_text())
    if args.only:
        cells = [c for c in cells if c["cell_id"] in set(args.only)]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    grid = Grid(args)
    say("grid", GRID_VERSION, len(cells), "cells; students", sorted(grid.urls))

    def run(cell):
        status = Path(args.out) / cell["cell_id"] / "cell_status.json"
        try:
            report = grid.run_cell(cell)
            status.write_text(json.dumps({"state": "done", "unix": time.time(),
                                          "frontier_gaps": {a["arm_id"]: a.get("frontier_gaps") for a in report["per_arm"]},
                                          "frontier_gaps_judged": {a["arm_id"]: a.get("frontier_gaps_judged")
                                                                   for a in report["per_arm"]}}, indent=1) + "\n")
            return cell["cell_id"], "done"
        except BaseException as exc:  # noqa: BLE001 - one cell's failure never stops the others
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text(json.dumps({"state": "failed", "unix": time.time(),
                                          "error": traceback.format_exc()[-3000:]}, indent=1) + "\n")
            say(cell["cell_id"], "FAILED", type(exc).__name__, str(exc)[:300])
            return cell["cell_id"], "failed"

    with ThreadPoolExecutor(args.max_cells) as pool:
        results = list(pool.map(run, cells))
    say("grid finished", json.dumps(dict(results)))


if __name__ == "__main__":
    main()
