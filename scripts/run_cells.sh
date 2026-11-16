#!/bin/bash
# Serve the students a cell list needs (one vLLM server per student, one GPU each, localhost only, as in the study),
# then run the grid: researcher calls, sandboxed checks, independent final evaluation, judge, reports.
#   bash scripts/run_cells.sh CELLS_JSON OUT_DIR [extra run_grid.py arguments, e.g. --judge none --only CELL_ID]
# Weights download from Hugging Face at the pinned revisions into HF_HOME (Gemma needs an accepted licence + HF_TOKEN).
set -uo pipefail
cd "$(dirname "$0")/.."
CELLS="$1"; OUT="$2"; shift 2
CV="${CONTROLLER_PYTHON:-python}"; VP="${VLLM_PYTHON:-python3}"
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer's sampler JIT needs nvcc; the study ran without it
NEEDED=$("$CV" -c "import json,sys; print(' '.join(sorted({c['student'] for c in json.load(open(sys.argv[1]))})))" "$CELLS")
# GPUs: those already assigned (CUDA_VISIBLE_DEVICES, e.g. by Slurm), otherwise every GPU nvidia-smi lists
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then GPU_LIST=(${CUDA_VISIBLE_DEVICES//,/ }); else
  GPU_LIST=($(seq 0 $(( $(nvidia-smi -L 2>/dev/null | wc -l) - 1 )))); fi
COUNT=$(echo $NEEDED | wc -w)
if [ "$COUNT" -gt "${#GPU_LIST[@]}" ]; then
  echo "need $COUNT GPUs (one per student: $NEEDED), found ${#GPU_LIST[@]}; use a per-student cell file such as cells/v1_qwen4b.json" >&2
  exit 2
fi
PIDS=(); URLS=(); WATCH=(); i=0
while read -r name repo rev; do
  port=$((8101 + i))
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} "$VP" -m vllm.entrypoints.openai.api_server --model "$repo" --revision "$rev" \
    --served-model-name "$name" --host 127.0.0.1 --port "$port" --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --seed 0 --generation-config vllm > "$OUT/vllm-$name.log" 2>&1 &
  PIDS+=($!); URLS+=("$name=http://127.0.0.1:$port"); WATCH+=("$name=$!"); i=$((i + 1))
done < <("$CV" scripts/students.py $NEEDED)
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
"$CV" - "${URLS[@]}" -- "${WATCH[@]}" <<'PY'
import os, sys, time, requests
args = sys.argv[1:]
cut = args.index("--")
pending = dict(u.split("=", 1) for u in args[:cut])
pids = {k: int(v) for k, v in (w.split("=", 1) for w in args[cut + 1:])}
deadline = time.time() + 1800
while pending and time.time() < deadline:
    for name, url in list(pending.items()):
        try:
            if requests.get(url + "/v1/models", timeout=5).ok:
                print("student server ready:", name, flush=True); pending.pop(name)
                continue
        except Exception:
            pass
        if not os.path.exists(f"/proc/{pids[name]}"):
            sys.exit(f"student server {name} exited before it was ready (see vllm-{name}.log)")
    time.sleep(10)
if pending:
    sys.exit("student servers not ready: " + ", ".join(pending))
PY
if [ $? -ne 0 ]; then for n in $NEEDED; do tail -20 "$OUT/vllm-$n.log"; done; exit 3; fi
"$CV" -u scripts/run_grid.py --cells "$CELLS" --out "$OUT" --student-url "${URLS[@]}" --sandbox ns_pool "$@"
STATUS=$?
echo "grid exit $STATUS"
exit $STATUS
