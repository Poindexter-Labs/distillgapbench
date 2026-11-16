#!/bin/bash
# One GPU, no API keys: the sandbox check, then the operator template on code output prediction for Qwen3-1.7B-Base
# (4 pairs: discovery, fresh final samples, report). Takes a few minutes after the weights download (about 4 GB).
#   bash scripts/run_smoke.sh [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-$HOME/runs/smoke}"
CV="${CONTROLLER_PYTHON:-python}"
"$CV" scripts/check_sandbox.py
bash scripts/run_cells.sh cells/smoke_template.json "$OUT"
"$CV" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
report = json.loads((Path(sys.argv[1]) / "template__qwen1p7b__output/report.json").read_text())
for arm in report["per_arm"]:
    print(arm["arm_id"], "pairs attempted:", arm.get("pairs_attempted"), "scored:", arm.get("pairs_final_scored"),
          "frontier gaps:", arm.get("frontier_gaps"), "stop:", arm.get("stop_reason"))
print("smoke OK:", Path(sys.argv[1]) / "template__qwen1p7b__output/report.json")
PY
