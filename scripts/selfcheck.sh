#!/bin/bash
# No GPU, no network, no keys: the test suite (fixture student, fake routes) and the namespace-sandbox check.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${CONTROLLER_PYTHON:-python}"
echo "== tests"
"$PY" -m pytest -q -p no:cacheprovider tests
echo "== sandbox"
"$PY" scripts/check_sandbox.py
