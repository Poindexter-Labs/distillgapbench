"""The namespace sandbox pool on Linux: many evaluations in a row, as the output-prediction template and the local
checks make them. Skipped where the pool cannot start (macOS, or a container without the documented options)."""
import random
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sandbox"), str(ROOT / "envs")]


@pytest.fixture(scope="module")
def pool():
    if not sys.platform.startswith("linux") or not shutil.which("unshare") or shutil.which("bwrap"):
        pytest.skip("needs Linux with unshare (and no bubblewrap)")
    import ns_pool
    from distillgap import sandbox

    ns_pool.install_pool(size=2)
    probe = sandbox.run_function("def f(x):\n    return x + 1", "f", [[1]], per_call_seconds=5.0)
    if probe.status != "ok":
        pytest.skip(f"sandbox pool unavailable here: {probe.detail}")
    return sandbox


def test_repeated_evaluations(pool):
    for i in range(6):
        res = pool.run_function("def f(x):\n    return x * 2", "f", [[i]], per_call_seconds=5.0)
        assert res.status == "ok" and res.calls[0]["value"] == 2 * i


def test_output_template_pairs(pool):
    from distillgap import adapters

    rng = random.Random(940000)
    for index in range(4):
        pair = adapters.ADAPTERS["output"].template_pair(rng, index)
        assert pair["control"]["call"] == pair["challenge"]["call"]
