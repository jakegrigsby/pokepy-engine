"""CI gate: PRNG frame trace comparison harness."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ENGINE_ROOT = Path(__file__).resolve().parents[1]
METAMON_ROOT = ENGINE_ROOT.parents[2]
SHOWDOWN_BIN = METAMON_ROOT / "server" / "pokemon-showdown" / "pokemon-showdown"

pytestmark = pytest.mark.skipif(
    not SHOWDOWN_BIN.exists(),
    reason="Showdown CLI required for frame trace tests",
)

TRACE_MATRIX = [
    (1, "slp_spore", 999),
    (1, "par_twave", 12345),
    (2, "slp_spore", 424242),
    (3, "slp_spore", 500),
]


def _run_trace(gen: int, scenario: str, seed: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["POKEPY_PRNG_TRACE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(ENGINE_ROOT / "scripts" / "sim_frame_trace.py"),
            "--gen",
            str(gen),
            "--scenario",
            scenario,
            "--seed",
            str(seed),
            "--turns",
            "3",
            "--max-rows",
            "20",
        ],
        cwd=str(ENGINE_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("gen,scenario,seed", TRACE_MATRIX)
def test_sim_frame_trace_no_divergence(gen, scenario, seed):
    proc = _run_trace(gen, scenario, seed)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "no divergence" in proc.stdout.lower() or "0 divergences" in proc.stdout.lower()
    )
