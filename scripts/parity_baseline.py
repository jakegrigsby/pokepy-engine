#!/usr/bin/env python3
"""Print pass/fail counts for parity suites (baseline scoreboard).

Recipes:
  # Live-diff vs Showdown CLI (requires server/pokemon-showdown/pokemon-showdown):
  cd metamon/env/pokepy-engine
  python -m pytest tests/test_pokepy_multigen_live_diff.py -q

  # Recorded regression goldens (no live binary):
  python -m pytest tests/test_pokepy_parity_regressions.py -q

  # Frame trace for a single scenario (first divergence):
  POKEPY_PRNG_TRACE=1 python scripts/sim_frame_trace.py --gen 1 --scenario slp_spore --seed 999
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(suite: str) -> tuple[int, int, int]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", suite, "-q", "--tb=no"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    passed = failed = skipped = 0
    for line in out.splitlines():
        line = line.strip()
        if line.endswith("passed") or " passed" in line:
            parts = line.replace(",", "").split()
            for i, p in enumerate(parts):
                if p == "passed" and i:
                    passed = int(parts[i - 1])
                elif p == "failed" and i:
                    failed = int(parts[i - 1])
                elif p == "skipped" and i:
                    skipped = int(parts[i - 1])
    return passed, failed, skipped


def main() -> None:
    suites = [
        ("live_diff", "tests/test_pokepy_multigen_live_diff.py"),
        ("regressions", "tests/test_pokepy_parity_regressions.py"),
    ]
    print("Parity baseline scoreboard")
    print("=" * 60)
    for name, path in suites:
        p, f, s = _run(path)
        print(f"{name:14}  passed={p:4}  failed={f:4}  skipped={s:4}")
    print()
    print(
        "Frame trace: POKEPY_PRNG_TRACE=1 python scripts/sim_frame_trace.py --gen N --scenario NAME --seed S"
    )


if __name__ == "__main__":
    main()
