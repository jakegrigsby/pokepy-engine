#!/usr/bin/env python3
"""Fast targeted pokepy vs Showdown live-diff probe.

Examples:
  python scripts/parity_probe.py --gen 3 --scenario slp_spore --seed 999
  python scripts/parity_probe.py --gen 1 --scenario slp_hypnosis --turns 3 -v
  POKEPY_PARITY_TURNS=3 python scripts/parity_probe.py --gen 3 --scenario slp_spore
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ENGINE_ROOT / "scripts"
TESTS_DIR = ENGINE_ROOT / "tests"
for d in (str(SCRIPTS_DIR), str(TESTS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

from parity_heuristic_e2e import (  # noqa: E402
    SHOWDOWN_BIN,
    parity_n_turns,
    run_live_diff,
)
from test_pokepy_multigen_live_diff import (
    _STATUS_SCENARIOS,
    _status_mirror_team,
)  # noqa: E402

_ROW_FIELDS = ("p0_hp", "p0_max_hp", "p0_status", "p1_hp", "p1_max_hp", "p1_status")


def _format_row(prefix: str, row: dict) -> str:
    parts = [f"turn={row.get('turn')}"]
    for f in _ROW_FIELDS:
        parts.append(f"{f}={row.get(f)}")
    if row.get("p0_action") is not None:
        parts.append(f"p0={row.get('p0_action')!r}")
        parts.append(f"p1={row.get('p1_action')!r}")
    return f"{prefix}: " + " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted parity live-diff probe")
    parser.add_argument("--gen", type=int, required=True, choices=[1, 2, 3, 4, 9])
    parser.add_argument(
        "--scenario",
        choices=sorted(_STATUS_SCENARIOS.keys()),
        help="Tier B status mirror scenario",
    )
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help=f"battle turns (default: POKEPY_PARITY_TURNS or {parity_n_turns()})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print all turn rows"
    )
    args = parser.parse_args()

    if not SHOWDOWN_BIN.exists():
        print(f"error: Showdown CLI not found at {SHOWDOWN_BIN}", file=sys.stderr)
        return 2

    if args.scenario is None:
        print("error: --scenario required", file=sys.stderr)
        return 2

    n_turns = args.turns if args.turns is not None else parity_n_turns()
    team = _status_mirror_team(args.gen, args.scenario)
    py_rows, show_rows, mismatch, meta = run_live_diff(
        team,
        team,
        seed=args.seed,
        n_turns=n_turns,
        gen=args.gen,
    )

    if meta.get("returncode") != 0:
        print(f"Showdown error: {meta}", file=sys.stderr)
        return 1

    print(
        f"gen={args.gen} scenario={args.scenario} seed={args.seed} "
        f"turns={n_turns} mismatch={mismatch!r}"
    )

    if args.verbose or mismatch:
        by_sh = {r["turn"]: r for r in show_rows}
        for py_row in py_rows:
            turn = py_row["turn"]
            print(_format_row("py", py_row))
            if turn in by_sh:
                print(_format_row("sh", by_sh[turn]))
            print()

    return 0 if mismatch is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
