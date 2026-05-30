#!/usr/bin/env python3
"""Compare pokepy vs Showdown PRNG frame consumption for a parity scenario.

Usage:
  POKEPY_PRNG_TRACE=1 python scripts/sim_frame_trace.py --gen 1 --scenario slp_spore --seed 999
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Enable tracing before gen5_prng is imported.
os.environ["POKEPY_PRNG_TRACE"] = "1"

ENGINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ENGINE_ROOT / "scripts"
TESTS_DIR = ENGINE_ROOT / "tests"
for d in (str(SCRIPTS_DIR), str(TESTS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

from parity_heuristic_e2e import (  # noqa: E402
    SHOWDOWN_BIN,
    parity_n_turns,
    run_pokepy_battle,
    team_to_showdown_packed,
)
from pokepy.core.gen_profile import profile_for_gen  # noqa: E402
from pokepy.data.loader import (
    load_game_data,
    load_id_mappings,
    load_move_effect_data,
)  # noqa: E402
from pokepy.data.type_charts import load_type_chart_for_gen  # noqa: E402
from pokepy.utils import gen5_prng as prng_mod  # noqa: E402
from test_pokepy_multigen_live_diff import (
    _STATUS_SCENARIOS,
    _status_mirror_team,
)  # noqa: E402

_PRNGTRACE_RE = re.compile(r"^PRNGTRACE\s+(\d+)\s+([\d.]+)\s+(.*)$")


def _clear_pokepy_trace() -> None:
    prng_mod._PRNG_TRACE_LOG.clear()


def _pokepy_trace() -> List[Tuple[int, str]]:
    return [(int(v), str(label)) for v, label in prng_mod._PRNG_TRACE_LOG]


def _run_showdown_trace(
    seed_tuple: Tuple[int, int, int, int],
    packed0: str,
    packed1: str,
    p0_actions: List[str],
    p1_actions: List[str],
    *,
    battle_format: str,
    choice_log: List[Tuple[str, str]],
    timeout_s: int = 120,
) -> Tuple[List[Tuple[int, str]], int, str]:
    lines = [
        f'>start {{"formatid":"{battle_format}","seed":{list(seed_tuple)}}}',
        f'>player p1 {{"name":"P1","team":"{packed0}"}}',
        f'>player p2 {{"name":"P2","team":"{packed1}"}}',
    ]
    for player, action in choice_log:
        lines.append(f">{player} {action}")
    script = ("\n".join(lines) + "\n").encode()

    env = os.environ.copy()
    env["POKEPY_PRNG_TRACE"] = "1"
    proc = subprocess.run(
        [str(SHOWDOWN_BIN), "simulate-battle"],
        input=script,
        capture_output=True,
        cwd=str(SHOWDOWN_BIN.parent),
        timeout=timeout_s,
        env=env,
    )
    stderr = proc.stderr.decode(errors="replace")
    trace: List[Tuple[int, str]] = []
    for line in stderr.splitlines():
        m = _PRNGTRACE_RE.match(line.strip())
        if m:
            trace.append((int(m.group(1)), m.group(3).strip()))
    return trace, proc.returncode, stderr


def _print_alignment(
    pokepy: List[Tuple[int, str]],
    showdown: List[Tuple[int, str]],
    *,
    max_rows: int | None = None,
) -> int | None:
    n = max(len(pokepy), len(showdown))
    if max_rows is not None:
        n = min(n, max_rows)
    print(
        f"{'idx':>4} | {'py_val':>12} | pokepy_label | {'sh_val':>12} | showdown_label | MATCH"
    )
    print("-" * 100)
    first_div: int | None = None
    for i in range(n):
        py_val, py_lab = pokepy[i] if i < len(pokepy) else ("-", "-")
        sh_val, sh_lab = showdown[i] if i < len(showdown) else ("-", "-")
        if py_val == "-" or sh_val == "-":
            match = "LEN"
        else:
            match = "OK" if int(py_val) == int(sh_val) else "DIFF"
        if match != "OK" and first_div is None:
            first_div = i
        print(
            f"{i:4d} | {str(py_val):>12} | {py_lab[:40]:40} | {str(sh_val):>12} | {sh_lab[:40]:40} | {match}"
        )
    print()
    if first_div is None and len(pokepy) == len(showdown):
        print("no divergence")
    elif first_div is not None:
        print(f"first divergence at index {first_div}")
    else:
        print(f"length mismatch: pokepy={len(pokepy)} showdown={len(showdown)}")
    return first_div


def main() -> int:
    parser = argparse.ArgumentParser(description="PRNG frame trace comparison")
    parser.add_argument("--gen", type=int, required=True, choices=[1, 2, 3, 4, 9])
    parser.add_argument(
        "--scenario", choices=sorted(_STATUS_SCENARIOS.keys()), required=True
    )
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=80)
    args = parser.parse_args()

    if not SHOWDOWN_BIN.exists():
        print(f"error: Showdown CLI not found at {SHOWDOWN_BIN}", file=sys.stderr)
        return 2

    n_turns = args.turns if args.turns is not None else parity_n_turns()
    profile = profile_for_gen(args.gen)
    gd = load_game_data(gen=args.gen)
    me = load_move_effect_data(gen=args.gen)
    mappings = load_id_mappings(gen=args.gen)
    chart = load_type_chart_for_gen(args.gen)
    team = _status_mirror_team(args.gen, args.scenario)
    seed_tuple = (args.seed & 0xFFFF, (args.seed >> 16) & 0xFFFF, 0, 0)

    choice_log: List[Tuple[str, str]] = []
    _clear_pokepy_trace()
    _, p0_actions, p1_actions = run_pokepy_battle(
        team,
        team,
        gd,
        me,
        mappings,
        chart,
        args.seed,
        n_turns,
        gen=args.gen,
        choice_log=choice_log,
    )
    pokepy_trace = _pokepy_trace()

    show_trace, rc, stderr = _run_showdown_trace(
        seed_tuple,
        team_to_showdown_packed(team, mappings),
        team_to_showdown_packed(team, mappings),
        p0_actions,
        p1_actions,
        battle_format=profile.battle_format,
        choice_log=choice_log,
    )
    if rc != 0:
        print(f"Showdown failed rc={rc}", file=sys.stderr)
        print(stderr[-2000:], file=sys.stderr)
        return 1

    print(
        f"gen={args.gen} scenario={args.scenario} seed={args.seed} "
        f"turns={n_turns} frames: pokepy={len(pokepy_trace)} showdown={len(show_trace)}"
    )
    div = _print_alignment(pokepy_trace, show_trace, max_rows=args.max_rows)
    return 0 if div is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
