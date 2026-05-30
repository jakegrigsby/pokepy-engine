"""End-to-end parity harness: pokepy battles vs Showdown scripted replay.

Used by tests/test_pokepy_parity_regressions.py. Parameterized by ``format``
(default ``gen9ou``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pokepy.core.constants import (
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    M_ACTIVE0,
    M_ACTIVE1,
    POKEMON_SIZE,
    PHASE_FORCED_SWITCH,
    NUM_BATTLE_ACTIONS,
)
from pokepy.core.bitpack import get_status
from pokepy.core.gen_profile import profile_for_format
from pokepy.data.loader import load_game_data, load_id_mappings, load_move_effect_data
from pokepy.engine import step_battle, step_forced_switch_for_gen
from pokepy.engine.action_mask import get_battle_action_mask
from pokepy.env.battle_env import init_battle_state
from pokepy.utils.gen5_prng import Gen5PRNG

REPO_ROOT = Path(__file__).resolve().parents[4]
SHOWDOWN_BIN = REPO_ROOT / "server" / "pokemon-showdown" / "pokemon-showdown"

# --------------------------------------------------------------------------
# Showdown ground-truth cache
#
# `simulate-battle` is a pure function of its stdin script (format + seed +
# packed teams + the scripted action log). Identical script -> identical
# stdout, deterministically. Re-running the suite after a pokepy change that
# does not alter the action sequence therefore re-simulates the exact same
# battles. We memoize the raw stdout on disk keyed by a hash of the script so
# repeat runs skip the (slow) `node` subprocess entirely.
#
# Correctness: the key covers everything Showdown reads, so any change to the
# inputs (including a different pokepy-driven action log) yields a different
# key and forces a fresh run. A Showdown rebuild changes the version token and
# busts the whole cache. Only successful (returncode 0) runs are cached.
#
# Controls:
#   POKEPY_SHOWDOWN_CACHE=0           disable read+write
#   POKEPY_SHOWDOWN_CACHE_DIR=<path>  override location
#   POKEPY_SHOWDOWN_CACHE_VERSION=<s> extra token to force-bust
#   POKEPY_PRNG_TRACE=1               bypass cache (need live stderr trace)
# --------------------------------------------------------------------------


def _showdown_cache_enabled() -> bool:
    if os.environ.get("POKEPY_PRNG_TRACE") == "1":
        return False
    return os.environ.get("POKEPY_SHOWDOWN_CACHE", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def _showdown_cache_dir() -> Path:
    override = os.environ.get("POKEPY_SHOWDOWN_CACHE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / ".cache" / "showdown_parity"


_SHOWDOWN_VERSION_TOKEN: Optional[str] = None


def _showdown_version_token() -> str:
    """Token that changes when the vendored Showdown build changes.

    Hash the CONTENT (not mtime) of every compiled file under dist/sim and
    dist/data: running the CLI recompiles dist/ on each invocation, bumping
    mtimes without changing behavior (which would bust the cache every run),
    while edits to gen-specific conditions/scripts/moves change content in
    those trees but not necessarily battle.js. Content-hashing the whole
    behavior surface invalidates the cache on any real engine change and
    nothing else. Memoized per process; bump POKEPY_SHOWDOWN_CACHE_VERSION
    to force-invalidate manually.
    """
    global _SHOWDOWN_VERSION_TOKEN
    if _SHOWDOWN_VERSION_TOKEN is None:
        extra = os.environ.get("POKEPY_SHOWDOWN_CACHE_VERSION", "")
        h = hashlib.sha256()
        try:
            dist = SHOWDOWN_BIN.parent / "dist"
            files = sorted(
                p for sub in ("sim", "data") for p in (dist / sub).rglob("*.js")
            )
            for p in files:
                h.update(str(p.relative_to(dist)).encode())
                h.update(p.read_bytes())
            digest = h.hexdigest()[:16]
        except OSError:
            digest = "0"
        _SHOWDOWN_VERSION_TOKEN = f"{extra}:{digest}"
    return _SHOWDOWN_VERSION_TOKEN


def _showdown_cache_path(script: str) -> Path:
    key = hashlib.sha256(
        (_showdown_version_token() + "\0" + script).encode("utf-8")
    ).hexdigest()
    return _showdown_cache_dir() / key[:2] / f"{key}.log"


def _showdown_cache_read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _showdown_cache_write(path: Path, raw: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(raw, encoding="utf-8")
        os.replace(tmp, path)  # atomic; safe under xdist workers
    except OSError:
        pass


# CI / full sweeps use 10. Set POKEPY_PARITY_TURNS=3 while debugging turn-2/3 drift.
_DEFAULT_PARITY_TURNS = 10


def parity_n_turns(default: int = _DEFAULT_PARITY_TURNS) -> int:
    raw = __import__("os").environ.get("POKEPY_PARITY_TURNS")
    if raw is None or raw == "":
        return default
    return max(1, int(raw))


def _active_off(state, side: int) -> int:
    battle = state.battle_state
    active = int(battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    return base + active * POKEMON_SIZE


def _snapshot_side(state, side: int) -> Dict[str, int]:
    off = _active_off(state, side)
    bs = state.battle_state
    hp = int(bs[off + 1])
    max_hp = int(bs[off + 2])
    status = int(get_status(int(bs[off + 12])))
    return {"hp": hp, "max_hp": max_hp, "status": status}


def action_to_showdown_str(action: int, *, forced_suffix: str = "") -> str:
    if action < 4:
        base = f"move {action + 1}"
    else:
        base = f"switch {action - 4 + 1}"
    if forced_suffix:
        return f"{base}+{forced_suffix}"
    return base


def simple_heuristic_action(
    state,
    side: int,
    game_data,
    mask: Optional[np.ndarray] = None,
) -> int:
    if mask is None:
        mask = get_battle_action_mask(state, side, game_data)
    legal = [i for i in range(NUM_BATTLE_ACTIONS) if mask[i]]
    if not legal:
        return 0
    moves = [a for a in legal if a < 4]
    if moves:
        return moves[0]
    return legal[0]


def simple_heuristic_forced_switch(state, side: int, game_data) -> int:
    mask = get_battle_action_mask(state, side, game_data)
    switches = [a for a in range(4, NUM_BATTLE_ACTIONS) if mask[a]]
    return switches[0] if switches else 4


_STAT_LABELS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")


def _side_from_showdown_ident(player: str) -> Optional[int]:
    if player.startswith("p1"):
        return 0
    if player.startswith("p2"):
        return 1
    return None


def _parse_hp_token(token: str) -> Optional[Tuple[int, int]]:
    faint = re.match(r"^(\d+)\s+fnt$", token.strip())
    if faint:
        return int(faint.group(1)), 0
    m = re.match(r"(\d+)/(\d+)", token)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def team_to_showdown_export(team: Dict[str, Any], mappings) -> str:
    lines: List[str] = []
    team_evs = team.get("evs")
    team_ivs = team.get("ivs")
    team_natures = team.get("natures")
    team_genders = team.get("genders")
    for i, sp in enumerate(team["species"]):
        name = mappings.species_names.get(int(sp), str(sp))
        gender = "M"
        if team_genders is not None and i < len(team_genders):
            gender = str(team_genders[i]).strip().upper()[:1] or "M"
        if gender in ("M", "F"):
            name = f"{name} ({gender})"
        item_id = int(team["items"][i])
        item = mappings.item_names.get(item_id, "") if item_id > 0 else ""
        header = f"{name} @ {item}" if item else name
        lines.append(header)
        abil_id = int(team["abilities"][i])
        if abil_id > 0:
            abil = mappings.ability_names.get(abil_id, "")
            if abil:
                lines.append(f"Ability: {abil}")
        level = int(team["levels"][i])
        lines.append(f"Level: {level}")
        if team_evs is not None and i < len(team_evs):
            ev_parts = [
                f"{int(team_evs[i][j])} {_STAT_LABELS[j]}"
                for j in range(6)
                if int(team_evs[i][j]) > 0
            ]
            if ev_parts:
                lines.append(f"EVs: {' / '.join(ev_parts)}")
            else:
                lines.append("EVs: 0 HP / 0 Atk / 0 Def / 0 SpA / 0 SpD / 0 Spe")
        if team_ivs is not None and i < len(team_ivs):
            iv_parts = [
                f"{int(team_ivs[i][j])} {_STAT_LABELS[j]}"
                for j in range(6)
                if int(team_ivs[i][j]) != 31
            ]
            if iv_parts:
                lines.append(f"IVs: {' / '.join(iv_parts)}")
        if team_natures is not None and i < len(team_natures):
            nature = str(team_natures[i]).strip()
            if nature:
                lines.append(f"{nature.title()} Nature")
        move_names = []
        for mid in team["moves"][i]:
            if int(mid) >= 0:
                move_names.append(mappings.move_names.get(int(mid), str(mid)))
        for mn in move_names:
            lines.append(f"- {mn}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


_PACK_TEAM_MEMO: Dict[str, str] = {}


def team_to_showdown_packed(team: Dict[str, Any], mappings) -> str:
    export = team_to_showdown_export(team, mappings)
    # pack-team is a pure function of the export string. The same handful of
    # mirror teams are packed thousands of times across the suite, each
    # spawning a ~2s `node` process. Memoize on disk + in-process so we only
    # ever shell out once per distinct team per Showdown build.
    if export in _PACK_TEAM_MEMO:
        return _PACK_TEAM_MEMO[export]

    cache_path = None
    if _showdown_cache_enabled():
        cache_path = _showdown_cache_path("pack-team\0" + export)
        cached = _showdown_cache_read(cache_path)
        if cached is not None:
            packed = cached.strip()
            _PACK_TEAM_MEMO[export] = packed
            return packed

    proc = subprocess.run(
        [str(SHOWDOWN_BIN), "pack-team"],
        input=export.encode(),
        capture_output=True,
        cwd=str(SHOWDOWN_BIN.parent),
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pack-team failed: {proc.stderr.decode(errors='replace')}")
    packed = proc.stdout.decode().strip()
    if cache_path is not None:
        _showdown_cache_write(cache_path, packed)
    _PACK_TEAM_MEMO[export] = packed
    return packed


def _parse_showdown_log(raw: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    turn = 0
    p0_hp = p0_max = p0_status = 0
    p1_hp = p1_max = p1_status = 0
    turn_p0_action = turn_p1_action = ""

    def _apply_hp(side: Optional[int], hp: int, mx: int) -> None:
        nonlocal p0_hp, p0_max, p1_hp, p1_max
        if side is None:
            return
        # Showdown emits percentage-mod duplicates (e.g. 88/100) alongside
        # absolute HP (203/231). Keep the absolute scale for pokepy parity.
        if mx == 0:
            if side == 0:
                p0_hp = hp
            else:
                p1_hp = hp
            return
        if side == 0:
            if mx <= 100 and p0_max > 100:
                return
            p0_hp, p0_max = hp, mx
        else:
            if mx <= 100 and p1_max > 100:
                return
            p1_hp, p1_max = hp, mx

    def _flush(row_type: str = "normal"):
        nonlocal rows
        if turn <= 0:
            return
        rows.append(
            {
                "type": row_type,
                "turn": turn,
                "p0_hp": p0_hp,
                "p0_max_hp": p0_max,
                "p0_status": p0_status,
                "p1_hp": p1_hp,
                "p1_max_hp": p1_max,
                "p1_status": p1_status,
                "p0_action": turn_p0_action,
                "p1_action": turn_p1_action,
            }
        )

    for line in raw.splitlines():
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd == "turn":
            _flush()
            turn_m = re.match(r"(\d+)", parts[2])
            if not turn_m:
                continue
            turn = int(turn_m.group(1))
            turn_p0_action = turn_p1_action = ""
        elif cmd == "move":
            player = parts[2]
            side = _side_from_showdown_ident(player)
            action = f"move {parts[3]}"
            if side == 0:
                turn_p0_action = (
                    action if not turn_p0_action else turn_p0_action + "+" + action
                )
            elif side == 1:
                turn_p1_action = (
                    action if not turn_p1_action else turn_p1_action + "+" + action
                )
        elif cmd in ("switch", "drag", "replace"):
            player = parts[2]
            side = _side_from_showdown_ident(player)
            if len(parts) > 4:
                parsed = _parse_hp_token(parts[4])
                if parsed is not None:
                    hp_val, max_val = parsed
                    if max_val == 0 and side == 0 and p0_max > 0:
                        _apply_hp(side, hp_val, p0_max)
                    elif max_val == 0 and side == 1 and p1_max > 0:
                        _apply_hp(side, hp_val, p1_max)
                    else:
                        _apply_hp(side, hp_val, max_val)
        elif cmd == "faint":
            player = parts[2]
            side = _side_from_showdown_ident(player)
            if side == 0:
                p0_hp = 0
            elif side == 1:
                p1_hp = 0
        elif cmd == "-damage" or cmd == "damage":
            player = parts[2]
            side = _side_from_showdown_ident(player)
            parsed = _parse_hp_token(parts[3])
            if parsed is None:
                continue
            hp_val, max_val = parsed
            if max_val == 0:
                if side == 0 and p0_max > 0:
                    _apply_hp(side, hp_val, p0_max)
                elif side == 1 and p1_max > 0:
                    _apply_hp(side, hp_val, p1_max)
                else:
                    _apply_hp(side, hp_val, 0)
            else:
                _apply_hp(side, hp_val, max_val)
        elif cmd == "-heal" or cmd == "heal":
            player = parts[2]
            side = _side_from_showdown_ident(player)
            parsed = _parse_hp_token(parts[3])
            if parsed is None:
                continue
            _apply_hp(side, parsed[0], parsed[1])
        elif cmd == "-status":
            player = parts[2]
            side = _side_from_showdown_ident(player)
            status_name = parts[3].lower()
            status_map = {
                "brn": 1,
                "par": 2,
                "slp": 3,
                "frz": 4,
                "psn": 5,
                "tox": 6,
            }
            st = status_map.get(status_name, 0)
            if side == 0:
                p0_status = st
            elif side == 1:
                p1_status = st
        elif cmd == "-curestatus":
            player = parts[2]
            side = _side_from_showdown_ident(player)
            if side == 0:
                p0_status = 0
            elif side == 1:
                p1_status = 0

    _flush()
    return rows


def compare_battle_rows(
    py_rows: List[Dict[str, Any]],
    show_rows: List[Dict[str, Any]],
    *,
    fields: Tuple[str, ...] = (
        "p0_hp",
        "p0_max_hp",
        "p0_status",
        "p1_hp",
        "p1_max_hp",
        "p1_status",
    ),
) -> Optional[str]:
    """Return a human-readable first mismatch, or None if rows agree."""
    by_py = {int(r["turn"]): r for r in py_rows if r.get("type") == "normal"}
    by_sh = {int(r["turn"]): r for r in show_rows if r.get("type") == "normal"}
    common = sorted(set(by_py) & set(by_sh))
    if not common:
        return f"no overlapping turns (py={sorted(by_py)[:5]} show={sorted(by_sh)[:5]})"
    for turn in common:
        py_row = by_py[turn]
        sh_row = by_sh[turn]
        for field in fields:
            if py_row.get(field) != sh_row.get(field):
                return (
                    f"turn {turn} {field}: pokepy={py_row.get(field)!r} "
                    f"showdown={sh_row.get(field)!r} "
                    f"(actions py {py_row.get('p0_action')!r}/"
                    f"{py_row.get('p1_action')!r} "
                    f"show {sh_row.get('p0_action')!r}/"
                    f"{sh_row.get('p1_action')!r})"
                )
    return None


def run_live_diff(
    team0: Dict[str, Any],
    team1: Dict[str, Any],
    seed: int,
    n_turns: int,
    *,
    gen: int = 9,
    battle_format: Optional[str] = None,
    timeout_s: int = 120,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """Run pokepy + Showdown on identical actions; return rows and first mismatch."""
    profile = profile_for_format(battle_format or f"gen{gen}ou")
    gd = load_game_data(gen=gen)
    me = load_move_effect_data(gen=gen)
    mappings = load_id_mappings(gen=gen)
    from pokepy.data.type_charts import load_type_chart_for_gen

    chart = load_type_chart_for_gen(gen)
    choice_log: List[Tuple[str, str]] = []
    py_rows, p0_actions, p1_actions = run_pokepy_battle(
        team0,
        team1,
        gd,
        me,
        mappings,
        chart,
        seed,
        n_turns,
        gen=gen,
        choice_log=choice_log,
    )
    seed_tuple = (seed & 0xFFFF, (seed >> 16) & 0xFFFF, 0, 0)
    show_rows, _raw, meta = run_showdown(
        seed_tuple,
        team_to_showdown_packed(team0, mappings),
        team_to_showdown_packed(team1, mappings),
        p0_actions,
        p1_actions,
        n_turns,
        battle_format=battle_format or profile.battle_format,
        timeout_s=timeout_s,
        choice_log=choice_log,
    )
    mismatch = None
    if meta.get("returncode") == 0 and show_rows:
        mismatch = compare_battle_rows(py_rows, show_rows)
    return py_rows, show_rows, mismatch, meta


def _resolve_pending_forced_switches(
    state,
    gen: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    prng,
    p0_actions: List[str],
    p1_actions: List[str],
    choice_log: Optional[List[Tuple[str, str]]],
) -> None:
    """Resolve post-faint forced switches after a normal turn step."""
    while int(state.phase) == PHASE_FORCED_SWITCH and not bool(state.done):
        forced_side = int(state.forced_switch_side)
        if forced_side in (1, 2):
            fa1 = simple_heuristic_forced_switch(state, 1, game_data)
            action1 = action_to_showdown_str(fa1, forced_suffix="forced")
            p1_actions.append(action1)
            if choice_log is not None:
                choice_log.append(("p2", action1))
            step_forced_switch_for_gen(
                gen,
                state,
                fa1,
                1,
                game_data,
                move_effects,
                type_chart,
                prng,
            )
        if int(state.phase) != PHASE_FORCED_SWITCH or bool(state.done):
            break
        forced_side = int(state.forced_switch_side)
        if forced_side in (0, 2):
            fa0 = simple_heuristic_forced_switch(state, 0, game_data)
            action0 = action_to_showdown_str(fa0, forced_suffix="forced")
            p0_actions.append(action0)
            if choice_log is not None:
                choice_log.append(("p1", action0))
            step_forced_switch_for_gen(
                gen,
                state,
                fa0,
                0,
                game_data,
                move_effects,
                type_chart,
                prng,
            )


def run_pokepy_battle(
    team0: Dict[str, Any],
    team1: Dict[str, Any],
    game_data,
    move_effects,
    mappings,
    type_chart: np.ndarray,
    seed: int,
    n_turns: int,
    gen: int = 9,
    *,
    choice_log: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    state = init_battle_state(team0, team1, game_data, seed, gen=gen)
    prng = Gen5PRNG((seed & 0xFFFF, (seed >> 16) & 0xFFFF, 0, 0))
    rows: List[Dict[str, Any]] = []
    p0_actions: List[str] = []
    p1_actions: List[str] = []

    while len(rows) < int(n_turns):
        if bool(state.done):
            break

        if int(state.phase) == PHASE_FORCED_SWITCH:
            _resolve_pending_forced_switches(
                state,
                gen,
                game_data,
                move_effects,
                type_chart,
                prng,
                p0_actions,
                p1_actions,
                choice_log,
            )
            continue

        mask0 = get_battle_action_mask(state, 0, game_data)
        mask1 = get_battle_action_mask(state, 1, game_data)
        a0 = simple_heuristic_action(state, 0, game_data, mask0)
        a1 = simple_heuristic_action(state, 1, game_data, mask1)
        action0 = action_to_showdown_str(a0)
        action1 = action_to_showdown_str(a1)
        p0_actions.append(action0)
        p1_actions.append(action1)
        if choice_log is not None:
            choice_log.append(("p1", action0))
            choice_log.append(("p2", action1))

        step_battle(
            gen,
            state,
            a0,
            a1,
            game_data,
            move_effects,
            type_chart,
            prng,
            defer_p1_forced_switch=True,
        )

        _resolve_pending_forced_switches(
            state,
            gen,
            game_data,
            move_effects,
            type_chart,
            prng,
            p0_actions,
            p1_actions,
            choice_log,
        )

        turn = int(state.turn)
        if turn <= 0:
            continue
        s0 = _snapshot_side(state, 0)
        s1 = _snapshot_side(state, 1)
        rows.append(
            {
                "type": "normal",
                "turn": turn,
                "p0_hp": s0["hp"],
                "p0_max_hp": s0["max_hp"],
                "p0_status": s0["status"],
                "p1_hp": s1["hp"],
                "p1_max_hp": s1["max_hp"],
                "p1_status": s1["status"],
                "p0_action": action0,
                "p1_action": action1,
            }
        )

    return rows, p0_actions, p1_actions


def run_showdown(
    seed_tuple: Tuple[int, int, int, int],
    packed0: str,
    packed1: str,
    p0_actions: List[str],
    p1_actions: List[str],
    n_turns: int,
    *,
    battle_format: str = "gen9ou",
    timeout_s: int = 120,
    choice_log: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    if not SHOWDOWN_BIN.exists():
        meta = {
            "returncode": 1,
            "timeout": False,
            "error": f"Showdown CLI not found at {SHOWDOWN_BIN}",
        }
        return [], None, meta

    lines = [
        f'>start {{"formatid":"{battle_format}","seed":{list(seed_tuple)}}}',
        f'>player p1 {{"name":"P1","team":"{packed0}"}}',
        f'>player p2 {{"name":"P2","team":"{packed1}"}}',
    ]
    if choice_log:
        for player, action in choice_log:
            lines.append(f">{player} {action}")
    else:
        n = min(len(p0_actions), len(p1_actions), int(n_turns))
        for i in range(n):
            lines.append(f">p1 {p0_actions[i]}")
            lines.append(f">p2 {p1_actions[i]}")

    script = "\n".join(lines) + "\n"

    cache_enabled = _showdown_cache_enabled()
    cache_path = _showdown_cache_path(script) if cache_enabled else None
    if cache_path is not None:
        cached = _showdown_cache_read(cache_path)
        if cached is not None:
            meta = {
                "returncode": 0,
                "timeout": False,
                "error": None,
                "cached": True,
            }
            return _parse_showdown_log(cached), cached, meta

    try:
        proc = subprocess.run(
            [str(SHOWDOWN_BIN), "simulate-battle"],
            input=script.encode(),
            capture_output=True,
            cwd=str(SHOWDOWN_BIN.parent),
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return [], None, {"returncode": -1, "timeout": True, "error": "timeout"}

    raw = proc.stdout.decode(errors="replace")
    meta = {
        "returncode": proc.returncode,
        "timeout": False,
        "error": proc.stderr.decode(errors="replace") if proc.returncode else None,
        "cached": False,
    }
    if proc.returncode != 0:
        return [], raw, meta
    if cache_path is not None:
        _showdown_cache_write(cache_path, raw)
    return _parse_showdown_log(raw), raw, meta
