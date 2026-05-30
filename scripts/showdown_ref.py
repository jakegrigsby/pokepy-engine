"""Showdown ground-truth reference helpers (engine-agnostic).

Extracted from ``parity_heuristic_e2e`` so the verbatim-port harness
(``pokepy.showdown``) can diff against live Showdown without importing the
removed packed-state engine. Everything here is a pure function of its inputs:
team dicts + an id->name ``mappings`` object + a scripted action log. Showdown
output is memoized on disk keyed by a content hash of the script.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
SHOWDOWN_BIN = REPO_ROOT / "server" / "pokemon-showdown" / "pokemon-showdown"

_STAT_LABELS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
_DEFAULT_PARITY_TURNS = 10


def parity_n_turns(default: int = _DEFAULT_PARITY_TURNS) -> int:
    raw = os.environ.get("POKEPY_PARITY_TURNS")
    if raw is None or raw == "":
        return default
    return max(1, int(raw))


# --------------------------------------------------------------------------
# Showdown ground-truth cache (content-hashed; see parity_heuristic_e2e docs)
# --------------------------------------------------------------------------


def _showdown_cache_enabled() -> bool:
    if os.environ.get("POKEPY_PRNG_TRACE") == "1":
        return False
    return os.environ.get("POKEPY_SHOWDOWN_CACHE", "1").lower() not in ("0", "false", "no")


def _showdown_cache_dir() -> Path:
    override = os.environ.get("POKEPY_SHOWDOWN_CACHE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / ".cache" / "showdown_parity"


_SHOWDOWN_VERSION_TOKEN: Optional[str] = None


def _showdown_version_token() -> str:
    global _SHOWDOWN_VERSION_TOKEN
    if _SHOWDOWN_VERSION_TOKEN is None:
        extra = os.environ.get("POKEPY_SHOWDOWN_CACHE_VERSION", "")
        h = hashlib.sha256()
        try:
            dist = SHOWDOWN_BIN.parent / "dist"
            files = sorted(p for sub in ("sim", "data") for p in (dist / sub).rglob("*.js"))
            for p in files:
                h.update(str(p.relative_to(dist)).encode())
                h.update(p.read_bytes())
            digest = h.hexdigest()[:16]
        except OSError:
            digest = "0"
        _SHOWDOWN_VERSION_TOKEN = f"{extra}:{digest}"
    return _SHOWDOWN_VERSION_TOKEN


def _showdown_cache_path(script: str) -> Path:
    key = hashlib.sha256((_showdown_version_token() + "\0" + script).encode("utf-8")).hexdigest()
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
        os.replace(tmp, path)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Team -> Showdown packed string
# --------------------------------------------------------------------------


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
        for mid in team["moves"][i]:
            if int(mid) >= 0:
                lines.append(f"- {mappings.move_names.get(int(mid), str(mid))}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


_PACK_TEAM_MEMO: Dict[str, str] = {}


def team_to_showdown_packed(team: Dict[str, Any], mappings) -> str:
    export = team_to_showdown_export(team, mappings)
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


# --------------------------------------------------------------------------
# Showdown log parsing + comparison
# --------------------------------------------------------------------------


def _packed_team_size(packed: str) -> int:
    """Number of mons in a Showdown packed-team string (mons split by ']')."""
    if not packed:
        return 0
    return packed.count("]") + 1


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
            side = _side_from_showdown_ident(parts[2])
            action = f"move {parts[3]}"
            if side == 0:
                turn_p0_action = action if not turn_p0_action else turn_p0_action + "+" + action
            elif side == 1:
                turn_p1_action = action if not turn_p1_action else turn_p1_action + "+" + action
        elif cmd in ("switch", "drag", "replace"):
            side = _side_from_showdown_ident(parts[2])
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
            side = _side_from_showdown_ident(parts[2])
            if side == 0:
                p0_hp = 0
            elif side == 1:
                p1_hp = 0
        elif cmd in ("-damage", "damage"):
            side = _side_from_showdown_ident(parts[2])
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
        elif cmd in ("-heal", "heal"):
            side = _side_from_showdown_ident(parts[2])
            parsed = _parse_hp_token(parts[3])
            if parsed is None:
                continue
            _apply_hp(side, parsed[0], parsed[1])
        elif cmd == "-status":
            side = _side_from_showdown_ident(parts[2])
            status_map = {"brn": 1, "par": 2, "slp": 3, "frz": 4, "psn": 5, "tox": 6}
            st = status_map.get(parts[3].lower(), 0)
            if side == 0:
                p0_status = st
            elif side == 1:
                p1_status = st
        elif cmd == "-curestatus":
            side = _side_from_showdown_ident(parts[2])
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
                    f"(actions py {py_row.get('p0_action')!r}/{py_row.get('p1_action')!r} "
                    f"show {sh_row.get('p0_action')!r}/{sh_row.get('p1_action')!r})"
                )
    return None


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
        meta = {"returncode": 1, "timeout": False, "error": f"Showdown CLI not found at {SHOWDOWN_BIN}"}
        return [], None, meta

    lines = [
        f'>start {{"formatid":"{battle_format}","seed":{list(seed_tuple)}}}',
        f'>player p1 {{"name":"P1","team":"{packed0}"}}',
        f'>player p2 {{"name":"P2","team":"{packed1}"}}',
    ]
    # Gen5+ formats open with team preview; lead in natural team order so the
    # replay matches the engine (which leads with team slot 0).
    gen_m = re.match(r"gen(\d+)", battle_format)
    if gen_m and int(gen_m.group(1)) >= 5:
        order0 = "".join(str(i + 1) for i in range(_packed_team_size(packed0)))
        order1 = "".join(str(i + 1) for i in range(_packed_team_size(packed1)))
        lines.append(f">p1 team {order0}")
        lines.append(f">p2 team {order1}")
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
            meta = {"returncode": 0, "timeout": False, "error": None, "cached": True}
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
