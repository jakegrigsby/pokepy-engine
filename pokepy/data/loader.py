"""Standalone .npy game data loader.

Reads Pokemon Showdown data tables bundled with pokepy at
`pokepy/data/extracted/` (or per-gen subdirs `extracted/gen{N}/`).
Override the location with the `POKEPY_DATA_PATH` environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as _f
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class GameData:
    """Pure-numpy game data tables."""

    type_chart: np.ndarray
    species_base_stats: np.ndarray
    species_types: np.ndarray
    species_weight: np.ndarray
    species_abilities: np.ndarray
    move_base_power: np.ndarray
    move_type: np.ndarray
    move_category: np.ndarray
    move_priority: np.ndarray
    move_accuracy: np.ndarray
    move_pp: np.ndarray
    move_crit_ratio: np.ndarray
    move_flags: np.ndarray
    move_target: np.ndarray
    item_fling_power: Optional[np.ndarray] = None
    item_is_berry: Optional[np.ndarray] = None
    item_is_choice: Optional[np.ndarray] = None


@dataclass
class MoveEffectData:
    """Move effect arrays indexed by move ID. Built from move_effects.MOVE_EFFECTS."""

    effect_type: np.ndarray
    status: np.ndarray
    status_chance: np.ndarray
    stat_target: np.ndarray
    stat_changes: np.ndarray
    stat_chance: np.ndarray
    volatile: np.ndarray
    volatile_chance: np.ndarray
    recoil: np.ndarray
    heal: np.ndarray
    hazard: np.ndarray
    weather: np.ndarray
    terrain: np.ndarray
    hits_min: np.ndarray
    hits_max: np.ndarray


@dataclass
class IDMappings:
    species_to_idx: Dict[str, int]
    move_to_idx: Dict[str, int]
    ability_to_idx: Dict[str, int]
    item_to_idx: Dict[str, int]
    type_to_idx: Dict[str, int]
    species_names: Dict[int, str]
    move_names: Dict[int, str]
    ability_names: Dict[int, str]
    item_names: Dict[int, str]


_cached_by_gen: Dict[int, GameData] = {}
_cached_mappings_by_gen: Dict[int, IDMappings] = {}
_cached_move_effects_by_gen: Dict[int, MoveEffectData] = {}


def get_data_path(gen: int = 9) -> Path:
    """Return extracted data directory for ``gen`` (gen9 uses legacy root path)."""
    env = os.environ.get("POKEPY_DATA_PATH")
    if env:
        base = Path(env).expanduser().resolve()
    else:
        base = Path(__file__).resolve().parent / "extracted"
    if int(gen) == 9:
        return base
    sub = base / f"gen{int(gen)}"
    if sub.exists():
        return sub
    return sub


def load_game_data(
    data_path: Optional[Path] = None, gen: int = 9
) -> GameData:
    global _cached_by_gen
    gen = int(gen)
    if _cached_by_gen.get(gen) is not None and data_path is None:
        return _cached_by_gen[gen]

    if data_path is None:
        data_path = get_data_path(gen)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Pokemon Showdown data not found at {data_path} (gen={gen}). "
            "Set POKEPY_DATA_PATH or run scripts/extract_ps_data.py --gen N."
        )

    def _load(name: str, optional: bool = False) -> Optional[np.ndarray]:
        p = data_path / f"{name}.npy"
        if not p.exists():
            if optional:
                return None
            raise FileNotFoundError(p)
        return np.load(p)

    gd = GameData(
        type_chart=_load("type_chart").astype(np.float32),
        species_base_stats=_load("species_base_stats"),
        species_types=_load("species_types"),
        species_weight=_load("species_weight"),
        species_abilities=_load("species_abilities"),
        move_base_power=_load("move_base_power"),
        move_type=_load("move_type"),
        move_category=_load("move_category"),
        move_priority=_load("move_priority"),
        move_accuracy=_load("move_accuracy"),
        move_pp=_load("move_pp"),
        move_crit_ratio=_load("move_crit_ratio"),
        move_flags=_load("move_flags"),
        move_target=_load("move_target"),
        item_fling_power=_load("item_fling_power", optional=True),
        item_is_berry=_load("item_is_berry", optional=True),
        item_is_choice=_load("item_is_choice", optional=True),
    )

    moves_json = data_path / "moves.json"
    if not moves_json.exists():
        moves_json = data_path.parent / "moves.json"
    if moves_json.exists():
        try:
            with open(moves_json) as _f:
                _moves = json.load(_f)
            acc_arr = gd.move_accuracy
            flags_arr = gd.move_flags
            _FLAG_MUSTPRESSURE = np.uint32(0x80000)
            for _name, _entry in _moves.items():
                _num = int(_entry.get("num", 0))
                if not (0 < _num < acc_arr.shape[0]):
                    continue
                _acc = _entry.get("accuracy", 100)
                if _acc is True:
                    acc_arr[_num] = 127
                _flags = _entry.get("flags") or {}
                if _flags.get("mustpressure"):
                    flags_arr[_num] = np.uint32(
                        int(flags_arr[_num]) | int(_FLAG_MUSTPRESSURE)
                    )
        except Exception:
            pass

    if data_path is None:
        _cached_by_gen[gen] = gd
    else:
        _cached_by_gen[gen] = gd
    return gd


def load_id_mappings(
    data_path: Optional[Path] = None, gen: int = 9
) -> IDMappings:
    """Load string -> int ID mappings from id_mappings.json."""
    global _cached_mappings_by_gen
    gen = int(gen)
    if _cached_mappings_by_gen.get(gen) is not None and data_path is None:
        return _cached_mappings_by_gen[gen]
    if data_path is None:
        data_path = get_data_path(gen)
    with open(data_path / "id_mappings.json") as f:
        data = json.load(f)
    m = IDMappings(
        species_to_idx=data["species_id_to_idx"],
        move_to_idx=data["move_id_to_idx"],
        ability_to_idx=data["ability_id_to_idx"],
        item_to_idx=data["item_id_to_idx"],
        type_to_idx=data["type_to_idx"],
        species_names={int(k): v for k, v in data["species_names"].items()},
        move_names={int(k): v for k, v in data["move_names"].items()},
        ability_names={int(k): v for k, v in data["ability_names"].items()},
        item_names={int(k): v for k, v in data["item_names"].items()},
    )
    _cached_mappings_by_gen[gen] = m
    return m


def load_move_effect_data(
    move_to_idx: Optional[Dict[str, int]] = None,
    gen: int = 9,
) -> MoveEffectData:
    """Build MoveEffectData from pokepy.data.move_effects.MOVE_EFFECTS."""
    global _cached_move_effects_by_gen
    gen = int(gen)
    if _cached_move_effects_by_gen.get(gen) is not None and move_to_idx is None:
        return _cached_move_effects_by_gen[gen]
    if move_to_idx is None:
        move_to_idx = load_id_mappings(gen=gen).move_to_idx

    from pokepy.data.move_effects import create_move_effect_arrays

    num_moves = max(move_to_idx.values()) + 1 if move_to_idx else 1000
    arrays = create_move_effect_arrays(move_to_idx, num_moves)
    me = MoveEffectData(
        effect_type=np.asarray(arrays[0]),
        status=np.asarray(arrays[1]),
        status_chance=np.asarray(arrays[2]),
        stat_target=np.asarray(arrays[3]),
        stat_changes=np.asarray(arrays[4]),
        stat_chance=np.asarray(arrays[5]),
        volatile=np.asarray(arrays[6]),
        volatile_chance=np.asarray(arrays[7]),
        recoil=np.asarray(arrays[8]),
        heal=np.asarray(arrays[9]),
        hazard=np.asarray(arrays[10]),
        weather=np.asarray(arrays[11]),
        terrain=np.asarray(arrays[12]),
        hits_min=np.asarray(arrays[13]),
        hits_max=np.asarray(arrays[14]),
    )
    _cached_move_effects_by_gen[gen] = me
    _apply_gen_move_effect_patches(me, gen, move_to_idx)
    return me


def _apply_gen_move_effect_patches(
    me: MoveEffectData, gen: int, move_to_idx: Dict[str, int]
) -> None:
    """Patch move-effect tables for generation-specific Showdown mod deltas."""
    if int(gen) != 1:
        return
    from pokepy.data.move_effects import STAT_SPA, STAT_SPD

    psychic_idx = move_to_idx.get("psychic")
    if psychic_idx is not None:
        me.stat_target[psychic_idx] = 1
        me.stat_chance[psychic_idx] = 33
        me.stat_changes[psychic_idx, :] = 0
        me.stat_changes[psychic_idx, STAT_SPA] = -1
        me.stat_changes[psychic_idx, STAT_SPD] = -1
