"""Standalone .npy game data loader.

Reads Pokemon Showdown data tables bundled with pokepy at
`pokepy/data/extracted/`. Override the location with the
`POKEPY_DATA_PATH` environment variable. Field names match the disk
filenames so the loader is also a documentation of which tables exist.
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

_cached: Optional[GameData] = None
_cached_mappings: Optional[IDMappings] = None
_cached_move_effects: Optional[MoveEffectData] = None

def get_data_path() -> Path:
    """Pokemon Showdown data is bundled at pokepy/data/extracted/.

    Override with the POKEPY_DATA_PATH environment variable.
    """
    env = os.environ.get("POKEPY_DATA_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent / "extracted"

def load_game_data(data_path: Optional[Path] = None) -> GameData:
    global _cached
    if _cached is not None and data_path is None:
        return _cached

    if data_path is None:
        data_path = get_data_path()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Pokemon Showdown data not found at {data_path}. "
            "Set POKEPY_DATA_PATH or reinstall pokepy with bundled data."
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

    # Patch metadata that the extracted numpy tables still miss. Today that is:
    # - `accuracy: true` moves, which the extractor clips to 100 even though
    #   pokepy's hit logic expects >100 to mean truly never-miss.
    # - Showdown's `mustpressure` flag, which governs extra PP loss against
    #   Pressure for moves like Stealth Rock / Spikes.
    moves_json = data_path.parent / "moves.json"
    if moves_json.exists():
        try:
            with open(moves_json) as _f:
                _moves = json.load(_f)
            acc_arr = gd.move_accuracy
            flags_arr = gd.move_flags
            # `mustpressure` is not yet encoded in the extracted move_flags.npy
            # table, so backfill it here from moves.json using an unused bit.
            _FLAG_MUSTPRESSURE = np.uint32(0x80000)
            for _name, _entry in _moves.items():
                _num = int(_entry.get("num", 0))
                if not (0 < _num < acc_arr.shape[0]):
                    continue
                _acc = _entry.get("accuracy", 100)
                if _acc is True:
                    # Move accuracy is int8; 127 is the max value that still
                    # compares >100 in the engine.
                    acc_arr[_num] = 127
                _flags = _entry.get("flags") or {}
                if _flags.get("mustpressure"):
                    flags_arr[_num] = np.uint32(int(flags_arr[_num]) | int(_FLAG_MUSTPRESSURE))
        except Exception:
            pass

    if data_path is None:
        _cached = gd
    return gd

def load_id_mappings(data_path: Optional[Path] = None) -> IDMappings:
    """Load string -> int ID mappings from id_mappings.json."""
    global _cached_mappings
    if _cached_mappings is not None and data_path is None:
        return _cached_mappings
    if data_path is None:
        data_path = get_data_path()
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
    if data_path is None:
        _cached_mappings = m
    return m

def load_move_effect_data(move_to_idx: Optional[Dict[str, int]] = None) -> MoveEffectData:
    """Build MoveEffectData from pokepy.data.move_effects.MOVE_EFFECTS."""
    global _cached_move_effects
    if _cached_move_effects is not None and move_to_idx is None:
        return _cached_move_effects
    if move_to_idx is None:
        move_to_idx = load_id_mappings().move_to_idx

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
    _cached_move_effects = me
    return me
