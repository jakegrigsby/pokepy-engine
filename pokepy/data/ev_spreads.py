"""EV spread → nature inference (standalone EV-spread inference table).

Each entry: (EVs, nature_mods).
- EVs is [HP, Atk, Def, SpA, SpD, Spe] summing to ≤508.
- nature_mods is the multiplier per stat (HP, Atk, Def, SpA, SpD, Spe).
  Typical natures: +Spe -SpA = Jolly = (1.0, 1.0, 1.0, 0.9, 1.0, 1.1).

The 50 spreads cover the most common gen9ou EV+nature combos. We use the
EV pattern to *guess* nature for teams loaded from npz (which only stores
raw EVs, not natures).
"""

from __future__ import annotations

import numpy as np

# 50 (EVs, nature_mods) tuples — top gen9ou patterns. Order matters: when
# matching by closest EV pattern, ties are broken by index (lower is
# preferred since the list is sorted by frequency).
_SPREAD_DATA = [
    ([0, 252, 0, 0, 4, 252], [1.0, 1.0, 1.0, 0.9, 1.0, 1.1]),    #  0: Jolly phys sweeper
    ([0, 0, 0, 252, 4, 252], [1.0, 0.9, 1.0, 1.0, 1.0, 1.1]),    #  1: Timid spec sweeper
    ([0, 252, 0, 0, 4, 252], [1.0, 1.1, 1.0, 0.9, 1.0, 1.0]),    #  2: Adamant phys sweeper
    ([0, 252, 4, 0, 0, 252], [1.0, 1.0, 1.0, 0.9, 1.0, 1.1]),    #  3: Jolly phys
    ([0, 0, 0, 252, 4, 252], [1.0, 0.9, 1.0, 1.1, 1.0, 1.0]),    #  4: Modest spec sweeper
    ([0, 252, 4, 0, 0, 252], [1.0, 1.1, 1.0, 0.9, 1.0, 1.0]),    #  5: Adamant phys
    ([252, 0, 252, 0, 4, 0], [1.0, 0.9, 1.1, 1.0, 1.0, 1.0]),    #  6: Bold phys wall
    ([252, 0, 4, 0, 252, 0], [1.0, 1.0, 1.0, 0.9, 1.1, 1.0]),    #  7: Careful spec wall
    ([0, 0, 4, 252, 0, 252], [1.0, 0.9, 1.0, 1.0, 1.0, 1.1]),    #  8: Timid spec
    ([252, 252, 0, 0, 4, 0], [1.0, 1.1, 1.0, 0.9, 1.0, 1.0]),    #  9: Adamant bulky phys
    ([252, 0, 0, 252, 4, 0], [1.0, 0.9, 1.0, 1.1, 1.0, 1.0]),    # 10: Modest bulky spec
    ([252, 0, 252, 0, 4, 0], [1.0, 1.0, 1.1, 0.9, 1.0, 1.0]),    # 11: Impish phys wall
    ([0, 4, 0, 252, 0, 252], [1.0, 1.0, 1.0, 1.0, 0.9, 1.1]),    # 12: Naive mixed
    ([252, 4, 0, 0, 0, 252], [1.0, 1.0, 1.0, 0.9, 1.0, 1.1]),    # 13: Jolly bulky speed
    ([4, 252, 0, 0, 0, 252], [1.0, 1.0, 1.0, 0.9, 1.0, 1.1]),    # 14: Jolly fast phys
    ([0, 0, 4, 252, 0, 252], [1.0, 0.9, 1.0, 1.1, 1.0, 1.0]),    # 15: Modest spec
    ([248, 0, 252, 0, 8, 0], [1.0, 1.0, 1.1, 0.9, 1.0, 1.0]),    # 16: Impish tank
    ([4, 0, 0, 252, 0, 252], [1.0, 0.9, 1.0, 1.0, 1.0, 1.1]),    # 17: Timid fast spec
    ([248, 252, 0, 0, 8, 0], [1.0, 1.1, 1.0, 0.9, 1.0, 1.0]),    # 18: Adamant bulky
    ([0, 0, 124, 132, 0, 252], [1.0, 0.9, 1.0, 1.0, 1.0, 1.1]),  # 19: Timid balanced
    ([252, 0, 16, 0, 240, 0], [1.0, 1.0, 1.0, 1.0, 1.1, 0.9]),   # 20: Sassy spdef
    ([252, 4, 252, 0, 0, 0], [1.0, 1.0, 1.1, 0.9, 1.0, 1.0]),    # 21: Impish wall
    ([248, 0, 252, 0, 8, 0], [1.0, 0.9, 1.1, 1.0, 1.0, 1.0]),    # 22: Bold tank
    ([252, 0, 252, 4, 0, 0], [1.0, 0.9, 1.1, 1.0, 1.0, 1.0]),    # 23: Bold wall
    ([252, 4, 0, 0, 252, 0], [1.0, 1.0, 1.0, 0.9, 1.1, 1.0]),    # 24: Careful spdef
    ([252, 0, 252, 0, 4, 0], [1.0, 1.0, 1.1, 1.0, 1.0, 0.9]),    # 25: Relaxed wall
    ([252, 0, 4, 0, 0, 252], [1.0, 0.9, 1.0, 1.0, 1.0, 1.1]),    # 26: Timid support
    ([248, 0, 248, 0, 0, 12], [1.0, 0.9, 1.1, 1.0, 1.0, 1.0]),   # 27: Bold tank
    ([252, 0, 0, 4, 252, 0], [1.0, 0.9, 1.0, 1.0, 1.1, 1.0]),    # 28: Calm spdef
    ([252, 0, 88, 0, 0, 168], [1.0, 1.0, 1.0, 0.9, 1.0, 1.1]),   # 29: Jolly defensive
    ([252, 0, 204, 0, 0, 52], [1.0, 0.9, 1.1, 1.0, 1.0, 1.0]),   # 30: Bold balanced
    ([252, 0, 4, 0, 252, 0], [1.0, 1.0, 1.0, 1.0, 1.1, 0.9]),    # 31: Sassy spdef
    ([4, 252, 0, 0, 0, 252], [1.0, 1.1, 1.0, 0.9, 1.0, 1.0]),    # 32: Adamant fast
    ([252, 0, 4, 0, 252, 0], [1.0, 0.9, 1.0, 1.0, 1.1, 1.0]),    # 33: Calm spdef
    ([248, 0, 0, 252, 8, 0], [1.0, 0.9, 1.0, 1.1, 1.0, 1.0]),    # 34: Modest bulky spec
    ([248, 0, 8, 0, 252, 0], [1.0, 1.0, 1.0, 1.0, 1.1, 0.9]),    # 35: Sassy spdef
    ([248, 0, 252, 8, 0, 0], [1.0, 0.9, 1.1, 1.0, 1.0, 1.0]),    # 36: Bold tank
    ([248, 0, 8, 0, 252, 0], [1.0, 1.0, 1.0, 0.9, 1.1, 1.0]),    # 37: Careful spdef
    ([252, 0, 252, 0, 0, 4], [1.0, 0.9, 1.1, 1.0, 1.0, 1.0]),    # 38: Bold wall
    ([252, 0, 0, 4, 0, 252], [1.0, 0.9, 1.0, 1.0, 1.0, 1.1]),    # 39: Timid support
    ([0, 252, 0, 4, 0, 252], [1.0, 1.0, 1.0, 1.0, 0.9, 1.1]),    # 40: Naive mixed
    ([248, 8, 252, 0, 0, 0], [1.0, 1.0, 1.1, 0.9, 1.0, 1.0]),    # 41: Impish wall
    ([248, 0, 0, 8, 0, 252], [1.0, 0.9, 1.0, 1.0, 1.0, 1.1]),    # 42: Timid support
    ([224, 32, 0, 0, 0, 252], [1.0, 1.0, 1.0, 0.9, 1.0, 1.1]),   # 43: Jolly bulky
    ([80, 0, 0, 252, 0, 176], [1.0, 0.9, 1.0, 1.1, 1.0, 1.0]),   # 44: Modest fast spec
    ([248, 0, 8, 0, 0, 252], [1.0, 0.9, 1.0, 1.0, 1.0, 1.1]),    # 45: Timid bulky
    ([252, 0, 160, 0, 0, 96], [1.0, 0.9, 1.1, 1.0, 1.0, 1.0]),   # 46: Bold speed
    ([244, 0, 252, 0, 12, 0], [1.0, 1.0, 1.1, 0.9, 1.0, 1.0]),   # 47: Impish tank
    ([248, 0, 244, 0, 0, 16], [1.0, 0.9, 1.1, 1.0, 1.0, 1.0]),   # 48: Bold tank
    ([112, 252, 0, 0, 0, 144], [1.0, 1.1, 1.0, 0.9, 1.0, 1.0]),  # 49: Adamant balanced
]

EV_SPREADS_ARRAY = np.array([d[0] for d in _SPREAD_DATA], dtype=np.int16)
NATURE_MODS_ARRAY = np.array([d[1] for d in _SPREAD_DATA], dtype=np.float32)

def infer_nature_from_evs(evs: np.ndarray) -> np.ndarray:
    """Find the closest EV spread by L1 distance and return its nature mods.
    `evs` is a [6] array (HP/Atk/Def/SpA/SpD/Spe). Returns a [6] float array.
    """
    evs_arr = np.asarray(evs, dtype=np.int16).reshape(6)
    diffs = np.abs(EV_SPREADS_ARRAY - evs_arr).sum(axis=1)
    idx = int(np.argmin(diffs))
    return NATURE_MODS_ARRAY[idx].copy()
