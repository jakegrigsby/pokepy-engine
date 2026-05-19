"""Stat calculation matching Pokemon Showdown formulas.

calc_stat_gen1 / calc_stat_modern are exact integer ports of the canonical
Showdown formulas. Boost multipliers use the (2 + b)/2 / 2/(2 - b) form,
which produces identical floats to the lookup-table version.
"""

from __future__ import annotations

import math
import numpy as np

def calc_stat_gen1(base, level, is_hp: bool) -> np.int16:
    """Gen 1 stat (assumes 15 DVs, 65535 stat exp)."""
    base_i = int(base)
    lvl_i = int(level)
    dv = 15
    stat_exp = 65535
    stat_exp_bonus = int(math.floor(math.sqrt(stat_exp) / 4))
    if is_hp:
        stat = (((base_i + dv) * 2 + stat_exp_bonus) * lvl_i // 100) + lvl_i + 10
    else:
        stat = (((base_i + dv) * 2 + stat_exp_bonus) * lvl_i // 100) + 5
    return np.int16(max(0, min(32767, stat)))

def calc_stat_modern(base, level, iv: int, ev: int, is_hp: bool, nature_mult: float = 1.0) -> np.int16:
    """Modern (Gen 3+) stat formula."""
    base_i = int(base)
    lvl_i = int(level)
    iv_i = int(iv)
    ev_i = int(ev)
    if is_hp:
        stat = (2 * base_i + iv_i + ev_i // 4) * lvl_i // 100 + lvl_i + 10
    else:
        stat = (2 * base_i + iv_i + ev_i // 4) * lvl_i // 100 + 5
        # Match Showdown: float multiply, then int16 cast (truncates toward zero).
        stat = int(np.int16(np.float32(stat) * np.float32(nature_mult)))
    return np.int16(max(0, min(32767, stat)))

def get_boost_multiplier(boost) -> np.float32:
    """Stat multiplier from boost stage (-6..+6)."""
    b = max(-6, min(6, int(boost)))
    if b >= 0:
        return np.float32((2.0 + b) / 2.0)
    return np.float32(2.0 / (2.0 - b))
