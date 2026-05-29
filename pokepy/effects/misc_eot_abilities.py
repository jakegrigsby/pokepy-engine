"""End-of-turn ability effects: Solar Power, Bad Dreams.

Port of multi_format_fast_env.py lines 5165-5200. Run during the EOT
pipeline before residual status damage.
"""

from __future__ import annotations

import numpy as np

from pokepy.core.constants import (
    OFF_FIELD,
    F_WEATHER,
    WEATHER_SUN,
    STATUS_SLEEP,
    ABILITY_MAGIC_GUARD,
    ABILITY_SOLAR_POWER,
)
from pokepy.core.bitpack import get_status

ABILITY_BAD_DREAMS = 123


def apply_misc_eot_abilities(battle: np.ndarray, p0_off: int, p1_off: int) -> None:
    """Solar Power EOT damage, Bad Dreams EOT damage. Mutates battle."""
    p0_off = int(p0_off)
    p1_off = int(p1_off)
    weather = int(battle[OFF_FIELD + F_WEATHER])
    in_sun = weather == WEATHER_SUN

    # Solar Power: -1/8 max HP each turn in Sun. Air Lock / Cloud Nine
    # suppress the weather entirely (they're active-mon-scoped, so any
    # active Air Lock/Cloud Nine kills Solar Power's sun damage too).
    # Magic Guard is mutually exclusive with Solar Power (only one ability
    # slot), so the prior `ab == ABILITY_MAGIC_GUARD` check was dead code.
    # Utility Umbrella blocks sun/rain effects on the holder.
    from pokepy.core.constants import (
        ABILITY_AIR_LOCK,
        ABILITY_CLOUD_NINE,
        OFF_SIDE0 as _OS0,
        OFF_SIDE1 as _OS1,
        OFF_META as _OM,
        M_ACTIVE0 as _MA0,
        M_ACTIVE1 as _MA1,
        POKEMON_SIZE as _PS,
        ITEM_UTILITY_UMBRELLA as _UMB,
    )

    _a0 = int(battle[_OM + _MA0])
    _a1 = int(battle[_OM + _MA1])
    _ab0_sp = int(battle[_OS0 + _a0 * _PS + 5])
    _ab1_sp = int(battle[_OS1 + _a1 * _PS + 5])
    _weather_sup = _ab0_sp in (ABILITY_AIR_LOCK, ABILITY_CLOUD_NINE) or _ab1_sp in (
        ABILITY_AIR_LOCK,
        ABILITY_CLOUD_NINE,
    )
    for poff in (p0_off, p1_off):
        ab = int(battle[poff + 5])
        if ab != ABILITY_SOLAR_POWER or not in_sun or _weather_sup:
            continue
        hp = int(battle[poff + 1])
        if hp <= 0:
            continue
        if int(battle[poff + 6]) == _UMB:
            continue
        max_hp = int(battle[poff + 2])
        dmg = max(1, max_hp // 8)
        battle[poff + 1] = np.int16(max(0, hp - dmg))

    # Bad Dreams: opponent has Bad Dreams, this mon is asleep → -1/8 max HP
    for poff, opp_off in ((p0_off, p1_off), (p1_off, p0_off)):
        opp_ab = int(battle[opp_off + 5])
        if opp_ab != ABILITY_BAD_DREAMS:
            continue
        my_status = get_status(int(battle[poff + 12]))
        if my_status != STATUS_SLEEP:
            continue
        my_hp = int(battle[poff + 1])
        if my_hp <= 0:
            continue
        my_ab = int(battle[poff + 5])
        if my_ab == ABILITY_MAGIC_GUARD:
            continue
        max_hp = int(battle[poff + 2])
        dmg = max(1, max_hp // 8)
        battle[poff + 1] = np.int16(max(0, my_hp - dmg))
