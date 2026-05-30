"""Generation-specific mechanic and control-flow branches."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from pokepy.core.gen_profile import GenProfile, profile_for_gen
from pokepy.utils.gen5_prng import Gen5PRNG


def quick_claw_roll(profile: GenProfile, gen5_prng: Gen5PRNG) -> Optional[bool]:
    """End-of-turn Quick Claw roll (gen2/gen3 only in Showdown)."""
    if profile.gen == 2:
        return gen5_prng.random(256) < 60
    if profile.gen == 3:
        return gen5_prng.random(5) < 1
    return None


def modify_damage_order(profile: GenProfile) -> str:
    """Return damage pipeline ordering label for this gen."""
    if profile.gen <= 3:
        return "gen3"
    if profile.gen == 4:
        return "gen4"
    return "modern"


def hit_step_order(profile: GenProfile) -> tuple:
    """Return ordered hit-step names (accuracy, protect, type immunity, etc.)."""
    if profile.gen <= 4:
        return ("try_primary_hit", "secondary", "spread")
    return ("try_primary_hit", "secondary", "spread", "dynamax")


def sleep_turn_range(profile: GenProfile) -> tuple:
    if profile.gen == 1:
        return (1, 7)
    if profile.gen == 2:
        return (2, 8)
    if profile.gen == 3:
        return (1, 4)
    return (1, 3)


def crit_multiplier(profile: GenProfile) -> float:
    return float(profile.crit_damage_mult)


def full_para_roll(profile: GenProfile, gen5_prng: Gen5PRNG) -> bool:
    return gen5_prng.random(profile.full_para_denom) < profile.full_para_num


def powder_blocks_grass(profile: GenProfile, defender_types: tuple) -> bool:
    if not profile.powder_grass_immune:
        return False
    return 4 in defender_types  # TYPE_GRASS index in modern tables


def apply_gen_end_turn_rolls(
    battle: np.ndarray,
    profile: GenProfile,
    gen5_prng: Gen5PRNG,
) -> None:
    """Gen-specific end-of-turn PRNG draws (Quick Claw etc.)."""
    _ = battle
    quick_claw_roll(profile, gen5_prng)


def damage_fn_for_gen(gen: int):
    """Return the generation-appropriate calc_damage helper."""
    g = int(gen)
    if g == 1:
        from pokepy.mechanics.damage_gen1 import calc_damage_gen1

        return calc_damage_gen1
    if g == 2:
        from pokepy.mechanics.damage_gen2 import calc_damage_gen2

        return calc_damage_gen2
    from pokepy.mechanics.damage_gen9 import calc_damage_gen9

    return calc_damage_gen9


def profile_for_state(state, default_gen: int = 9) -> GenProfile:
    gen = getattr(state, "format_gen", None)
    if gen is not None:
        return profile_for_gen(int(gen))
    return profile_for_gen(default_gen)


__all__ = [
    "apply_gen_end_turn_rolls",
    "crit_multiplier",
    "damage_fn_for_gen",
    "full_para_roll",
    "hit_step_order",
    "modify_damage_order",
    "powder_blocks_grass",
    "profile_for_state",
    "quick_claw_roll",
    "sleep_turn_range",
]
