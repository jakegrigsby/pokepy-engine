"""Per-generation battle configuration for the shared modern core."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Literal, Optional

from pokepy.core.constants import (
    FORMAT_GEN1OU,
    FORMAT_GEN9OU,
)

PhysSpecMode = Literal["type", "move"]


@dataclass(frozen=True)
class GenProfile:
    """Feature flags and parity-critical constants that vary by generation."""

    gen: int
    format_id: int
    battle_format: str
    phys_spec_mode: PhysSpecMode
    crit_damage_mult: float
    # Index by clamped crit stage (1..crit_stage_max). Index 0 unused (None).
    crit_stage_denoms: tuple
    crit_stage_max: int = 4
    has_abilities: bool = True
    has_items: bool = True
    has_natures_evs: bool = True
    has_tera: bool = False
    has_terrain: bool = False
    has_teampreview: bool = False
    enabled_hazards: frozenset = frozenset()
    # gen6+ ability-set weather lasts 5/8 turns; gen5 and earlier is permanent.
    ability_weather_limited: bool = False
    type_chart_name: str = "type_chart"
    data_subdir: str = ""
    # Showdown gen1-2: randomChance(63, 256); gen3+: randomChance(1, 4).
    full_para_num: int = 1
    full_para_denom: int = 4
    # Powder/spore Grass immunity (Showdown battle-actions.ts: gen >= 6).
    powder_grass_immune: bool = True

    def crit_denom_for_stage(self, crit_stage: int) -> int:
        stage = max(1, min(self.crit_stage_max, int(crit_stage)))
        denom = self.crit_stage_denoms[stage]
        if denom is None:
            raise ValueError(f"No crit denom for stage {stage} in gen {self.gen}")
        return int(denom)

    def hazard_enabled(self, hazard_name: str) -> bool:
        return hazard_name in self.enabled_hazards


def crit_denom_from_table(crit_stage: int, profile: GenProfile) -> int:
    """Map clamped crit stage to Showdown randomChance denominator."""
    if crit_stage >= profile.crit_stage_max:
        return profile.crit_denom_for_stage(profile.crit_stage_max)
    return profile.crit_denom_for_stage(crit_stage)


# gen9: sim/battle-actions.ts gen>=7 -> [0, 24, 8, 2, 1], crit mult 1.5
GEN9_PROFILE = GenProfile(
    gen=9,
    format_id=FORMAT_GEN9OU,
    battle_format="gen9ou",
    phys_spec_mode="move",
    crit_damage_mult=1.5,
    crit_stage_denoms=(None, 24, 8, 2, 1),
    crit_stage_max=4,
    has_abilities=True,
    has_items=True,
    has_natures_evs=True,
    has_tera=True,
    has_terrain=True,
    has_teampreview=True,
    enabled_hazards=frozenset({"spikes", "stealthrock", "toxicspikes", "stickyweb"}),
    ability_weather_limited=True,
    type_chart_name="type_chart",
    data_subdir="",
)

# gen4: gen<=5 critMult [0,16,8,4,3,2], crit damage x2
GEN4_PROFILE = GenProfile(
    gen=4,
    format_id=FORMAT_GEN9OU,  # distinct FORMAT_GEN4OU not yet in constants
    battle_format="gen4ou",
    phys_spec_mode="move",
    crit_damage_mult=2.0,
    crit_stage_denoms=(None, 16, 8, 4, 3),
    crit_stage_max=4,
    has_abilities=True,
    has_items=True,
    has_natures_evs=True,
    has_tera=False,
    has_terrain=False,
    has_teampreview=False,
    enabled_hazards=frozenset({"spikes", "stealthrock", "toxicspikes"}),
    ability_weather_limited=False,
    type_chart_name="type_chart",
    data_subdir="gen4",
    powder_grass_immune=False,
)

# gen3: type-based phys/spec split, no SR
GEN3_PROFILE = GenProfile(
    gen=3,
    format_id=FORMAT_GEN9OU,
    battle_format="gen3ou",
    phys_spec_mode="type",
    crit_damage_mult=2.0,
    crit_stage_denoms=(None, 16, 8, 4, 3),
    crit_stage_max=4,
    has_abilities=True,
    has_items=True,
    has_natures_evs=True,
    has_tera=False,
    has_terrain=False,
    has_teampreview=False,
    enabled_hazards=frozenset({"spikes"}),
    ability_weather_limited=False,
    type_chart_name="type_chart",
    data_subdir="gen3",
    powder_grass_immune=False,
)

# gen2: items, no abilities
GEN2_PROFILE = GenProfile(
    gen=2,
    format_id=FORMAT_GEN9OU,
    battle_format="gen2ou",
    phys_spec_mode="type",
    crit_damage_mult=2.0,
    crit_stage_denoms=(None, 16, 8, 4, 3),
    crit_stage_max=4,
    has_abilities=False,
    has_items=True,
    has_natures_evs=False,
    has_tera=False,
    has_terrain=False,
    has_teampreview=False,
    enabled_hazards=frozenset({"spikes"}),
    ability_weather_limited=False,
    type_chart_name="type_chart",
    data_subdir="gen2",
    full_para_num=63,
    full_para_denom=256,
    powder_grass_immune=False,
)

# gen1: combined Special, type split, speed crit handled in damage_gen1
GEN1_PROFILE = GenProfile(
    gen=1,
    format_id=FORMAT_GEN1OU,
    battle_format="gen1ou",
    phys_spec_mode="type",
    crit_damage_mult=2.0,
    crit_stage_denoms=(None, 16, 8, 4, 3),
    crit_stage_max=4,
    has_abilities=False,
    has_items=False,
    has_natures_evs=False,
    has_tera=False,
    has_terrain=False,
    has_teampreview=False,
    enabled_hazards=frozenset(),
    type_chart_name="type_chart",
    data_subdir="gen1",
    full_para_num=63,
    full_para_denom=256,
    powder_grass_immune=False,
)

PROFILE_REGISTRY: Dict[int, GenProfile] = {
    1: GEN1_PROFILE,
    2: GEN2_PROFILE,
    3: GEN3_PROFILE,
    4: GEN4_PROFILE,
    9: GEN9_PROFILE,
}


def profile_for_gen(gen: int) -> GenProfile:
    try:
        return PROFILE_REGISTRY[int(gen)]
    except KeyError as exc:
        raise KeyError(f"No GenProfile registered for gen {gen}") from exc


def parse_battle_format(battle_format: str) -> int:
    """Parse ``gen4ou`` -> 4. Raises ValueError if unknown."""
    m = re.match(r"^gen(\d+)", str(battle_format).lower())
    if not m:
        raise ValueError(f"Cannot parse generation from format {battle_format!r}")
    gen = int(m.group(1))
    if gen not in PROFILE_REGISTRY:
        raise ValueError(
            f"Generation {gen} from format {battle_format!r} is not registered"
        )
    return gen


def profile_for_format(battle_format: str) -> GenProfile:
    return profile_for_gen(parse_battle_format(battle_format))


def registered_gens() -> frozenset:
    return frozenset(PROFILE_REGISTRY.keys())


def is_format_supported(battle_format: str) -> bool:
    try:
        parse_battle_format(battle_format)
        return True
    except ValueError:
        return False
