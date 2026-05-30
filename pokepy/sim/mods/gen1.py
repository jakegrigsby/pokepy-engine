"""Gen1 mod — data/mods/gen1/conditions.ts overrides."""

from pokepy.core.gen_profile import GenProfile
from pokepy.sim.conditions import register_gen1


def apply(profile: GenProfile) -> None:
    register_gen1()
