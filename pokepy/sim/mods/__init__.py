"""Per-generation handler registration before each turn."""

from __future__ import annotations

from pokepy.core.gen_profile import GenProfile
from pokepy.sim import conditions
from pokepy.sim.mods import gen1, gen2, gen3, gen4


def apply_mod(profile: GenProfile) -> None:
    """Register condition/ability/item/field/residual handlers for this gen."""
    conditions.register_all(profile)
    if profile.gen == 1:
        gen1.apply(profile)
    elif profile.gen == 2:
        gen2.apply(profile)
    elif profile.gen == 3:
        gen3.apply(profile)
    elif profile.gen == 4:
        gen4.apply(profile)

    if profile.has_items:
        from pokepy.sim import items

        items.register_all(profile)
    if profile.has_abilities:
        from pokepy.sim import abilities

        abilities.register_all(profile)

    from pokepy.sim import field, residual

    field.register_all(profile)
    # Per-mon misc residual bundle (salt cure, partial trap, aqua ring, yawn,
    # weather-ability healing). Registered for all gens; each sub-effect
    # self-gates so it no-ops when the volatile/ability is absent.
    residual.register_all(profile)
