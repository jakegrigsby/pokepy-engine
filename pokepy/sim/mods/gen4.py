"""Gen4 mod — gen3 sleep + Early Bird handled in effects/status_apply."""

from pokepy.core.gen_profile import GenProfile
from pokepy.sim.mods import gen3


def apply(profile: GenProfile) -> None:
    gen3.apply(profile)
