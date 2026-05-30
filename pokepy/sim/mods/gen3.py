"""Gen3 mod — sleep onBeforeMove decrement, hazards."""

from pokepy.core.gen_profile import GenProfile
from pokepy.sim import dispatch
from pokepy.sim.conditions import (
    on_modern_slp_before_move,
    register_all as register_conditions,
)


def apply(profile: GenProfile) -> None:
    register_conditions(profile)
    fmt = int(profile.format_id)
    handlers = dispatch.get_handlers(fmt, "slp")
    handlers.pop(dispatch.BeforeMove, None)
    dispatch.register(
        fmt, "slp", dispatch.BeforeMove, on_modern_slp_before_move, priority=10
    )
