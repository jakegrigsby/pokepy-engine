"""Gen2 mod — full para 63/256, endTurn Quick Claw pre-roll in turn.py."""

from pokepy.core.gen_profile import GenProfile
from pokepy.sim import dispatch
from pokepy.sim.conditions import (
    on_gen1_par_before_move,
    register_all as register_conditions,
)


def apply(profile: GenProfile) -> None:
    register_conditions(profile)
    fmt = int(profile.format_id)
    dispatch.get_handlers(fmt, "par").pop(dispatch.BeforeMove, None)
    dispatch.register(
        fmt, "par", dispatch.BeforeMove, on_gen1_par_before_move, priority=2
    )
