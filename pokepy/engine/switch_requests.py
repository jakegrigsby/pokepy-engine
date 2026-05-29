"""Decision-point requests emitted by the Gen9 battle turn generator."""

from __future__ import annotations

from typing import Dict, Iterable

from pokepy.core.constants import M_ACTIVE0, M_ACTIVE1, OFF_META, OFF_SIDE0, OFF_SIDE1
from pokepy.effects.auto_switch import auto_switch as fx_auto_switch


class SwitchRequest:
    """Policy must choose replacement slot(s) before the turn continues.

    ``sides`` lists physical pokepy sides (0 or 1) that need a choice at this
    moment. The driver ``send``s back a dict ``{side: team_slot_0_5}``.
    """

    __slots__ = ("sides",)

    def __init__(self, sides: Iterable[int]):
        self.sides = tuple(int(s) for s in sides)


def slot_from_pokepy_action(action: int) -> int:
    return max(0, min(5, int(action) - 4))


def pokepy_action_from_slot(slot: int) -> int:
    return int(slot) + 4


def active_slot_for_side(battle, side: int) -> int:
    meta = M_ACTIVE0 if int(side) == 0 else M_ACTIVE1
    return int(battle[OFF_META + meta])


def auto_switch_slot(
    battle,
    side: int,
    current_active: int,
    *,
    order=None,
) -> int:
    side_base = OFF_SIDE0 if int(side) == 0 else OFF_SIDE1
    kwargs = {}
    if order is not None:
        kwargs["order"] = order
    return int(
        fx_auto_switch(
            battle,
            side_base,
            int(current_active),
            True,
            **kwargs,
        )
    )


def resolve_switch_choices_sync(
    state,
    battle,
    request: SwitchRequest,
    *,
    side_order0,
    side_order1,
    resolve_mid_turn_switch0=None,
) -> Dict[int, int]:
    """Back-compat resolver used by the synchronous ``step_battle_gen9`` wrapper."""
    choices: Dict[int, int] = {}
    for side in request.sides:
        active = active_slot_for_side(battle, side)
        if side == 0 and resolve_mid_turn_switch0 is not None:
            choices[0] = slot_from_pokepy_action(int(resolve_mid_turn_switch0(state)))
        else:
            order = side_order0 if side == 0 else side_order1
            choices[side] = auto_switch_slot(battle, side, active, order=order)
    return choices
