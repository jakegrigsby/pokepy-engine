"""Battle action queue — mirrors sim/battle-queue.ts order values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

# battle-queue.ts resolveAction order constants
ORDER_TEAM = 1
ORDER_START = 2
ORDER_INSTANT_SWITCH = 3
ORDER_BEFORE_TURN = 4
ORDER_BEFORE_TURN_MOVE = 5
ORDER_RUN_SWITCH = 101
ORDER_SWITCH = 103
ORDER_MOVE = 200
ORDER_RESIDUAL = 300


@dataclass
class Action:
    choice: str
    order: int = ORDER_MOVE
    priority: int = 0
    fractional_priority: float = 0.0
    speed: int = 0
    sub_order: int = 0
    effect_order: int = 0
    side: int = 0
    move_slot: int = -1
    move_id: int = -1
    target_side: int = -1
    switch_slot: int = -1
    index: int = 0


class BattleQueue:
    """Action queue with Showdown sort semantics."""

    def __init__(self) -> None:
        self.list: List[Action] = []

    def add_choice(self, action: Action) -> None:
        self.list.append(action)

    def insert_choice(self, action: Action, index: int) -> None:
        self.list.insert(index, action)

    def resolve_action(self, choice: str, **kwargs) -> Action:
        return Action(choice=choice, **kwargs)

    def clear(self) -> None:
        self.list.clear()

    def shift(self) -> Optional[Action]:
        if not self.list:
            return None
        return self.list.pop(0)

    @staticmethod
    def compare_priority(a: Action, b: Action) -> int:
        """battle.ts:404-410 comparePriority."""
        ao = a.order if a.order else 4294967296
        bo = b.order if b.order else 4294967296
        if ao != bo:
            return ao - bo
        if a.priority != b.priority:
            return b.priority - a.priority
        fa = a.fractional_priority or 0.0
        fb = b.fractional_priority or 0.0
        if fa != fb:
            return int((fb - fa) * 1000)
        if a.speed != b.speed:
            return b.speed - a.speed
        if a.sub_order != b.sub_order:
            return a.sub_order - b.sub_order
        if a.effect_order != b.effect_order:
            return a.effect_order - b.effect_order
        return 0

    def sort(self, speed_sort: Callable[[List[Action]], None]) -> None:
        """battle-queue.ts:413-416 — speedSort only (comparePriority inside)."""
        speed_sort(self.list)

    def get_action_speed(self, side: int, battle, profile) -> int:
        from pokepy.core.constants import F_TRICK_ROOM, OFF_FIELD, STATUS_PARALYSIS
        from pokepy.sim.views import FieldView

        mon = FieldView(battle).active(side)
        spe = mon.spe
        if mon.status == STATUS_PARALYSIS:
            spe = max(1, spe // 4)
        if int(battle[OFF_FIELD + F_TRICK_ROOM]):
            return max(1, 10000 - spe)
        return max(1, spe)
