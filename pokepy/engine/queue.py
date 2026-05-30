"""Turn action queue + speed-ordered resolution for the bitpack engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from pokepy.engine.dispatch import BitpackBattleContext

ACTION_ORDER = {
    "team": 1,
    "start": 2,
    "instaswitch": 3,
    "beforeTurn": 4,
    "beforeTurnMove": 5,
    "revivalblessing": 6,
    "runSwitch": 101,
    "switch": 103,
    "megaEvo": 104,
    "megaEvoX": 104,
    "megaEvoY": 104,
    "runDynamax": 105,
    "terastallize": 106,
    "priorityChargeMove": 107,
    "shift": 200,
    "move": 200,
    "residual": 300,
}


class Action(dict):
    """Showdown-style action dict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value


class BattleQueue:
    def __init__(self, ctx: "BitpackBattleContext"):
        self.ctx = ctx
        self.list: List[Action] = []

    def __len__(self) -> int:
        return len(self.list)

    def __bool__(self) -> bool:
        return bool(self.list)

    def shift(self) -> Optional[Action]:
        if not self.list:
            return None
        return self.list.pop(0)

    def peek(self) -> Optional[Action]:
        return self.list[0] if self.list else None

    def push(self, action: Action) -> None:
        self.list.append(action)

    def unshift(self, action: Action) -> None:
        self.list.insert(0, action)

    def clear(self) -> None:
        self.list = []

    def sort(self) -> "BattleQueue":
        self.ctx.speed_sort(self.list)
        return self

    def resolve_action(self, action: Action, *, mid_turn: bool = False) -> List[Action]:
        if action is None:
            raise ValueError("Action not passed to resolveAction")
        if action.get("choice") == "pass":
            return []
        if not isinstance(action, Action):
            action = Action(action)
        actions: List[Action] = [action]

        if not action.get("order"):
            choice = action.get("choice")
            if choice in ACTION_ORDER:
                action["order"] = ACTION_ORDER[choice]
            else:
                action["order"] = 200

        if not mid_turn and action.get("choice") == "move":
            poff = action.get("pokemon_offset")
            if poff is not None:
                action["fractionalPriority"] = self.ctx.run_event(
                    "FractionalPriority",
                    int(poff),
                    relay_var=0,
                )
        return actions

    def insert_choice(
        self,
        *,
        choice: str,
        side: int,
        action_index: int,
        pokemon_offset: int,
        move_id: Optional[int] = None,
        target: int = 0,
        priority: int = 0,
        speed: int = 0,
    ) -> Action:
        act = Action(
            choice=choice,
            side=int(side),
            action_index=int(action_index),
            pokemon_offset=int(pokemon_offset),
            move_id=move_id,
            target=target,
            priority=int(priority),
            speed=int(speed),
        )
        self.list.append(act)
        return act


__all__ = ["Action", "BattleQueue", "ACTION_ORDER"]
