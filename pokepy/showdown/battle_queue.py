"""BattleQueue: turn action list + speed-ordered resolution.

Slice scope: 'move' and 'switch' actions for singles. Ordering follows
Showdown (sim/battle-queue.ts + battle.ts action resolution): actions are
sorted by Battle.comparePriority via speedSort, so genuine speed ties consume a
PRNG shuffle frame exactly like the real sim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from pokepy.showdown.battle import Battle
    from pokepy.showdown.pokemon import Pokemon

# Action.order values (sim/battle-queue.ts resolveAction `orders` table).
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
    "residual": 300,
}


class Action(dict):
    """Thin dict subclass so callbacks can read action.choice etc. via keys."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value


class BattleQueue:
    def __init__(self, battle: "Battle"):
        self.battle = battle
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

    def push(self, action: Action):
        self.list.append(action)

    def unshift(self, action: Action):
        self.list.insert(0, action)

    def clear(self):
        self.list = []

    def sort(self):
        """Speed-sort the queue (resolves speed ties via PRNG shuffle)."""
        self.battle.speed_sort(self.list)
        return self

    # ------------------------------------------------------------------ #
    # Resolution + insertion (sim/battle-queue.ts resolveAction/insertChoice)
    # ------------------------------------------------------------------ #
    def resolve_action(self, action: "Action", mid_turn: bool = False) -> List["Action"]:
        if action is None:
            raise ValueError("Action not passed to resolveAction")
        if action.get("choice") == "pass":
            return []
        if not isinstance(action, Action):
            action = Action(action)
        actions = [action]

        if not action.get("side") and action.get("pokemon"):
            action["side"] = action["pokemon"].side
        if not action.get("move") and action.get("moveid"):
            action["move"] = self.battle.dex.get_active_move(action["moveid"])
        if not action.get("order"):
            choice = action.get("choice")
            if choice in ACTION_ORDER:
                action["order"] = ACTION_ORDER[choice]
            else:
                action["order"] = 200
                if choice not in ("move", "event"):
                    raise ValueError(f"Unexpected orderless action {choice}")

        if not mid_turn:
            if action.get("choice") == "move":
                # (beforeTurnMove / mega / tera / dynamax handling is Phase B)
                action["fractionalPriority"] = self.battle.run_event(
                    "FractionalPriority", action["pokemon"], None, action.get("move"), 0
                )
            elif action.get("choice") in ("switch", "instaswitch"):
                action["pokemon"].switch_flag = False

        if action.get("move"):
            action["move"] = self.battle.dex.get_active_move(action["move"])
            if not action.get("targetLoc"):
                target = self.battle.get_random_target(action["pokemon"], action["move"])
                action["targetLoc"] = 1 if target else 0
            action["originalTarget"] = self.battle.get_target(
                action["pokemon"], action["move"], action.get("targetLoc")
            )
        self.battle.get_action_speed(action)
        return actions

    def add_choice(self, choices):
        if isinstance(choices, list):
            for c in choices:
                self.add_choice(c)
            return
        for resolved in self.resolve_action(choices):
            self.list.append(resolved)

    def insert_choice(self, choices, mid_turn: bool = False):
        if isinstance(choices, list):
            for c in choices:
                self.insert_choice(c)
            return
        choice = choices
        if choice.get("pokemon"):
            choice["pokemon"].speed = choice["pokemon"].get_action_speed()
        actions = self.resolve_action(choice, mid_turn)

        first_index = None
        last_index = None
        for i, cur_action in enumerate(self.list):
            compared = self.battle.compare_priority(actions[0], cur_action)
            if compared <= 0 and first_index is None:
                first_index = i
            if compared < 0:
                last_index = i
                break

        if first_index is None:
            self.list.extend(actions)
        else:
            if last_index is None:
                last_index = len(self.list)
            if first_index == last_index:
                index = first_index
            else:
                index = self.battle.random(first_index, last_index + 1)
            self.list[index:index] = actions

    def cancel_action(self, pokemon) -> bool:
        before = len(self.list)
        self.list = [a for a in self.list if a.get("pokemon") is not pokemon]
        return len(self.list) != before
