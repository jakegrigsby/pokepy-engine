"""Gen1/gen2 condition overrides (Showdown data/mods/gen1|gen2/conditions.ts)."""

from __future__ import annotations

from pokepy.showdown.effects import register


@register("conditions", "slp", gen=1)
class Gen1Sleep:
    onBeforeMovePriority = 10
    onAfterMoveSelfPriority = 3

    @staticmethod
    def on_start(battle, target, source, source_effect):
        if source_effect and getattr(source_effect, "effectType", None) == "Move":
            battle.add("-status", target, "slp", f"[from] move: {source_effect.name}")
        else:
            battle.add("-status", target, "slp")
        battle.effect_state["startTime"] = battle.random(1, 8)
        battle.effect_state["time"] = battle.effect_state["startTime"]

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        pokemon.status_state["time"] = pokemon.status_state.get("time", 1) - 1
        if pokemon.status_state.get("time", 0) > 0:
            battle.add("cant", pokemon, "slp")
        pokemon.last_move = None
        return False

    @staticmethod
    def on_after_move_self(battle, pokemon, target, move):
        if pokemon.status_state.get("time", 0) <= 0:
            pokemon.cure_status()


@register("conditions", "frz", gen=1)
class Gen1Freeze:
    onBeforeMovePriority = 12

    @staticmethod
    def on_start(battle, target, source, source_effect):
        battle.add("-status", target, "frz")

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        battle.add("cant", pokemon, "frz")
        pokemon.last_move = None
        return False


@register("conditions", "par", gen=1)
class Gen1Paralysis:
    onBeforeMovePriority = 2

    @staticmethod
    def on_start(battle, target, source, source_effect):
        battle.add("-status", target, "par")

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        if battle.random_chance(63, 256):
            battle.add("cant", pokemon, "par")
            return False


@register("conditions", "par", gen=2)
class Gen2Paralysis:
    onBeforeMovePriority = 2

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        if battle.random_chance(63, 256):
            battle.add("cant", pokemon, "par")
            return False


@register("conditions", "slp", gen=2)
class Gen2Sleep:
    onBeforeMovePriority = 10

    @staticmethod
    def on_start(battle, target, source, source_effect):
        if source_effect and getattr(source_effect, "effectType", None) == "Move":
            battle.add("-status", target, "slp", f"[from] move: {source_effect.name}")
        else:
            battle.add("-status", target, "slp")
        battle.effect_state["time"] = battle.random(2, 8)

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        pokemon.status_state["time"] = pokemon.status_state.get("time", 1) - 1
        if pokemon.status_state.get("time", 0) <= 0:
            pokemon.cure_status()
            return
        battle.add("cant", pokemon, "slp")
        if getattr(move, "sleepUsable", None):
            return
        return False
