"""Gen3 condition overrides (Showdown data/mods/gen3/conditions.ts)."""

from __future__ import annotations

from pokepy.showdown.effects import register


@register("conditions", "slp", gen=3)
class Gen3Sleep:
    onBeforeMovePriority = 10

    @staticmethod
    def on_start(battle, target, source, source_effect):
        if source_effect and getattr(source_effect, "effectType", None) == "Move":
            battle.add("-status", target, "slp", f"[from] move: {source_effect.name}")
        else:
            battle.add("-status", target, "slp")
        battle.effect_state["time"] = battle.random(2, 6)
        battle.effect_state["skippedTime"] = 0

    @staticmethod
    def on_switch_in(battle, pokemon):
        battle.effect_state["time"] = battle.effect_state.get("time", 0) + battle.effect_state.get("skippedTime", 0)
        battle.effect_state["skippedTime"] = 0

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        if pokemon.has_ability("earlybird"):
            pokemon.status_state["time"] = pokemon.status_state.get("time", 1) - 1
        pokemon.status_state["time"] = pokemon.status_state.get("time", 1) - 1
        if pokemon.status_state.get("time", 0) <= 0:
            pokemon.cure_status()
            return
        battle.add("cant", pokemon, "slp")
        if getattr(move, "sleepUsable", None):
            battle.effect_state["skippedTime"] = battle.effect_state.get("skippedTime", 0) + 1
            return
        battle.effect_state["skippedTime"] = 0
        return False


@register("conditions", "frz", gen=3)
class Gen3Freeze:
    @staticmethod
    def on_damaging_hit(battle, damage, target, source, move):
        resolved = battle.dex.moves.get(move.id)
        if resolved.type == "Fire" and getattr(move, "category", None) != "Status":
            target.cure_status()
