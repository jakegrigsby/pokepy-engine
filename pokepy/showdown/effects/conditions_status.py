"""Primary status condition callbacks (gen9 base).

Source: ``server/pokemon-showdown/data/conditions.ts``.
"""

from __future__ import annotations

from pokepy.showdown.effects import register


@register("conditions", "par")
class Paralysis:
    onModifySpePriority = -101

    @staticmethod
    def on_start(battle, target, source, source_effect):
        if source_effect and getattr(source_effect, "effectType", None) == "Ability":
            battle.add(
                "-status",
                target,
                "par",
                "[from] ability: " + source_effect.name,
                f"[of] {source}",
            )
        else:
            battle.add("-status", target, "par")

    @staticmethod
    def on_modify_spe(battle, spe, pokemon, source=None, source_effect=None):
        spe = battle.final_modify(spe)
        if not pokemon.has_ability("quickfeet"):
            spe = spe * 50 // 100
        return spe

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        if battle.random_chance(1, 4):
            battle.add("cant", pokemon, "par")
            return False


@register("conditions", "psn")
class Poison:
    onResidualOrder = 9

    @staticmethod
    def on_start(battle, target, source, source_effect):
        if source_effect and getattr(source_effect, "effectType", None) == "Ability":
            battle.add(
                "-status",
                target,
                "psn",
                "[from] ability: " + source_effect.name,
                f"[of] {source}",
            )
        else:
            battle.add("-status", target, "psn")

    @staticmethod
    def on_residual(battle, pokemon, *args, **kwargs):
        battle.damage(pokemon.baseMaxhp // 8, pokemon)


@register("conditions", "tox")
class BadPoison:
    onResidualOrder = 9

    @staticmethod
    def on_start(battle, target, source, source_effect):
        battle.effect_state["stage"] = 0
        if source_effect and getattr(source_effect, "id", None) == "toxicorb":
            battle.add("-status", target, "tox", "[from] item: Toxic Orb")
        elif source_effect and getattr(source_effect, "effectType", None) == "Ability":
            battle.add(
                "-status",
                target,
                "tox",
                "[from] ability: " + source_effect.name,
                f"[of] {source}",
            )
        else:
            battle.add("-status", target, "tox")

    @staticmethod
    def on_switch_in(battle, pokemon):
        battle.effect_state["stage"] = 0

    @staticmethod
    def on_residual(battle, pokemon, *args, **kwargs):
        stage = battle.effect_state.get("stage", 0)
        if stage < 15:
            battle.effect_state["stage"] = stage + 1
        stage = battle.effect_state["stage"]
        dmg = battle.clamp_int_range(pokemon.baseMaxhp // 16, 1) * stage
        battle.damage(dmg, pokemon)


@register("conditions", "slp")
class Sleep:
    onBeforeMovePriority = 10

    @staticmethod
    def on_start(battle, target, source, source_effect):
        if source_effect and getattr(source_effect, "effectType", None) == "Ability":
            battle.add(
                "-status",
                target,
                "slp",
                "[from] ability: " + source_effect.name,
                f"[of] {source}",
            )
        elif source_effect and getattr(source_effect, "effectType", None) == "Move":
            battle.add("-status", target, "slp", f"[from] move: {source_effect.name}")
        else:
            battle.add("-status", target, "slp")
        start_time = battle.random(2, 5)
        battle.effect_state["startTime"] = start_time
        battle.effect_state["time"] = start_time

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        if pokemon.has_ability("earlybird"):
            pokemon.status_state["time"] = pokemon.status_state.get("time", 1) - 1
        pokemon.status_state["time"] = pokemon.status_state.get("time", 1) - 1
        if pokemon.status_state.get("time", 0) <= 0:
            pokemon.cure_status()
            return
        battle.add("cant", pokemon, "slp")
        if getattr(move, "sleepUsable", None) or (getattr(move, "flags", None) or {}).get("sleepUsable"):
            return
        return False


@register("conditions", "frz")
class Freeze:
    onBeforeMovePriority = 10

    @staticmethod
    def on_start(battle, target, source, source_effect):
        if source_effect and getattr(source_effect, "effectType", None) == "Ability":
            battle.add(
                "-status",
                target,
                "frz",
                "[from] ability: " + source_effect.name,
                f"[of] {source}",
            )
        else:
            battle.add("-status", target, "frz")

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        flags = getattr(move, "flags", None) or {}
        if flags.get("defrost") and not (move.id == "burnup" and not pokemon.has_type("Fire")):
            return
        if battle.random_chance(1, 5):
            pokemon.cure_status()
            return
        battle.add("cant", pokemon, "frz")
        return False

    @staticmethod
    def on_modify_move(battle, move, pokemon):
        flags = getattr(move, "flags", None) or {}
        if flags.get("defrost"):
            battle.add("-curestatus", pokemon, "frz", f"[from] move: {move}")
            pokemon.clear_status()

    @staticmethod
    def on_after_move_secondary(battle, target, source, move):
        if getattr(move, "thawsTarget", None):
            target.cure_status()

    @staticmethod
    def on_damaging_hit(battle, damage, target, source, move):
        if move.type == "Fire" and getattr(move, "category", None) != "Status" and move.id != "polarflare":
            target.cure_status()
