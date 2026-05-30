"""Gen2 condition overrides (Showdown data/mods/gen2/conditions.ts)."""

from __future__ import annotations

from pokepy.showdown.effects import register


def _residual_dmg(battle, pokemon):
    volatile = pokemon.volatiles.get("residualdmg")
    if not volatile:
        return
    counter = volatile.get("counter", 0)
    dmg = battle.clamp_int_range(pokemon.maxhp // 16 * counter, 1)
    battle.damage(dmg, pokemon)


@register("conditions", "residualdmg", gen=2)
class Gen2ResidualDmgVolatile:
    onAfterMoveSelfPriority = 100

    @staticmethod
    def on_start(battle, target):
        battle.effect_state["counter"] = 0

    @staticmethod
    def on_after_move_self(battle, pokemon, target, move):
        if pokemon.status in ("brn", "psn", "tox"):
            pokemon.volatiles.setdefault("residualdmg", {"id": "residualdmg", "target": pokemon})
            pokemon.volatiles["residualdmg"]["counter"] = pokemon.volatiles["residualdmg"].get("counter", 0) + 1

    @staticmethod
    def on_after_switch_in_self(battle, pokemon):
        if pokemon.status in ("brn", "psn", "tox"):
            pokemon.volatiles.setdefault("residualdmg", {"id": "residualdmg", "target": pokemon})
            pokemon.volatiles["residualdmg"]["counter"] = pokemon.volatiles["residualdmg"].get("counter", 0) + 1


@register("conditions", "brn", gen=2)
class Gen2Burn:
    onAfterMoveSelfPriority = 3

    @staticmethod
    def on_start(battle, target, source, source_effect):
        battle.add("-status", target, "brn")

    @staticmethod
    def on_after_move_self(battle, pokemon, target, move):
        _residual_dmg(battle, pokemon)

    @staticmethod
    def on_after_switch_in_self(battle, pokemon):
        _residual_dmg(battle, pokemon)


@register("conditions", "psn", gen=2)
class Gen2Poison:
    onAfterMoveSelfPriority = 3

    @staticmethod
    def on_start(battle, target, source, source_effect):
        battle.add("-status", target, "psn")

    @staticmethod
    def on_after_move_self(battle, pokemon, target, move):
        _residual_dmg(battle, pokemon)

    @staticmethod
    def on_after_switch_in_self(battle, pokemon):
        _residual_dmg(battle, pokemon)


@register("conditions", "tox", gen=2)
class Gen2BadPoison:
    onAfterMoveSelfPriority = 3

    @staticmethod
    def on_start(battle, target, source, source_effect):
        battle.add("-status", target, "tox")
        if "residualdmg" not in target.volatiles:
            target.volatiles["residualdmg"] = {"id": "residualdmg", "target": target, "counter": 0}
        target.volatiles["residualdmg"]["counter"] = 0

    @staticmethod
    def on_after_move_self(battle, pokemon, target, move):
        counter = pokemon.volatiles.get("residualdmg", {}).get("counter", 0)
        dmg = battle.clamp_int_range(pokemon.maxhp // 16 * counter, 1)
        battle.damage(dmg, pokemon, pokemon)

    @staticmethod
    def on_switch_in(battle, pokemon):
        pokemon.status = "psn"
        battle.add("-status", pokemon, "psn", "[silent]")

    @staticmethod
    def on_after_switch_in_self(battle, pokemon):
        battle.damage(battle.clamp_int_range(pokemon.maxhp // 16, 1), pokemon)


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


@register("conditions", "frz", gen=2)
class Gen2Freeze:
    onResidualOrder = 7

    @staticmethod
    def on_before_move(battle, pokemon, target, move):
        flags = getattr(move, "flags", None) or {}
        if flags.get("defrost"):
            return
        battle.add("cant", pokemon, "frz")
        return False

    @staticmethod
    def on_after_move_secondary(battle, target, source, move):
        secondary = getattr(move, "secondary", None)
        if (secondary and secondary.get("status") == "brn") or getattr(move, "statusRoll", None) == "brn":
            target.cure_status()

    @staticmethod
    def on_after_move_secondary_self(battle, pokemon, target, move):
        flags = getattr(move, "flags", None) or {}
        if flags.get("defrost"):
            pokemon.cure_status()

    @staticmethod
    def on_residual(battle, pokemon, *args, **kwargs):
        if battle.random_chance(25, 256):
            pokemon.cure_status()
