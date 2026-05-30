"""Burn status callbacks — worked translation template from A5/A6 handoff.

Source: ``server/pokemon-showdown/data/conditions.ts`` (``brn`` entry).
"""

from __future__ import annotations

from pokepy.showdown.effects import register


@register("conditions", "brn")
class Burn:
    onResidualOrder = 10

    @staticmethod
    def on_start(battle, target, source, source_effect):
        if source_effect and getattr(source_effect, "id", None) == "flameorb":
            battle.add("-status", target, "brn", "[from] item: Flame Orb")
        elif source_effect and getattr(source_effect, "effectType", None) == "Ability":
            battle.add(
                "-status",
                target,
                "brn",
                "[from] ability: " + source_effect.name,
                f"[of] {source}",
            )
        else:
            battle.add("-status", target, "brn")

    @staticmethod
    def on_residual(battle, pokemon, *args, **kwargs):
        battle.damage(pokemon.baseMaxhp // 16, pokemon)
