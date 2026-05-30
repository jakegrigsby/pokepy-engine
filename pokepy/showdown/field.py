"""Field: weather / terrain / pseudo-weather container.

Minimal but structured port of sim/field.ts. Holds the global battle
conditions and exposes the lookups the dispatch + pipeline read. Effect
handlers for weather/terrain are registered in pokepy.showdown.effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from pokepy.showdown.battle import Battle


class Field:
    def __init__(self, battle: "Battle"):
        self.battle = battle
        self.weather: str = ""
        self.weather_state: Dict[str, Any] = {"id": ""}
        self.terrain: str = ""
        self.terrain_state: Dict[str, Any] = {"id": ""}
        self.pseudo_weather: Dict[str, Dict[str, Any]] = {}

    @property
    def effect_type(self) -> str:
        return "Field"

    def get_weather(self):
        return self.battle.dex.conditions.get(self.weather)

    def get_terrain(self):
        return self.battle.dex.conditions.get(self.terrain)

    def get_pseudo_weather(self, status):
        from pokepy.showdown.dex import to_id

        return self.pseudo_weather.get(to_id(status))

    def is_weather(self, weather) -> bool:
        if not self.weather:
            return False
        from pokepy.showdown.dex import to_id

        if isinstance(weather, (list, tuple)):
            return self.weather in {to_id(w) for w in weather}
        return self.weather == to_id(weather)

    def is_terrain(self, terrain) -> bool:
        if not self.terrain:
            return False
        from pokepy.showdown.dex import to_id

        if isinstance(terrain, (list, tuple)):
            return self.terrain in {to_id(t) for t in terrain}
        return self.terrain == to_id(terrain)

    def suppressing_weather(self) -> bool:
        # Air Lock / Cloud Nine. Wired when those abilities are translated.
        for side in self.battle.sides:
            for pokemon in side.active:
                if pokemon and not pokemon.fainted and not pokemon.ignoring_ability():
                    ability = pokemon.get_ability()
                    if ability and ability.get("suppressWeather"):
                        return True
        return False
