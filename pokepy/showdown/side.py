"""Side: one player's team + active slot(s) + side conditions.

Slice scope: singles (one active slot). Port of the surface of sim/side.ts the
dispatch and pipeline read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pokepy.showdown.pokemon import Pokemon

if TYPE_CHECKING:
    from pokepy.showdown.battle import Battle


class Side:
    def __init__(self, battle: "Battle", n: int, team: List[Dict[str, Any]], name: str = ""):
        self.battle = battle
        self.n = n
        self.id = f"p{n + 1}"
        self.name = name or self.id
        self.foe: Optional["Side"] = None
        self.ally_side: Optional["Side"] = None

        self.pokemon: List[Pokemon] = []
        for i, pset in enumerate(team):
            self.pokemon.append(Pokemon(battle, self, pset, i))

        self.active: List[Optional[Pokemon]] = [None]  # singles
        self.side_conditions: Dict[str, Dict[str, Any]] = {}
        self.slot_conditions: List[Dict[str, Dict[str, Any]]] = [{}]
        self.faint_counter = 0
        self.z_move_used = False
        self.choice: Dict[str, Any] = {}

    @property
    def effect_type(self) -> str:
        return "Side"

    def __str__(self) -> str:
        return self.id

    @property
    def active_pokemon(self) -> Optional[Pokemon]:
        return self.active[0] if self.active else None

    def pokemon_left(self) -> int:
        return sum(1 for p in self.pokemon if not p.fainted)
