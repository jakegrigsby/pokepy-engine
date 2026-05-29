"""Universal* dataclasses ported from metamon/interface.py.

Pokepy's standalone copies of the metamon backend-agnostic obs format. Field
names and structure match metamon exactly so the obs builder is a 1:1 port.
Factory methods (`from_Battle`, `from_ReplayPokemon`, etc.) are NOT included —
those depend on poke_env / replay parser, neither of which pokepy needs. The
pokepy bridge `state_to_universal.py` constructs these from a `MultiFormatState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def clean_name(name: str) -> str:
    """Lowercase + strip non-alphanumerics. Matches metamon str_parsing.clean_name."""
    return "".join(c for c in str(name) if c.isalnum()).lower()


def clean_no_numbers(name: str) -> str:
    """Lowercase + strip non-alpha (no digits). Matches metamon clean_no_numbers."""
    return "".join(c for c in str(name) if c.isalpha()).lower()


@dataclass
class UniversalMove:
    name: str
    move_type: str
    category: str
    base_power: int
    accuracy: float
    priority: int
    current_pp: int
    max_pp: int

    @classmethod
    def blank(cls) -> "UniversalMove":
        return cls(
            name="nomove",
            move_type="nomove",
            category="nomove",
            base_power=0,
            accuracy=1.0,
            priority=0,
            current_pp=0,
            max_pp=0,
        )


@dataclass
class UniversalPokemon:
    name: str
    hp_pct: float
    types: str  # space-joined sorted type names ("fire grass")
    item: str
    ability: str
    lvl: int
    status: str
    effect: str
    moves: List[UniversalMove] = field(default_factory=list)

    atk_boost: int = 0
    spa_boost: int = 0
    def_boost: int = 0
    spd_boost: int = 0
    spe_boost: int = 0
    accuracy_boost: int = 0
    evasion_boost: int = 0

    base_atk: int = 0
    base_spa: int = 0
    base_def: int = 0
    base_spd: int = 0
    base_spe: int = 0
    base_hp: int = 0

    tera_type: str = "notype"
    base_species: str = ""

    @classmethod
    def blank(cls) -> "UniversalPokemon":
        return cls(
            name="<blank>",
            hp_pct=0.0,
            types="notype notype",
            item="noitem",
            ability="noability",
            lvl=100,
            status="nostatus",
            effect="noeffect",
            moves=[UniversalMove.blank() for _ in range(4)],
        )


@dataclass
class UniversalState:
    format: str
    player_active_pokemon: UniversalPokemon
    opponent_active_pokemon: UniversalPokemon
    available_switches: List[UniversalPokemon]
    player_prev_move: UniversalMove
    opponent_prev_move: UniversalMove
    opponents_remaining: int
    player_conditions: str
    opponent_conditions: str
    weather: str
    battle_field: str
    forced_switch: bool
    battle_won: bool
    battle_lost: bool
    can_tera: bool
    opponent_teampreview: List[str]
