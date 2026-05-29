"""Hand-crafted Gen 9 OU sample teams for evaluation.

Real Gen 9 OU teams sourced from common Smogon team templates. Names go
through `pokepy.data.loader.load_id_mappings()` to convert to IDs.

For larger team pools (50K+ from the Metamon HuggingFace dataset), build
a numpy npz with one row per team and load it via the data loader.

This file's teams are deliberately simple — competitive enough to actually
play battles, but not optimized — and only meant as smoke-test fixtures
until the full team pool integration lands.
"""

from __future__ import annotations

from typing import Dict, List, Any

import numpy as np

from pokepy.data.loader import GameData, IDMappings, load_game_data, load_id_mappings
from pokepy.data.item_aliases import encode_item_id

# A "named team" uses string names for everything; loaded into the engine via
# `team_names_to_ids`. Each entry is one of 6 pokemon.
NamedPokemon = Dict[str, Any]
NamedTeam = List[NamedPokemon]

# 5 sample Gen 9 OU teams. Names match Showdown's clean lowercase format.
SAMPLE_TEAMS_NAMED: List[NamedTeam] = [
    # Team 1: Hyper offense
    [
        dict(
            species="dragapult",
            moves=["dracometeor", "shadowball", "uturn", "fireblast"],
            item="choicespecs",
            ability="infiltrator",
            tera_type="ghost",
            level=100,
        ),
        dict(
            species="kingambit",
            moves=["kowtowcleave", "suckerpunch", "ironhead", "lowkick"],
            item="leftovers",
            ability="supremeoverlord",
            tera_type="dark",
            level=100,
        ),
        dict(
            species="greattusk",
            moves=["earthquake", "headlongrush", "iceshard", "knockoff"],
            item="boosterenergy",
            ability="protosynthesis",
            tera_type="ground",
            level=100,
        ),
        dict(
            species="ironvaliant",
            moves=["moonblast", "psyshock", "thunderbolt", "spirit_break"],
            item="boosterenergy",
            ability="quarkdrive",
            tera_type="fairy",
            level=100,
        ),
        dict(
            species="gholdengo",
            moves=["makeitrain", "shadowball", "nastyplot", "recover"],
            item="airballoon",
            ability="goodasgold",
            tera_type="steel",
            level=100,
        ),
        dict(
            species="ironmoth",
            moves=["fireblast", "sludgewave", "energyball", "uturn"],
            item="heavydutyboots",
            ability="quarkdrive",
            tera_type="grass",
            level=100,
        ),
    ],
    # Team 2: Bulky balance
    [
        dict(
            species="corviknight",
            moves=["bravebird", "bodypress", "roost", "uturn"],
            item="leftovers",
            ability="pressure",
            tera_type="dragon",
            level=100,
        ),
        dict(
            species="garganacl",
            moves=["saltcure", "recover", "earthquake", "stealthrock"],
            item="leftovers",
            ability="purifyingsalt",
            tera_type="ghost",
            level=100,
        ),
        dict(
            species="slowking",
            moves=["scald", "futuresight", "chillywater", "slackoff"],
            item="heavydutyboots",
            ability="regenerator",
            tera_type="water",
            level=100,
        ),
        dict(
            species="cinderace",
            moves=["pyroball", "uturn", "willowisp", "courtchange"],
            item="heavydutyboots",
            ability="libero",
            tera_type="grass",
            level=100,
        ),
        dict(
            species="rotomwash",
            moves=["voltswitch", "hydropump", "willowisp", "painsplit"],
            item="leftovers",
            ability="levitate",
            tera_type="ghost",
            level=100,
        ),
        dict(
            species="kingambit",
            moves=["kowtowcleave", "suckerpunch", "swordsdance", "ironhead"],
            item="blacksludge",
            ability="supremeoverlord",
            tera_type="dark",
            level=100,
        ),
    ],
    # Team 3: Stall (simpler — fewer high-investment effects)
    [
        dict(
            species="blissey",
            moves=["seismictoss", "softboiled", "calmmind", "shadowball"],
            item="heavydutyboots",
            ability="naturalcure",
            tera_type="poison",
            level=100,
        ),
        dict(
            species="toxapex",
            moves=["scald", "toxic", "recover", "haze"],
            item="blacksludge",
            ability="regenerator",
            tera_type="fairy",
            level=100,
        ),
        dict(
            species="dondozo",
            moves=["bodypress", "rest", "sleeptalk", "curse"],
            item="leftovers",
            ability="unaware",
            tera_type="grass",
            level=100,
        ),
        dict(
            species="clodsire",
            moves=["earthquake", "recover", "toxic", "spikes"],
            item="leftovers",
            ability="unaware",
            tera_type="dark",
            level=100,
        ),
        dict(
            species="skarmory",
            moves=["bodypress", "roost", "spikes", "whirlwind"],
            item="leftovers",
            ability="sturdy",
            tera_type="dragon",
            level=100,
        ),
        dict(
            species="alomomola",
            moves=["wish", "protect", "scald", "flipturn"],
            item="heavydutyboots",
            ability="regenerator",
            tera_type="water",
            level=100,
        ),
    ],
]


def _lookup(name: str, mapping: Dict[str, int], kind: str) -> int:
    """Look up a clean-name → ID; warn if missing."""
    key = "".join(c for c in str(name) if c.isalnum()).lower()
    if key in mapping:
        return int(mapping[key])
    # Try original (no clean) too
    if str(name) in mapping:
        return int(mapping[str(name)])
    return -1


def team_names_to_ids(
    team: NamedTeam, game_data: GameData, mappings: IDMappings
) -> Dict[str, Any]:
    """Convert a named-team list to the dict format `init_battle_state` expects."""
    species, moves, items, abilities, tera_types, levels = [], [], [], [], [], []
    for entry in team:
        sp_id = _lookup(entry["species"], mappings.species_to_idx, "species")
        if sp_id < 0:
            # Skip unknown species (data file might not have it)
            continue
        m_ids = []
        for m in entry["moves"]:
            mid = _lookup(m, mappings.move_to_idx, "move")
            m_ids.append(mid)
        while len(m_ids) < 4:
            m_ids.append(-1)
        m_ids = m_ids[:4]
        item_id = encode_item_id(entry["item"], mappings.item_to_idx)
        ability_id = _lookup(entry["ability"], mappings.ability_to_idx, "ability")
        tera_id = _lookup(entry["tera_type"], mappings.type_to_idx, "type")

        species.append(sp_id)
        moves.append(m_ids)
        items.append(item_id if item_id >= 0 else 0)
        abilities.append(ability_id if ability_id >= 0 else 0)
        tera_types.append(tera_id if tera_id >= 0 else 0)
        levels.append(int(entry.get("level", 100)))

    return dict(
        species=species,
        moves=moves,
        items=items,
        abilities=abilities,
        tera_types=tera_types,
        levels=levels,
    )


def load_sample_teams() -> List[Dict[str, Any]]:
    """Convert all SAMPLE_TEAMS_NAMED to engine-ready team dicts.

    Skips any team where ALL species lookups failed (data file mismatch).
    """
    gd = load_game_data()
    m = load_id_mappings()
    out: List[Dict[str, Any]] = []
    for team_named in SAMPLE_TEAMS_NAMED:
        team_dict = team_names_to_ids(team_named, gd, m)
        if len(team_dict["species"]) > 0:
            out.append(team_dict)
    return out
