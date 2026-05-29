#!/usr/bin/env python3
"""
Extract Pokemon data from Pokemon Showdown TypeScript files.

This script parses PS data files and converts them to NumPy arrays
that can be loaded efficiently by the JAX battle engine.

Usage:
    python scripts/extract_ps_data.py --ps-path ../pokemon-showdown --output data/
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


_NUM_RE = re.compile(r"\bnum:\s*(-?\d+)")


def _parse_entry_num(entry_content: str) -> int | None:
    """Parse Showdown's ``num`` field, ignoring e.g. ``spritenum``."""
    match = _NUM_RE.search(entry_content)
    if not match:
        return None
    return int(match.group(1))


def parse_ts_object(content: str) -> dict:
    """
    Parse a TypeScript object literal into a Python dict.

    This is a simplified parser that handles the PS data format.
    It's not a full TS parser but works for the data files.
    """
    # Remove TypeScript type annotations
    content = re.sub(r":\s*\w+(\[\])?\s*(?=[,\}\]])", "", content)

    # Remove 'as const' and similar
    content = re.sub(r"\s+as\s+const", "", content)
    content = re.sub(r"\s+as\s+\w+", "", content)

    # Convert TypeScript syntax to JSON-compatible
    # Single quotes to double quotes
    content = content.replace("'", '"')

    # Remove trailing commas
    content = re.sub(r",(\s*[\}\]])", r"\1", content)

    # Handle unquoted keys
    content = re.sub(r"(\{|\,)\s*([a-zA-Z_]\w*)\s*:", r'\1"\2":', content)

    # Handle special values
    content = content.replace("true", "True").replace("false", "False")
    content = content.replace("null", "None")

    try:
        return eval(content)
    except Exception as e:
        print(f"Warning: Failed to parse object: {e}")
        return {}


def extract_pokedex(ps_path: Path) -> dict:
    """Extract Pokemon species data from pokedex.ts."""
    pokedex_file = ps_path / "data" / "pokedex.ts"
    content = pokedex_file.read_text()

    # Find the Pokedex object - handle TypeScript type annotation
    match = re.search(r"export const Pokedex[^=]*=\s*\{", content)
    if not match:
        raise ValueError("Could not find Pokedex in file")

    # Extract species entries using a brace-counting approach
    species_data = {}

    # Find all species entry starts: \tspeciesid: {
    entry_starts = list(re.finditer(r"\n\t(\w+):\s*\{", content))

    for i, start_match in enumerate(entry_starts):
        species_id = start_match.group(1)
        start_pos = start_match.end()

        # Count braces to find the end of this entry
        brace_count = 1
        pos = start_pos
        while pos < len(content) and brace_count > 0:
            if content[pos] == "{":
                brace_count += 1
            elif content[pos] == "}":
                brace_count -= 1
            pos += 1

        entry_content = content[start_pos : pos - 1]

        try:
            # Parse individual fields
            entry = {}

            # num
            num = _parse_entry_num(entry_content)
            if num is not None:
                entry["num"] = num

            # name
            name_match = re.search(r'name:\s*["\']([^"\']+)["\']', entry_content)
            if name_match:
                entry["name"] = name_match.group(1)

            # baseSpecies: present only on alternate formes (e.g. Calyrex-Ice).
            # Base species (Regice, Yanmega, ...) never have this field, which is
            # the canonical way to distinguish a forme from a base species.
            base_species_match = re.search(
                r'baseSpecies:\s*["\']([^"\']+)["\']', entry_content
            )
            if base_species_match:
                entry["baseSpecies"] = base_species_match.group(1)

            # types
            types_match = re.search(r"types:\s*\[([^\]]+)\]", entry_content)
            if types_match:
                types_str = types_match.group(1)
                entry["types"] = [t.strip().strip("\"'") for t in types_str.split(",")]

            # baseStats - be more specific to avoid matching nested objects
            stats_match = re.search(
                r"baseStats:\s*\{\s*hp:\s*(\d+),\s*atk:\s*(\d+),\s*def:\s*(\d+),\s*spa:\s*(\d+),\s*spd:\s*(\d+),\s*spe:\s*(\d+)\s*\}",
                entry_content,
            )
            if stats_match:
                entry["baseStats"] = {
                    "hp": int(stats_match.group(1)),
                    "atk": int(stats_match.group(2)),
                    "def": int(stats_match.group(3)),
                    "spa": int(stats_match.group(4)),
                    "spd": int(stats_match.group(5)),
                    "spe": int(stats_match.group(6)),
                }

            # abilities
            abilities_match = re.search(r"abilities:\s*\{([^}]+)\}", entry_content)
            if abilities_match:
                abilities_str = abilities_match.group(1)
                abilities = {}
                for ab_match in re.finditer(
                    r'([0HIS]):\s*["\']([^"\']+)["\']', abilities_str
                ):
                    abilities[ab_match.group(1)] = ab_match.group(2)
                entry["abilities"] = abilities

            # weightkg
            weight_match = re.search(r"weightkg:\s*([\d.]+)", entry_content)
            if weight_match:
                entry["weightkg"] = float(weight_match.group(1))

            if "num" in entry and entry["num"] > 0:
                # Only keep base forms - skip alternate forms (mega, gmax, etc.)
                # These have the same Pokedex number and would overwrite base form stats
                # Skip if species_id ENDS WITH forme suffixes (not just contains)
                skip_suffixes = (
                    "mega",
                    "megax",
                    "megay",
                    "gmax",
                    "primal",
                    "origin",
                    "altered",
                    "sky",
                    "land",
                    "therian",
                    "zen",
                    "pirouette",
                    "blade",
                    "shield",
                    "complete",
                    "unbound",
                    "sunny",
                    "rainy",
                    "snowy",
                    "dusk",
                    "dawn",
                    "midnight",
                    "school",
                    "crest",
                    "busted",
                    "hangry",
                    "crowned",
                    "eternamax",
                    "ice",
                    "shadow",
                    "noice",
                    "gulping",
                    "gorging",
                    "lowkey",
                    "rapidstrike",
                    "dada",
                    "bloodmoon",
                    "cornerstone",
                    "wellspring",
                    "hearthflame",
                    "stellar",
                    "teal",
                    "tera",
                    "family",
                )

                # Pokemon that have these suffixes as part of their base name (not forms)
                # Also includes important alternate forms that have different stats/typing
                base_form_exceptions = {
                    "meganium",
                    "landorus",
                    "landorustherian",
                    "thundurus",
                    "thundurustherian",
                    "tornadus",
                    "tornadustherian",
                    "giratina",
                    "shaymin",
                    "keldeo",
                    "meloetta",
                    "aegislash",
                    "hoopa",
                    "zygarde",
                    "minior",
                    "wishiwashi",
                    "darmanitan",
                    "eiscue",
                    "morpeko",
                    "indeedee",
                    "zacian",
                    "zamazenta",
                    "eternatus",
                    "urshifu",
                    "calyrex",
                    "palafin",
                    "terapagos",
                    "ogerpon",
                    "ursalunabloodmoon",
                    "urshifurapidstrike",
                    "ogerponwellspring",
                    "ogerponhearthflame",
                    "ogerponcornerstone",
                    "ogerpontealtera",
                    "ogerponwellspringtera",
                    "ogerponhearthflametera",
                    "ogerponcornerstonetera",
                    # Alternate forms with different stats/typing
                    "shayminsky",
                    "weezinggalar",
                    "darmanitangalar",
                    "slowkinggalar",
                    "slowbrogalar",
                    "articunogalar",
                    "zapdosgalar",
                    "moltresgalar",
                    "typhlosionhisui",
                    "samurotthisui",
                    "decidueyehisui",
                    "arcaninehisui",
                    "lilliganthisui",
                    "zoroarkhisui",
                    "braviaryhisui",
                    "goodrahisui",
                    "avalugghisui",
                    "snaborax",
                }

                is_alternate_form = False
                species_lower = species_id.lower()

                # Only true formes carry a `baseSpecies` field. Gating the
                # suffix heuristic on this avoids false positives on base
                # species whose id merely ends in a forme word (regICE,
                # yanMEGA, stoutLAND, doubLADE, marSHADOW, finiZEN), which were
                # previously dropped by mistake. Forme keep/drop decisions are
                # otherwise unchanged.
                has_base_species = "baseSpecies" in entry

                if has_base_species and species_lower not in base_form_exceptions:
                    for suffix in skip_suffixes:
                        if species_lower.endswith(suffix) and species_lower != suffix:
                            is_alternate_form = True
                            break

                if not is_alternate_form:
                    species_data[species_id] = entry

        except Exception as e:
            continue

    return species_data


def extract_moves(ps_path: Path) -> dict:
    """Extract move data from moves.ts."""
    moves_file = ps_path / "data" / "moves.ts"
    content = moves_file.read_text()

    moves_data = {}

    # Pattern for move entries - need to handle deeply nested braces
    # Match from \tmoveid: { to the next \t} at same indentation
    pattern = r"\t(\w+):\s*\{([\s\S]*?)\n\t\},"

    for match in re.finditer(pattern, content):
        move_id = match.group(1)
        entry_content = match.group(2)

        try:
            entry = {}

            # num
            num = _parse_entry_num(entry_content)
            if num is not None:
                entry["num"] = num

            # name
            name_match = re.search(r'name:\s*["\']([^"\']+)["\']', entry_content)
            if name_match:
                entry["name"] = name_match.group(1)

            # basePower
            bp_match = re.search(r"basePower:\s*(\d+)", entry_content)
            if bp_match:
                entry["basePower"] = int(bp_match.group(1))
            else:
                entry["basePower"] = 0

            # accuracy
            acc_match = re.search(r"accuracy:\s*(\d+|true)", entry_content)
            if acc_match:
                acc_val = acc_match.group(1)
                entry["accuracy"] = True if acc_val == "true" else int(acc_val)
            else:
                entry["accuracy"] = 100

            # pp
            pp_match = re.search(r"pp:\s*(\d+)", entry_content)
            if pp_match:
                entry["pp"] = int(pp_match.group(1))

            # type
            type_match = re.search(r'type:\s*["\'](\w+)["\']', entry_content)
            if type_match:
                entry["type"] = type_match.group(1)

            # category
            cat_match = re.search(r'category:\s*["\'](\w+)["\']', entry_content)
            if cat_match:
                entry["category"] = cat_match.group(1)

            # priority
            pri_match = re.search(r"priority:\s*(-?\d+)", entry_content)
            if pri_match:
                entry["priority"] = int(pri_match.group(1))
            else:
                entry["priority"] = 0

            # target
            target_match = re.search(r'target:\s*["\'](\w+)["\']', entry_content)
            if target_match:
                entry["target"] = target_match.group(1)

            # flags
            flags_match = re.search(r"flags:\s*\{([^}]*)\}", entry_content)
            if flags_match:
                flags_str = flags_match.group(1)
                flags = {}
                for flag_match in re.finditer(r"(\w+):\s*1", flags_str):
                    flags[flag_match.group(1)] = 1
                entry["flags"] = flags

            # critRatio
            crit_match = re.search(r"critRatio:\s*(\d+)", entry_content)
            if crit_match:
                entry["critRatio"] = int(crit_match.group(1))
            else:
                entry["critRatio"] = 1

            # secondary effects (simplified)
            if "secondary:" in entry_content:
                entry["hasSecondary"] = True

            # recoil
            recoil_match = re.search(r"recoil:\s*\[(\d+),\s*(\d+)\]", entry_content)
            if recoil_match:
                entry["recoil"] = [
                    int(recoil_match.group(1)),
                    int(recoil_match.group(2)),
                ]

            # drain
            drain_match = re.search(r"drain:\s*\[(\d+),\s*(\d+)\]", entry_content)
            if drain_match:
                entry["drain"] = [int(drain_match.group(1)), int(drain_match.group(2))]

            if "num" in entry:
                moves_data[move_id] = entry

        except Exception as e:
            continue

    return moves_data


def extract_abilities(ps_path: Path) -> dict:
    """Extract ability data from abilities.ts."""
    abilities_file = ps_path / "data" / "abilities.ts"
    content = abilities_file.read_text()

    abilities_data = {}

    # Pattern to match ability entries (they start with tab and ID)
    # Need to match multi-line entries with function bodies
    pattern = r"\t(\w+):\s*\{([\s\S]*?)\n\t\},"

    for match in re.finditer(pattern, content):
        ability_id = match.group(1)
        entry_content = match.group(2)

        try:
            entry = {}

            # num
            num = _parse_entry_num(entry_content)
            if num is not None:
                entry["num"] = num

            # name
            name_match = re.search(r'name:\s*["\']([^"\']+)["\']', entry_content)
            if name_match:
                entry["name"] = name_match.group(1)

            # rating (can be float like 0.1)
            rating_match = re.search(r"rating:\s*(-?[\d.]+)", entry_content)
            if rating_match:
                entry["rating"] = float(rating_match.group(1))

            # Check for common ability effects
            if "onModifyAtk" in entry_content:
                entry["modifiesAtk"] = True
            if "onModifyDef" in entry_content:
                entry["modifiesDef"] = True
            if "onModifySpA" in entry_content:
                entry["modifiesSpA"] = True
            if "onModifySpD" in entry_content:
                entry["modifiesSpD"] = True
            if "onModifySpe" in entry_content:
                entry["modifiesSpe"] = True
            if "onBasePower" in entry_content:
                entry["modifiesBasePower"] = True
            if "onSwitchIn" in entry_content:
                entry["hasSwitchIn"] = True
            if "onStart" in entry_content:
                entry["hasOnStart"] = True
            if "onDamagingHit" in entry_content:
                entry["hasDamagingHit"] = True

            if "num" in entry:
                abilities_data[ability_id] = entry

        except Exception as e:
            continue

    return abilities_data


def extract_items(ps_path: Path) -> dict:
    """Extract item data from items.ts."""
    items_file = ps_path / "data" / "items.ts"
    content = items_file.read_text()

    items_data = {}

    # Brace-counting approach (same as extract_pokedex) so item entries
    # containing callbacks with nested blocks (e.g. Power Herb's onChargeMove
    # with an inner `if`) are captured in full. The old single-level-nesting
    # regex silently dropped ~285 such items, including their `num` field.
    entry_starts = list(re.finditer(r"\n\t(\w+):\s*\{", content))

    for start_match in entry_starts:
        item_id = start_match.group(1)
        start_pos = start_match.end()

        brace_count = 1
        pos = start_pos
        while pos < len(content) and brace_count > 0:
            if content[pos] == "{":
                brace_count += 1
            elif content[pos] == "}":
                brace_count -= 1
            pos += 1

        entry_content = content[start_pos : pos - 1]

        try:
            entry = {}

            # num
            num = _parse_entry_num(entry_content)
            if num is not None:
                entry["num"] = num

            # name
            name_match = re.search(r'name:\s*["\']([^"\']+)["\']', entry_content)
            if name_match:
                entry["name"] = name_match.group(1)

            # gen
            gen_match = re.search(r"gen:\s*(\d+)", entry_content)
            if gen_match:
                entry["gen"] = int(gen_match.group(1))

            # fling basePower
            fling_match = re.search(
                r"fling:\s*\{[^}]*basePower:\s*(\d+)", entry_content
            )
            if fling_match:
                entry["flingBasePower"] = int(fling_match.group(1))

            # isBerry
            if "isBerry: true" in entry_content or "isBerry:true" in entry_content:
                entry["isBerry"] = True

            # isChoice
            if "isChoice: true" in entry_content or "isChoice:true" in entry_content:
                entry["isChoice"] = True

            # megaStone
            mega_match = re.search(r'megaStone:\s*["\'](\w+)["\']', entry_content)
            if mega_match:
                entry["megaStone"] = mega_match.group(1)

            if "num" in entry:
                items_data[item_id] = entry

        except Exception as e:
            continue

    return items_data


def extract_typechart(ps_path: Path) -> dict:
    """Extract type chart from typechart.ts."""
    typechart_file = ps_path / "data" / "typechart.ts"
    content = typechart_file.read_text()

    type_names = [
        "Normal",
        "Fire",
        "Water",
        "Electric",
        "Grass",
        "Ice",
        "Fighting",
        "Poison",
        "Ground",
        "Flying",
        "Psychic",
        "Bug",
        "Rock",
        "Ghost",
        "Dragon",
        "Dark",
        "Steel",
        "Fairy",
        "Stellar",
    ]

    # Initialize chart with neutral (1.0) effectiveness
    # chart[defending_type][attacking_type] = effectiveness
    chart = {}
    for def_type in type_names:
        chart[def_type] = {}
        for atk_type in type_names:
            chart[def_type][atk_type] = 1.0

    # Parse damageTaken for each type
    # In PS: TypeName.damageTaken[AttackingType] = how much damage that type takes
    # 0 = immune (0x), 1 = resist (0.5x), 2 = neutral (omitted), 3 = weak (2x)
    for type_name in type_names:
        pattern = rf"{type_name.lower()}:\s*\{{[^}}]*damageTaken:\s*\{{([^}}]+)\}}"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            damage_str = match.group(1)

            # Parse each attacking type's effectiveness against this defending type
            for atk_match in re.finditer(r"(\w+):\s*(\d+)", damage_str):
                atk_type_str = atk_match.group(1)
                effectiveness_code = int(atk_match.group(2))

                # Find the attacking type (case-insensitive match)
                atk_type = None
                for t in type_names:
                    if t.lower() == atk_type_str.lower():
                        atk_type = t
                        break

                if atk_type:
                    # Convert PS damageTaken code to multiplier
                    # PS codes for damageTaken:
                    # 0 = neutral (1x) - this is the default
                    # 1 = super effective against this type (type takes 2x)
                    # 2 = not very effective against this type (type takes 0.5x)
                    # 3 = immune (type takes 0x)
                    if effectiveness_code == 1:
                        chart[type_name][atk_type] = 2.0  # Weak to this type
                    elif effectiveness_code == 2:
                        chart[type_name][atk_type] = 0.5  # Resists this type
                    elif effectiveness_code == 3:
                        chart[type_name][atk_type] = 0.0  # Immune
                    # effectiveness_code == 0 is neutral (1.0), already set

    return chart


def create_numpy_arrays(
    species_data: dict,
    moves_data: dict,
    abilities_data: dict,
    items_data: dict,
    typechart: dict,
    output_dir: Path,
) -> None:
    """Convert parsed data to NumPy arrays and save."""

    output_dir.mkdir(parents=True, exist_ok=True)

    type_names = [
        "Normal",
        "Fire",
        "Water",
        "Electric",
        "Grass",
        "Ice",
        "Fighting",
        "Poison",
        "Ground",
        "Flying",
        "Psychic",
        "Bug",
        "Rock",
        "Ghost",
        "Dragon",
        "Dark",
        "Steel",
        "Fairy",
        "Stellar",
    ]
    type_to_idx = {t.lower(): i for i, t in enumerate(type_names)}

    # === Type Chart ===
    type_chart_array = np.ones((19, 19), dtype=np.float32)
    for def_idx, def_type in enumerate(type_names):
        for atk_idx, atk_type in enumerate(type_names):
            if def_type in typechart and atk_type in typechart[def_type]:
                type_chart_array[def_idx, atk_idx] = typechart[def_type][atk_type]

    np.save(output_dir / "type_chart.npy", type_chart_array)
    print(f"Saved type chart: {type_chart_array.shape}")

    # === Species Data ===
    # Create ID mappings - use unique indices for each form
    # Sort by num first, then by species_id to ensure base forms come first
    species_list = sorted(
        species_data.items(), key=lambda x: (x[1].get("num", 9999), x[0])
    )
    species_id_to_idx = {}

    # Track which dex numbers have been seen and their base form index
    num_to_base_idx = {}  # num -> index of the BASE form (for normalized name lookup)

    # Assign unique indices to all species
    # Base forms get their national dex number, regional forms get higher indices
    next_regional_idx = 2000  # Start regional forms at index 2000
    max_species = 3000  # Accommodate all forms

    # Base stats array: (num_species, 6)
    species_base_stats = np.zeros((max_species, 6), dtype=np.int16)
    # Types array: (num_species, 2) - -1 for single type
    species_types = np.full((max_species, 2), -1, dtype=np.int8)
    # Weight array for weight-based moves
    species_weight = np.zeros(max_species, dtype=np.float32)

    species_names = {}

    for species_id, data in species_list:
        num = data.get("num", 0)
        if num <= 0:
            continue

        # Check if this is a regional form
        is_regional = any(
            suffix in species_id.lower()
            for suffix in ["paldea", "alola", "galar", "hisui", "starter"]
        )

        if is_regional or num in num_to_base_idx:
            # Regional forms and same-dex formes (Ogerpon masks, etc.) need unique indices.
            idx = next_regional_idx
            next_regional_idx += 1
        else:
            idx = num
            num_to_base_idx[num] = idx

        if idx >= max_species:
            continue

        species_id_to_idx[species_id] = idx
        species_names[idx] = data.get("name", species_id)

        # Base stats - use idx for array indexing
        stats = data.get("baseStats", {})
        species_base_stats[idx, 0] = stats.get("hp", 0)
        species_base_stats[idx, 1] = stats.get("atk", 0)
        species_base_stats[idx, 2] = stats.get("def", 0)
        species_base_stats[idx, 3] = stats.get("spa", 0)
        species_base_stats[idx, 4] = stats.get("spd", 0)
        species_base_stats[idx, 5] = stats.get("spe", 0)

        # Types
        types = data.get("types", [])
        for i, t in enumerate(types[:2]):
            type_idx = type_to_idx.get(t.lower(), -1)
            species_types[idx, i] = type_idx

        # Weight
        species_weight[idx] = data.get("weightkg", 0.0)

    np.save(output_dir / "species_base_stats.npy", species_base_stats)
    np.save(output_dir / "species_types.npy", species_types)
    np.save(output_dir / "species_weight.npy", species_weight)
    print(f"Saved species data for {len(species_id_to_idx)} Pokemon")

    # === Moves Data ===
    moves_list = sorted(moves_data.items(), key=lambda x: x[1].get("num", 9999))
    move_id_to_idx = {}

    max_moves = max(m["num"] for _, m in moves_list if m.get("num", 0) > 0) + 1

    # Move arrays
    move_base_power = np.zeros(max_moves, dtype=np.int16)
    move_accuracy = np.full(max_moves, 100, dtype=np.int8)
    move_pp = np.zeros(max_moves, dtype=np.int8)
    move_type = np.zeros(max_moves, dtype=np.int8)
    move_category = np.zeros(
        max_moves, dtype=np.int8
    )  # 0=status, 1=physical, 2=special
    move_priority = np.zeros(max_moves, dtype=np.int8)
    move_crit_ratio = np.ones(max_moves, dtype=np.int8)
    # Flags as bit array
    move_flags = np.zeros(max_moves, dtype=np.uint32)
    # Target type for doubles
    move_target = np.zeros(max_moves, dtype=np.int8)

    category_map = {"status": 0, "physical": 1, "special": 2}

    # Move target mapping for doubles
    target_map = {
        "normal": 0,
        "adjacentfoe": 0,
        "alladjacentfoes": 1,
        "alladjacent": 2,
        "self": 3,
        "adjacentally": 4,
        "adjacentallyorself": 5,
        "any": 6,
        "all": 7,
        "allies": 8,
        "allyside": 9,
        "foeside": 10,
        "scripted": 11,
        "randomnormal": 12,
    }
    flag_bits = {
        "contact": 0,
        "protect": 1,
        "mirror": 2,
        "sound": 3,
        "punch": 4,
        "bite": 5,
        "bullet": 6,
        "pulse": 7,
        "heal": 8,
        "recharge": 9,
        "charge": 10,
        "reflectable": 11,
        "snatch": 12,
        "gravity": 13,
        "dance": 14,
        "wind": 15,
        "slicing": 16,
        "powder": 17,
        "bypasssub": 18,
    }

    move_names = {}

    for move_id, data in moves_list:
        num = data.get("num", 0)
        if num <= 0 or num >= max_moves:
            continue

        move_id_to_idx[move_id] = num
        move_names[num] = data.get("name", move_id)

        move_base_power[num] = data.get("basePower", 0)
        acc = data.get("accuracy", 100)
        move_accuracy[num] = 127 if acc is True else min(int(acc), 100)
        move_pp[num] = data.get("pp", 0)
        move_type[num] = type_to_idx.get(data.get("type", "normal").lower(), 0)
        move_category[num] = category_map.get(data.get("category", "status").lower(), 0)
        move_priority[num] = data.get("priority", 0)
        move_crit_ratio[num] = data.get("critRatio", 1)

        # Target type
        target_str = data.get("target", "normal").lower()
        move_target[num] = target_map.get(target_str, 0)

        # Flags
        flags = data.get("flags", [])
        flag_value = 0
        if isinstance(flags, dict):
            for flag in flags:
                if flag.lower() in flag_bits:
                    flag_value |= 1 << flag_bits[flag.lower()]
        else:
            for flag in flags:
                if flag.lower() in flag_bits:
                    flag_value |= 1 << flag_bits[flag.lower()]
        move_flags[num] = flag_value

    np.save(output_dir / "move_base_power.npy", move_base_power)
    np.save(output_dir / "move_accuracy.npy", move_accuracy)
    np.save(output_dir / "move_pp.npy", move_pp)
    np.save(output_dir / "move_type.npy", move_type)
    np.save(output_dir / "move_category.npy", move_category)
    np.save(output_dir / "move_priority.npy", move_priority)
    np.save(output_dir / "move_crit_ratio.npy", move_crit_ratio)
    np.save(output_dir / "move_flags.npy", move_flags)
    np.save(output_dir / "move_target.npy", move_target)
    print(f"Saved move data for {len(move_id_to_idx)} moves")

    # === Abilities Data ===
    abilities_list = sorted(
        abilities_data.items(), key=lambda x: (x[1].get("num", 9999), x[0])
    )
    ability_id_to_idx = {}
    ability_names = {}
    used_ability_indices: set[int] = set()
    ability_num_totals: dict[int, int] = {}
    for _, data in abilities_list:
        num = data.get("num", 0)
        if num > 0:
            ability_num_totals[num] = ability_num_totals.get(num, 0) + 1

    def _assign_ability(ability_id: str, idx: int, name: str) -> None:
        used_ability_indices.add(idx)
        ability_id_to_idx[ability_id] = idx
        ability_names[idx] = name

    # Pass 1: abilities with a unique Showdown num keep that id (e.g. Sword of Ruin).
    for ability_id, data in abilities_list:
        num = data.get("num", 0)
        if num <= 0 or ability_num_totals[num] != 1:
            continue
        _assign_ability(ability_id, num, data.get("name", ability_id))

    # Pass 2: duplicated nums — first entry keeps num, the rest fill the next gaps.
    seen_num_count: dict[int, int] = {}
    for ability_id, data in abilities_list:
        num = data.get("num", 0)
        if num <= 0 or ability_num_totals[num] == 1:
            continue

        seen_num_count[num] = seen_num_count.get(num, 0) + 1
        if seen_num_count[num] == 1:
            idx = num
        else:
            idx = num + 1
            while idx in used_ability_indices:
                idx += 1
        _assign_ability(ability_id, idx, data.get("name", ability_id))

    max_abilities = max(used_ability_indices) + 1 if used_ability_indices else 310

    print(f"Processed {len(ability_id_to_idx)} abilities")

    # === Species Abilities Array ===
    species_abilities = np.full((max_species, 4), -1, dtype=np.int16)
    ability_slot_keys = ["0", "1", "H", "S"]

    for species_id, data in species_list:
        idx = species_id_to_idx.get(species_id)
        if idx is None or idx >= max_species:
            continue

        abilities = data.get("abilities", {})
        for slot_idx, slot_key in enumerate(ability_slot_keys):
            if slot_key in abilities:
                ability_name = (
                    abilities[slot_key].lower().replace(" ", "").replace("-", "")
                )
                ability_idx = ability_id_to_idx.get(ability_name)
                if ability_idx is not None:
                    species_abilities[idx, slot_idx] = ability_idx

    n_with_abilities = np.sum(species_abilities[:, 0] >= 0)
    np.save(output_dir / "species_abilities.npy", species_abilities)
    print(f"Saved species abilities for {n_with_abilities} Pokemon")

    # === Items Data ===
    items_list = sorted(items_data.items(), key=lambda x: x[1].get("num", 9999))
    item_id_to_idx = {}

    max_items = max(i["num"] for _, i in items_list if i.get("num", 0) > 0) + 1

    item_fling_power = np.zeros(max_items, dtype=np.int16)
    item_is_berry = np.zeros(max_items, dtype=bool)
    item_is_choice = np.zeros(max_items, dtype=bool)

    item_names = {}

    for item_id, data in items_list:
        num = data.get("num", 0)
        if num <= 0 or num >= max_items:
            continue

        item_id_to_idx[item_id] = num
        item_names[num] = data.get("name", item_id)
        item_fling_power[num] = data.get("flingBasePower", 0)
        item_is_berry[num] = data.get("isBerry", False)
        item_is_choice[num] = data.get("isChoice", False)

    np.save(output_dir / "item_fling_power.npy", item_fling_power)
    np.save(output_dir / "item_is_berry.npy", item_is_berry)
    np.save(output_dir / "item_is_choice.npy", item_is_choice)
    print(f"Saved item data for {len(item_id_to_idx)} items")

    # === ID Mappings (JSON) ===
    mappings = {
        "species_id_to_idx": species_id_to_idx,
        "move_id_to_idx": move_id_to_idx,
        "ability_id_to_idx": ability_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
        "species_names": {str(k): v for k, v in species_names.items()},
        "move_names": {str(k): v for k, v in move_names.items()},
        "ability_names": {str(k): v for k, v in ability_names.items()},
        "item_names": {str(k): v for k, v in item_names.items()},
        "type_to_idx": type_to_idx,
    }

    with open(output_dir / "id_mappings.json", "w") as f:
        json.dump(mappings, f, indent=2)

    # pokepy loader optionally patches accuracy/mustpressure from moves.json
    moves_json_path = output_dir.parent / "moves.json"
    moves_json = {}
    for move_id, data in moves_data.items():
        moves_json[move_id] = {
            "num": data.get("num"),
            "accuracy": data.get("accuracy", 100),
            "flags": data.get("flags", {}),
        }
    with open(moves_json_path, "w") as f:
        json.dump(moves_json, f)

    print(f"\nData extraction complete! Output saved to {output_dir}")


def main():
    repo_root = Path(__file__).resolve().parents[4]
    pokepy_data = Path(__file__).resolve().parents[1] / "pokepy" / "data"

    parser = argparse.ArgumentParser(description="Extract Pokemon data from PS")
    parser.add_argument(
        "--ps-path",
        type=Path,
        default=repo_root / "server" / "pokemon-showdown",
        help="Path to Pokemon Showdown directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=pokepy_data / "extracted",
        help="Output directory for NumPy arrays",
    )

    args = parser.parse_args()

    if not args.ps_path.exists():
        print(f"Error: PS path does not exist: {args.ps_path}")
        sys.exit(1)

    print("Extracting Pokemon Showdown data...")
    print(f"PS path: {args.ps_path}")
    print(f"Output: {args.output}")
    print()

    print("Extracting Pokedex...")
    species_data = extract_pokedex(args.ps_path)
    print(f"  Found {len(species_data)} species")

    print("Extracting moves...")
    moves_data = extract_moves(args.ps_path)
    print(f"  Found {len(moves_data)} moves")

    print("Extracting abilities...")
    abilities_data = extract_abilities(args.ps_path)
    print(f"  Found {len(abilities_data)} abilities")

    print("Extracting items...")
    items_data = extract_items(args.ps_path)
    print(f"  Found {len(items_data)} items")

    print("Extracting type chart...")
    typechart = extract_typechart(args.ps_path)
    print(f"  Extracted {len(typechart)} types")

    print("\nCreating NumPy arrays...")
    create_numpy_arrays(
        species_data, moves_data, abilities_data, items_data, typechart, args.output
    )


if __name__ == "__main__":
    main()
