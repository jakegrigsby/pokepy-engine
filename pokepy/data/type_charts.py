"""Type effectiveness charts.

Verbatim Convention: chart[defender_type, attacker_type] = damage_multiplier.
"""

from __future__ import annotations

import numpy as np

from pokepy.core.constants import (
    TYPE_NORMAL,
    TYPE_FIRE,
    TYPE_WATER,
    TYPE_ELECTRIC,
    TYPE_GRASS,
    TYPE_ICE,
    TYPE_FIGHTING,
    TYPE_POISON,
    TYPE_GROUND,
    TYPE_FLYING,
    TYPE_PSYCHIC,
    TYPE_BUG,
    TYPE_ROCK,
    TYPE_GHOST,
    TYPE_DRAGON,
    TYPE_DARK,
    TYPE_STEEL,
    TYPE_FAIRY,
)


def create_modern_type_chart() -> np.ndarray:
    """Create modern (Gen 6+) type chart. Shape (19, 19) float32."""
    chart = np.ones((19, 19), dtype=np.float32)

    # Immunities
    chart[TYPE_NORMAL, TYPE_GHOST] = 0.0
    chart[TYPE_GHOST, TYPE_NORMAL] = 0.0
    chart[TYPE_GHOST, TYPE_FIGHTING] = 0.0
    chart[TYPE_FLYING, TYPE_GROUND] = 0.0
    chart[TYPE_GROUND, TYPE_ELECTRIC] = 0.0
    chart[TYPE_DARK, TYPE_PSYCHIC] = 0.0
    chart[TYPE_FAIRY, TYPE_DRAGON] = 0.0
    chart[TYPE_STEEL, TYPE_POISON] = 0.0

    # Normal (attacker) — Rock and Steel RESIST Normal.
    # Showdown data/typechart.ts rock.damageTaken.Normal=2,
    # steel.damageTaken.Normal=2 (2 = resisted in Showdown encoding).
    chart[TYPE_ROCK, TYPE_NORMAL] = 0.5
    chart[TYPE_STEEL, TYPE_NORMAL] = 0.5

    # Fire
    chart[TYPE_GRASS, TYPE_FIRE] = 2.0
    chart[TYPE_ICE, TYPE_FIRE] = 2.0
    chart[TYPE_BUG, TYPE_FIRE] = 2.0
    chart[TYPE_STEEL, TYPE_FIRE] = 2.0
    chart[TYPE_FIRE, TYPE_FIRE] = 0.5
    chart[TYPE_WATER, TYPE_FIRE] = 0.5
    chart[TYPE_ROCK, TYPE_FIRE] = 0.5
    chart[TYPE_DRAGON, TYPE_FIRE] = 0.5

    # Water
    chart[TYPE_FIRE, TYPE_WATER] = 2.0
    chart[TYPE_GROUND, TYPE_WATER] = 2.0
    chart[TYPE_ROCK, TYPE_WATER] = 2.0
    chart[TYPE_WATER, TYPE_WATER] = 0.5
    chart[TYPE_GRASS, TYPE_WATER] = 0.5
    chart[TYPE_DRAGON, TYPE_WATER] = 0.5

    # Electric
    chart[TYPE_WATER, TYPE_ELECTRIC] = 2.0
    chart[TYPE_FLYING, TYPE_ELECTRIC] = 2.0
    chart[TYPE_ELECTRIC, TYPE_ELECTRIC] = 0.5
    chart[TYPE_GRASS, TYPE_ELECTRIC] = 0.5
    chart[TYPE_DRAGON, TYPE_ELECTRIC] = 0.5

    # Grass
    chart[TYPE_WATER, TYPE_GRASS] = 2.0
    chart[TYPE_GROUND, TYPE_GRASS] = 2.0
    chart[TYPE_ROCK, TYPE_GRASS] = 2.0
    chart[TYPE_FIRE, TYPE_GRASS] = 0.5
    chart[TYPE_GRASS, TYPE_GRASS] = 0.5
    chart[TYPE_POISON, TYPE_GRASS] = 0.5
    chart[TYPE_FLYING, TYPE_GRASS] = 0.5
    chart[TYPE_BUG, TYPE_GRASS] = 0.5
    chart[TYPE_DRAGON, TYPE_GRASS] = 0.5
    chart[TYPE_STEEL, TYPE_GRASS] = 0.5

    # Ice
    chart[TYPE_GRASS, TYPE_ICE] = 2.0
    chart[TYPE_GROUND, TYPE_ICE] = 2.0
    chart[TYPE_FLYING, TYPE_ICE] = 2.0
    chart[TYPE_DRAGON, TYPE_ICE] = 2.0
    chart[TYPE_FIRE, TYPE_ICE] = 0.5
    chart[TYPE_WATER, TYPE_ICE] = 0.5
    chart[TYPE_ICE, TYPE_ICE] = 0.5
    chart[TYPE_STEEL, TYPE_ICE] = 0.5

    # Fighting
    chart[TYPE_NORMAL, TYPE_FIGHTING] = 2.0
    chart[TYPE_ICE, TYPE_FIGHTING] = 2.0
    chart[TYPE_ROCK, TYPE_FIGHTING] = 2.0
    chart[TYPE_DARK, TYPE_FIGHTING] = 2.0
    chart[TYPE_STEEL, TYPE_FIGHTING] = 2.0
    chart[TYPE_POISON, TYPE_FIGHTING] = 0.5
    chart[TYPE_FLYING, TYPE_FIGHTING] = 0.5
    chart[TYPE_PSYCHIC, TYPE_FIGHTING] = 0.5
    chart[TYPE_BUG, TYPE_FIGHTING] = 0.5
    chart[TYPE_FAIRY, TYPE_FIGHTING] = 0.5

    # Poison
    chart[TYPE_GRASS, TYPE_POISON] = 2.0
    chart[TYPE_FAIRY, TYPE_POISON] = 2.0
    chart[TYPE_POISON, TYPE_POISON] = 0.5
    chart[TYPE_GROUND, TYPE_POISON] = 0.5
    chart[TYPE_ROCK, TYPE_POISON] = 0.5
    chart[TYPE_GHOST, TYPE_POISON] = 0.5

    # Ground
    chart[TYPE_FIRE, TYPE_GROUND] = 2.0
    chart[TYPE_ELECTRIC, TYPE_GROUND] = 2.0
    chart[TYPE_POISON, TYPE_GROUND] = 2.0
    chart[TYPE_ROCK, TYPE_GROUND] = 2.0
    chart[TYPE_STEEL, TYPE_GROUND] = 2.0
    chart[TYPE_GRASS, TYPE_GROUND] = 0.5
    chart[TYPE_BUG, TYPE_GROUND] = 0.5

    # Flying
    chart[TYPE_GRASS, TYPE_FLYING] = 2.0
    chart[TYPE_FIGHTING, TYPE_FLYING] = 2.0
    chart[TYPE_BUG, TYPE_FLYING] = 2.0
    chart[TYPE_ELECTRIC, TYPE_FLYING] = 0.5
    chart[TYPE_ROCK, TYPE_FLYING] = 0.5
    chart[TYPE_STEEL, TYPE_FLYING] = 0.5

    # Psychic
    chart[TYPE_FIGHTING, TYPE_PSYCHIC] = 2.0
    chart[TYPE_POISON, TYPE_PSYCHIC] = 2.0
    chart[TYPE_PSYCHIC, TYPE_PSYCHIC] = 0.5
    chart[TYPE_STEEL, TYPE_PSYCHIC] = 0.5

    # Bug
    chart[TYPE_GRASS, TYPE_BUG] = 2.0
    chart[TYPE_PSYCHIC, TYPE_BUG] = 2.0
    chart[TYPE_DARK, TYPE_BUG] = 2.0
    chart[TYPE_FIRE, TYPE_BUG] = 0.5
    chart[TYPE_FIGHTING, TYPE_BUG] = 0.5
    chart[TYPE_POISON, TYPE_BUG] = 0.5
    chart[TYPE_FLYING, TYPE_BUG] = 0.5
    chart[TYPE_GHOST, TYPE_BUG] = 0.5
    chart[TYPE_STEEL, TYPE_BUG] = 0.5
    chart[TYPE_FAIRY, TYPE_BUG] = 0.5

    # Rock
    chart[TYPE_FIRE, TYPE_ROCK] = 2.0
    chart[TYPE_ICE, TYPE_ROCK] = 2.0
    chart[TYPE_FLYING, TYPE_ROCK] = 2.0
    chart[TYPE_BUG, TYPE_ROCK] = 2.0
    chart[TYPE_FIGHTING, TYPE_ROCK] = 0.5
    chart[TYPE_GROUND, TYPE_ROCK] = 0.5
    chart[TYPE_STEEL, TYPE_ROCK] = 0.5

    # Ghost
    chart[TYPE_PSYCHIC, TYPE_GHOST] = 2.0
    chart[TYPE_GHOST, TYPE_GHOST] = 2.0
    chart[TYPE_DARK, TYPE_GHOST] = 0.5

    # Dragon
    chart[TYPE_DRAGON, TYPE_DRAGON] = 2.0
    chart[TYPE_STEEL, TYPE_DRAGON] = 0.5

    # Dark
    chart[TYPE_PSYCHIC, TYPE_DARK] = 2.0
    chart[TYPE_GHOST, TYPE_DARK] = 2.0
    chart[TYPE_FIGHTING, TYPE_DARK] = 0.5
    chart[TYPE_DARK, TYPE_DARK] = 0.5
    chart[TYPE_FAIRY, TYPE_DARK] = 0.5

    # Steel
    chart[TYPE_ICE, TYPE_STEEL] = 2.0
    chart[TYPE_ROCK, TYPE_STEEL] = 2.0
    chart[TYPE_FAIRY, TYPE_STEEL] = 2.0
    chart[TYPE_FIRE, TYPE_STEEL] = 0.5
    chart[TYPE_WATER, TYPE_STEEL] = 0.5
    chart[TYPE_ELECTRIC, TYPE_STEEL] = 0.5
    chart[TYPE_STEEL, TYPE_STEEL] = 0.5

    # Fairy
    chart[TYPE_FIGHTING, TYPE_FAIRY] = 2.0
    chart[TYPE_DRAGON, TYPE_FAIRY] = 2.0
    chart[TYPE_DARK, TYPE_FAIRY] = 2.0
    chart[TYPE_FIRE, TYPE_FAIRY] = 0.5
    chart[TYPE_POISON, TYPE_FAIRY] = 0.5
    chart[TYPE_STEEL, TYPE_FAIRY] = 0.5

    return chart


MODERN_TYPE_CHART = create_modern_type_chart()

_TYPE_CHART_CACHE: dict[int, np.ndarray] = {9: MODERN_TYPE_CHART}


def _type_chart_from_showdown_json(gen: int) -> np.ndarray:
    """Build type chart from metamon showdown_dex static JSON."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[4] / "backend" / "showdown_dex" / "static"
    chart_path = root / "typechart" / f"gen{gen}typechart.json"
    if not chart_path.exists():
        raise FileNotFoundError(chart_path)
    with open(chart_path) as f:
        json_chart = json.load(f)

    from pokepy.core.constants import (
        TYPE_NORMAL,
        TYPE_FIRE,
        TYPE_WATER,
        TYPE_ELECTRIC,
        TYPE_GRASS,
        TYPE_ICE,
        TYPE_FIGHTING,
        TYPE_POISON,
        TYPE_GROUND,
        TYPE_FLYING,
        TYPE_PSYCHIC,
        TYPE_BUG,
        TYPE_ROCK,
        TYPE_GHOST,
        TYPE_DRAGON,
        TYPE_DARK,
        TYPE_STEEL,
        TYPE_FAIRY,
    )

    name_to_id = {
        "NORMAL": TYPE_NORMAL,
        "FIRE": TYPE_FIRE,
        "WATER": TYPE_WATER,
        "ELECTRIC": TYPE_ELECTRIC,
        "GRASS": TYPE_GRASS,
        "ICE": TYPE_ICE,
        "FIGHTING": TYPE_FIGHTING,
        "POISON": TYPE_POISON,
        "GROUND": TYPE_GROUND,
        "FLYING": TYPE_FLYING,
        "PSYCHIC": TYPE_PSYCHIC,
        "BUG": TYPE_BUG,
        "ROCK": TYPE_ROCK,
        "GHOST": TYPE_GHOST,
        "DRAGON": TYPE_DRAGON,
        "DARK": TYPE_DARK,
        "STEEL": TYPE_STEEL,
        "FAIRY": TYPE_FAIRY,
    }
    chart = np.ones((19, 19), dtype=np.float32)
    for def_name, data in json_chart.items():
        def_id = name_to_id.get(str(def_name).upper())
        if def_id is None:
            continue
        for atk_name, taken in data.get("damageTaken", {}).items():
            atk_id = name_to_id.get(str(atk_name).upper())
            if atk_id is None:
                continue
            if taken == 3:
                chart[def_id, atk_id] = 0.0
            elif taken == 1:
                chart[def_id, atk_id] = 2.0
            elif taken == 2:
                chart[def_id, atk_id] = 0.5
            else:
                chart[def_id, atk_id] = 1.0
    return chart


def load_type_chart_for_gen(gen: int = 9) -> np.ndarray:
    """Return type effectiveness chart for ``gen`` (cached).

    Prefer metamon ``showdown_dex/static/typechart/gen{N}typechart.json`` when
    present — it matches Showdown's live Dex.forGen(N) effectiveness. The
    legacy ``type_chart.npy`` extracted from PS mod diffs can drift (e.g. gen4
    Psychic self-resist missing).
    """
    gen = int(gen)
    if gen in _TYPE_CHART_CACHE:
        return _TYPE_CHART_CACHE[gen]
    if gen == 9:
        chart = MODERN_TYPE_CHART
    else:
        try:
            chart = _type_chart_from_showdown_json(gen)
        except FileNotFoundError:
            try:
                from pokepy.data.loader import load_game_data

                chart = load_game_data(gen=gen).type_chart
            except FileNotFoundError:
                if gen >= 6:
                    chart = MODERN_TYPE_CHART
                else:
                    raise
    _TYPE_CHART_CACHE[gen] = chart
    return chart
