"""Pokepy battle env wrapper.

A thin convenience layer over pokepy.engine.battle_gen9 that:
1. Initializes a `MultiFormatState` from a pair of teams (species/moves/items/
   abilities/evs/tera) by writing the bit-packed flat buffer.
2. Exposes a Gymnasium-style `step()` returning Kakuna-format observations.
3. Provides action masking via pokepy.engine.action_mask.

Designed for single-battle, scalar Python use. Numpy-batched and gymnasium
VectorEnv variants come in phase 10.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pokepy.core.state import MultiFormatState
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    F_CHOICE_LOCK_0,
    F_CHOICE_LOCK_1,
    F_LAST_MOVE_0,
    F_LAST_MOVE_1,
    F_DISABLE_0,
    F_DISABLE_1,
    F_WEATHER,
    F_TERRAIN,
    PHASE_BATTLE,
    PHASE_FORCED_SWITCH,
    NUM_BATTLE_ACTIONS,
    FORMAT_GEN9OU,
)
from pokepy.data.loader import (
    GameData,
    IDMappings,
    load_game_data,
    load_id_mappings,
    load_move_effect_data,
    MoveEffectData,
)
from pokepy.core.gen_profile import GEN9_PROFILE, GenProfile, profile_for_gen
from pokepy.data.type_charts import MODERN_TYPE_CHART, load_type_chart_for_gen
from pokepy.engine import get_engine, step_battle, step_forced_switch_for_gen
from pokepy.engine.action_mask import get_action_mask
from pokepy.mechanics.stats import calc_stat_modern
from pokepy.utils.gen5_prng import Gen5PRNG
from pokepy.obs.kakuna_obs import build_kakuna_obs
from pokepy.obs.state_to_universal import state_to_universal_state

# A trivially valid 1-pokemon team for smoke testing.
DEFAULT_TEAM: Dict[str, Any] = dict(
    species=[1],  # bulbasaur
    moves=[[33, 14, 22, 75]],  # tackle, growl, vine whip, razor leaf-ish
    abilities=[0],
    items=[0],
    tera_types=[4],  # grass
    levels=[100],
)


def _stats_from_resolved_inputs(
    species_id: int,
    level: int,
    game_data: GameData,
    evs: List[int],
    ivs: List[int],
    nature_mods: List[float],
    *,
    gen: int = 9,
) -> List[int]:
    base = game_data.species_base_stats[species_id]
    if gen <= 2:
        # Showdown sim/pokemon.ts: gen<=2 masks IVs with `& 30` (even DV encoding).
        ivs = [int(v) & 30 for v in ivs]
    return [
        int(
            calc_stat_modern(
                int(base[i]),
                level,
                int(ivs[i]),
                int(evs[i]),
                is_hp=(i == 0),
                nature_mult=float(nature_mods[i]),
            )
        )
        for i in range(6)
    ]


def _resolve_stats_inputs(
    evs: Optional[List[int]] = None,
    ivs: Optional[List[int]] = None,
    nature: Optional[Any] = None,
    *,
    use_natures_evs: bool = True,
) -> Tuple[List[int], List[int], List[float]]:
    if not use_natures_evs:
        if evs is None:
            evs = [84] * 6
        if ivs is None:
            ivs = [31] * 6
        return list(evs), list(ivs), [1.0] * 6
    explicit_nature_mods = _nature_to_mods(nature) if nature is not None else None
    if evs is not None and explicit_nature_mods is not None:
        # Caller supplied both EVs and nature — use literally (Showdown parity,
        # scripted teams). Do not snap to the gen9ou spread table.
        if ivs is None:
            ivs = [31] * 6
        return list(evs), list(ivs), list(explicit_nature_mods)
    if evs is None:
        snapped_evs = [84] * 6
        nature_mods = [1.0] * 6
    else:
        from pokepy.data.ev_spreads import EV_SPREADS_ARRAY, NATURE_MODS_ARRAY

        evs_arr = np.asarray(evs, dtype=np.int16).reshape(6)
        diffs = np.abs(EV_SPREADS_ARRAY - evs_arr).sum(axis=1)
        target_mods = explicit_nature_mods
        if target_mods is not None:
            mask = np.all(
                np.isclose(
                    NATURE_MODS_ARRAY, np.asarray(target_mods, dtype=np.float32)
                ),
                axis=1,
            )
            if np.any(mask):
                masked_diffs = np.where(mask, diffs, np.iinfo(np.int32).max)
                idx = int(np.argmin(masked_diffs))
            else:
                idx = int(np.argmin(diffs))
        else:
            idx = int(np.argmin(diffs))
        snapped_evs = EV_SPREADS_ARRAY[idx].tolist()
        nature_mods = NATURE_MODS_ARRAY[idx].tolist()
    if ivs is None:
        ivs = [31] * 6
    return list(snapped_evs), list(ivs), list(nature_mods)


def _stats_for_pokemon(
    species_id: int,
    level: int,
    game_data: GameData,
    evs: Optional[List[int]] = None,
    ivs: Optional[List[int]] = None,
    nature: Optional[Any] = None,
) -> List[int]:
    """Compute the 6 stats from species base stats.

    To produce a canonical stat byte layout, we snap raw EVs to the closest
    gen9ou EV spread from the 50-spread table and use the spread's EVs +
    nature, not the raw EVs. This may deviate from the original Showdown
    team's exact spread but yields a consistent canonical encoding.

    If ``nature`` is supplied (as a nature name like "Modest" / "Jolly" or
    as an explicit 6-tuple of multipliers) we restrict the snap to rows
    whose nature mods match, avoiding the ``np.argmin`` tiebreak that
    otherwise returns the wrong nature when multiple rows share the same
    L1 distance from the input EV tuple.
    """
    snapped_evs, ivs, nature_mods = _resolve_stats_inputs(
        evs=evs, ivs=ivs, nature=nature
    )
    return _stats_from_resolved_inputs(
        species_id,
        level,
        game_data,
        snapped_evs,
        ivs,
        nature_mods,
    )


# Showdown nature table (HP / Atk / Def / SpA / SpD / Spe multipliers).
# Source: pokemon-showdown/data/natures.ts.
_NATURE_TABLE: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "hardy": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "lonely": (1.0, 1.1, 0.9, 1.0, 1.0, 1.0),
    "brave": (1.0, 1.1, 1.0, 1.0, 1.0, 0.9),
    "adamant": (1.0, 1.1, 1.0, 0.9, 1.0, 1.0),
    "naughty": (1.0, 1.1, 1.0, 1.0, 0.9, 1.0),
    "bold": (1.0, 0.9, 1.1, 1.0, 1.0, 1.0),
    "docile": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "relaxed": (1.0, 1.0, 1.1, 1.0, 1.0, 0.9),
    "impish": (1.0, 1.0, 1.1, 0.9, 1.0, 1.0),
    "lax": (1.0, 1.0, 1.1, 1.0, 0.9, 1.0),
    "timid": (1.0, 0.9, 1.0, 1.0, 1.0, 1.1),
    "hasty": (1.0, 1.0, 0.9, 1.0, 1.0, 1.1),
    "serious": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "jolly": (1.0, 1.0, 1.0, 0.9, 1.0, 1.1),
    "naive": (1.0, 1.0, 1.0, 1.0, 0.9, 1.1),
    "modest": (1.0, 0.9, 1.0, 1.1, 1.0, 1.0),
    "mild": (1.0, 1.0, 0.9, 1.1, 1.0, 1.0),
    "quiet": (1.0, 1.0, 1.0, 1.1, 1.0, 0.9),
    "bashful": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "rash": (1.0, 1.0, 1.0, 1.1, 0.9, 1.0),
    "calm": (1.0, 0.9, 1.0, 1.0, 1.1, 1.0),
    "gentle": (1.0, 1.0, 0.9, 1.0, 1.1, 1.0),
    "sassy": (1.0, 1.0, 1.0, 1.0, 1.1, 0.9),
    "careful": (1.0, 1.0, 1.0, 0.9, 1.1, 1.0),
    "quirky": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
}


def _nature_to_mods(nature: Any) -> Optional[Tuple[float, ...]]:
    """Normalize a nature spec (name string or 6-tuple) to a mults tuple."""
    if nature is None:
        return None
    if isinstance(nature, str):
        return _NATURE_TABLE.get(nature.strip().lower())
    try:
        mods = tuple(float(x) for x in nature)
    except (TypeError, ValueError):
        return None
    if len(mods) != 6:
        return None
    return mods


def _write_pokemon(
    battle: np.ndarray,
    side_base: int,
    slot: int,
    *,
    species_id: int,
    level: int,
    type1: int,
    type2: int,
    ability_id: int,
    item_id: int,
    stats: List[int],
    tera_type: int = 0,
) -> None:
    p = side_base + slot * POKEMON_SIZE
    battle[p + 0] = species_id
    battle[p + 1] = stats[0]  # current_hp
    battle[p + 2] = stats[0]  # max_hp
    battle[p + 3] = level
    # Single-typed pokemon: store type2 = type1 so the damage calc's
    # `if def_type2 == def_type1: eff2 = 1.0` path is taken.
    type2_real = type2 if type2 >= 0 else type1
    raw = (type1 & 0xFF) | ((type2_real & 0xFF) << 8)
    battle[p + 4] = np.int16(raw if raw < 0x8000 else raw - 0x10000)
    battle[p + 5] = ability_id
    battle[p + 6] = item_id
    battle[p + 7] = stats[1]
    battle[p + 8] = stats[2]
    battle[p + 9] = stats[3]
    battle[p + 10] = stats[4]
    battle[p + 11] = stats[5]
    battle[p + 12] = 0  # status / status_turns
    battle[p + 13] = 0x6666  # neutral boosts (atk/def/spa/spd)
    # Pack neutral boosts (spe/acc/eva) into bits 0-11 and tera_type into bits 12-15.
    # Tera_type 0-17 fits in 4 bits if we mask it; some types like Stellar (18) won't fit.
    tera_nibble = (int(tera_type) & 0xF) if tera_type >= 0 else 0
    boosts14_raw = 0x0666 | (tera_nibble << 12)
    battle[p + 14] = np.int16(
        boosts14_raw if boosts14_raw < 0x8000 else boosts14_raw - 0x10000
    )
    # Pokemon flags layout :
    #   bit 0 (0x1):  fainted
    #   bit 1 (0x2):  is_active (unused at init)
    #   bit 3 (0x8):  tera_used (set when tera is activated mid-battle)
    #   bit 6 (0x40): disguise intact — ONLY set if ability is Disguise
    #   bit 7 (0x80): had_item on current entry — for Unburden
    #   bit 8 (0x100): once-per-battle ability triggered
    #   bit 9 (0x200): Flash Fire activated
    #   bit 5 (0x20): Charge / Electromorphosis / Wind Power volatile active
    #   bit 12 (0x1000): Booster Energy paradox boost active on this entry
    # NOTE: Showdown does NOT use a separate "has_tera" bit (0x4); tera
    # availability is inferred from `tera_used` not being set.
    from pokepy.core.constants import ABILITY_DISGUISE

    flags = 0
    if ability_id == ABILITY_DISGUISE:
        flags |= 0x40  # disguise intact (only on Disguise users)
    if item_id > 0:
        flags |= 0x80  # had_item at start
    battle[p + 15] = flags


def init_battle_state(
    team0: Dict[str, Any],
    team1: Dict[str, Any],
    game_data: Optional[GameData] = None,
    seed: int = 12345,
    gen: int = 9,
) -> MultiFormatState:
    """Build a fully-initialized MultiFormatState ready for battle stepping."""
    profile = profile_for_gen(gen)
    if game_data is None:
        game_data = load_game_data(gen=gen)

    state = MultiFormatState.create_empty(format_id=profile.format_id)
    state.phase = np.int8(PHASE_BATTLE)
    state.gen5_seed = np.uint64(seed)

    def _write_team(team: Dict[str, Any], state_team_arrs, side_base: int):
        (
            species,
            moves,
            items,
            abilities,
            tera,
            pp_arr,
            evs_full,
            ivs_full,
            nature_mods_full,
        ) = state_team_arrs
        team_evs = team.get("evs")
        team_ivs = team.get("ivs")
        team_natures = team.get("natures")
        for i, sp in enumerate(team["species"]):
            species[i] = sp
            moves[i] = team["moves"][i]
            ability_id = 0 if not profile.has_abilities else team["abilities"][i]
            item_id = 0 if not profile.has_items else team["items"][i]
            items[i] = item_id
            abilities[i] = ability_id
            tera[i] = team["tera_types"][i]
            for j, mid in enumerate(team["moves"][i]):
                if mid >= 0:
                    # Showdown competitive teams have max PP (3 PP Ups = base * 1.6)
                    base_pp = int(np.asarray(game_data.move_pp)[mid])
                    pp_arr[i, j] = (base_pp * 8) // 5  # 1.6x rounded down
            level = team["levels"][i]
            evs_i = team_evs[i] if team_evs is not None and i < len(team_evs) else None
            ivs_i = team_ivs[i] if team_ivs is not None and i < len(team_ivs) else None
            nat_i = (
                team_natures[i]
                if team_natures is not None and i < len(team_natures)
                else None
            )
            resolved_evs, resolved_ivs, resolved_nature_mods = _resolve_stats_inputs(
                evs=evs_i,
                ivs=ivs_i,
                nature=nat_i,
                use_natures_evs=profile.has_natures_evs,
            )
            stats = _stats_from_resolved_inputs(
                sp,
                level,
                game_data,
                resolved_evs,
                resolved_ivs,
                resolved_nature_mods,
                gen=profile.gen,
            )
            evs_full[i] = np.asarray(resolved_evs, dtype=np.int16)
            ivs_full[i] = np.asarray(resolved_ivs, dtype=np.int16)
            nature_mods_full[i] = np.asarray(resolved_nature_mods, dtype=np.float32)
            types = game_data.species_types[sp]
            type1, type2 = int(types[0]), int(types[1])
            _write_pokemon(
                state.battle_state,
                side_base,
                i,
                species_id=sp,
                level=level,
                type1=type1,
                type2=type2,
                ability_id=ability_id,
                item_id=item_id,
                stats=stats,
                tera_type=team["tera_types"][i],
            )

    _write_team(
        team0,
        (
            state.team_species,
            state.team_moves,
            state.team_items,
            state.team_abilities,
            state.team_tera,
            state.team_pp,
            state.team_evs_full,
            state.team_ivs_full,
            state.team_nature_mods,
        ),
        OFF_SIDE0,
    )
    _write_team(
        team1,
        (
            state.opp_species,
            state.opp_moves,
            state.opp_items,
            state.opp_abilities,
            state.opp_tera,
            state.opp_pp,
            state.opp_evs_full,
            state.opp_ivs_full,
            state.opp_nature_mods,
        ),
        OFF_SIDE1,
    )

    bs = state.battle_state
    bs[OFF_META + M_ACTIVE0] = 0
    bs[OFF_META + M_ACTIVE1] = 0
    # Charging-move sentinel = -1 (not charging). Default 0 would make the
    # engine think the pokemon was charging move 0 last turn.
    from pokepy.core.constants import (
        M_CHARGING_0,
        M_CHARGING_1,
        M_LOCKED_MOVE_0,
        M_LOCKED_MOVE_1,
        M_LOCKED_TURNS_0,
        M_LOCKED_TURNS_1,
        M_RECHARGE_0,
        M_RECHARGE_1,
        M_ACTIVE_MOVE_ACTIONS_0,
        M_ACTIVE_MOVE_ACTIONS_1,
        OFF_MOVES,
    )

    bs[OFF_META + M_CHARGING_0] = -1
    bs[OFF_META + M_CHARGING_1] = -1
    # Lockedmove (Outrage / Petal Dance / Thrash / Raging Fury) and
    # mustrecharge (Hyper Beam family) state lives in OFF_MOVES.
    bs[OFF_MOVES + M_LOCKED_MOVE_0] = -1
    bs[OFF_MOVES + M_LOCKED_MOVE_1] = -1
    bs[OFF_MOVES + M_LOCKED_TURNS_0] = 0
    bs[OFF_MOVES + M_LOCKED_TURNS_1] = 0
    bs[OFF_MOVES + M_RECHARGE_0] = 0
    bs[OFF_MOVES + M_RECHARGE_1] = 0
    bs[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = 0
    bs[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = 0
    # Side-condition sentinel fields also default to -1
    for off in (
        F_CHOICE_LOCK_0,
        F_CHOICE_LOCK_1,
        F_LAST_MOVE_0,
        F_LAST_MOVE_1,
        F_DISABLE_0,
        F_DISABLE_1,
    ):
        bs[OFF_FIELD + off] = -1

    # Trigger switch-in abilities for the leadoff active mons. This sets
    # weather (Drought/Drizzle/Sand Stream/Snow Warning/etc.), terrain
    # (Electric/Grassy/Psychic/Misty Surge), Intimidate boosts, and Trace.
    # Showdown applies these on entry; pokepy must do it for the leads too,
    # not only on subsequent switches.
    # Showdown also burns hidden PRNG frames before turn 1:
    #   1. team-preview commitChoices queue.sort
    #   2. lead switchIn queue.insertChoice(runSwitch) on tied lead speeds
    #   3. startup runSwitch speedSort(allActive) on tied lead speeds
    #   4. any startup WeatherChange / TerrainChange eachEvent speedSorts
    #   5. post-runSwitch eachEvent('Update') on tied lead speeds
    # Record that startup sequence here so step_battle_gen9 can replay the
    # exact same PRNG calls on turn 0 before the first visible action.
    from pokepy.effects.abilities import apply_switch_in_ability
    from pokepy.effects import get_effective_speed
    from pokepy.engine.battle_gen9 import _consume_team_preview_queue_sort_frames

    class _RecordingPRNG:
        def __init__(self, base_prng: Gen5PRNG):
            self._prng = base_prng
            self.calls: List[Tuple[int, ...]] = []

        def random(self, *args):
            self.calls.append(tuple(int(a) for a in args))
            return self._prng.random(*args)

        def __getattr__(self, name):
            return getattr(self._prng, name)

    seed_lo = seed & 0xFFFF
    seed_hi = (seed >> 16) & 0xFFFF
    startup_prng = _RecordingPRNG(Gen5PRNG((seed_lo, seed_hi, 0, 0)))
    if profile.has_teampreview:
        _consume_team_preview_queue_sort_frames(bs, startup_prng)

    p0_off = OFF_SIDE0  # active = slot 0
    p1_off = OFF_SIDE1
    sp0 = get_effective_speed(bs, p0_off)
    sp1 = get_effective_speed(bs, p1_off)
    if sp0 == sp1:
        # BattleActions.switchIn inserts the second runSwitch action into a
        # 1-entry tied window, then BattleActions.runSwitch speedSorts the two
        # tied actives again before SwitchIn handlers fire.
        startup_prng.random(0, 2)
        startup_switch_roll = int(startup_prng.random(0, 2))
        p0_first = startup_switch_roll == 0
    else:
        p0_first = sp0 > sp1
    # Fog of war: reveal a lead's ability if its switch-in handler announced
    # (set weather/terrain, changed its own boosts/ability/paradox state, or
    # dropped the opponent's Attack via Intimidate). Mirrors the in-battle
    # reveal logic in battle_gen9. Leads are slot 0 of each side.
    from pokepy.core.bitpack import extract_boost as _extract_boost

    def _lead_sig(off: int) -> tuple:
        return (
            int(bs[off + 13]),
            int(bs[off + 14]) & 0x0FFF,
            int(bs[off + 5]),
            int(bs[off + 1]),
            int(bs[off + 12]),
            int(bs[off + 15]) & 0x7210,
        )

    def _lead_boosts(off: int) -> tuple:
        return (int(bs[off + 13]), int(bs[off + 14]))

    def _lead_boost_changed(before: tuple, after: tuple, *, raised: bool) -> bool:
        for shift in (0, 4, 8, 12):
            b, a = _extract_boost(before[0], shift), _extract_boost(after[0], shift)
            if (a > b) if raised else (a < b):
                return True
        for shift in (0, 4, 8):
            b, a = _extract_boost(before[1], shift), _extract_boost(after[1], shift)
            if (a > b) if raised else (a < b):
                return True
        return False

    def _lead_abil_arr(off: int):
        return (
            state.team_abilities_revealed
            if off < OFF_SIDE1
            else state.opp_abilities_revealed
        )

    def _apply_lead_ability(switcher_off: int, opp_off: int) -> None:
        field_b = (int(bs[OFF_FIELD + F_WEATHER]), int(bs[OFF_FIELD + F_TERRAIN]))
        sw_b = _lead_sig(switcher_off)
        opp_b = _lead_boosts(opp_off)
        apply_switch_in_ability(
            bs,
            switcher_off,
            opp_off,
            did_switch=True,
            gen5_prng=startup_prng,
            has_terrain=profile.has_terrain,
            ability_weather_limited=profile.ability_weather_limited,
        )
        field_a = (int(bs[OFF_FIELD + F_WEATHER]), int(bs[OFF_FIELD + F_TERRAIN]))
        sw_a = _lead_sig(switcher_off)
        opp_a = _lead_boosts(opp_off)
        if (
            field_a != field_b
            or sw_a != sw_b
            or _lead_boost_changed(opp_b, opp_a, raised=False)
        ):
            _lead_abil_arr(switcher_off)[0] = True
        if sw_a[2] != sw_b[2]:
            _lead_abil_arr(opp_off)[0] = True
        if _lead_boost_changed(opp_b, opp_a, raised=True):
            _lead_abil_arr(opp_off)[0] = True

    if profile.has_abilities:
        if p0_first:
            _apply_lead_ability(p0_off, p1_off)
            _apply_lead_ability(p1_off, p0_off)
        else:
            _apply_lead_ability(p1_off, p0_off)
            _apply_lead_ability(p0_off, p1_off)
    if sp0 == sp1:
        # Showdown's startup runSwitch + SwitchIn fieldEvent path can burn
        # multiple tied speedSort frames after the insertChoice roll. Formats
        # without teampreview skip the preview queue.sort but still spend the
        # third turn-0 shuffle before the post-runSwitch Update frame below.
        if not profile.has_teampreview:
            startup_prng.random(0, 2)
        startup_prng.random(0, 2)
    state.startup_prng_calls = tuple(startup_prng.calls)

    # Mark the leadoff opponent as revealed . The obs
    # adapter doesn't directly read opp_revealed, but downstream code might.
    state.opp_revealed[0] = True
    # Symmetric init for the side-1-perspective view of side 0.
    state.team_revealed[0] = True

    return state


class BattleEnv:
    """Single-battle gymnasium-style env wrapper."""

    def __init__(
        self,
        game_data: Optional[GameData] = None,
        mappings: Optional[IDMappings] = None,
        move_effects: Optional[MoveEffectData] = None,
        seed: int = 12345,
        gen: int = 9,
    ):
        self.gen = int(gen)
        self.profile = profile_for_gen(self.gen)
        self.game_data = game_data or load_game_data(gen=self.gen)
        self.mappings = mappings or load_id_mappings(gen=self.gen)
        self.move_effects = move_effects or load_move_effect_data(gen=self.gen)
        self.type_chart = load_type_chart_for_gen(self.gen)
        self.seed = seed
        self.prng = Gen5PRNG((seed & 0xFFFF, (seed >> 16) & 0xFFFF, 0, 0))
        self.state: Optional[MultiFormatState] = None

    def reset(
        self,
        team0: Dict[str, Any] = DEFAULT_TEAM,
        team1: Dict[str, Any] = DEFAULT_TEAM,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Build a fresh battle state.

        Showdown `Battle.start()` creates a fresh PRNG from the battle seed
        (sim/battle.ts:1859). Pokepy must do the same: re-seed `self.prng`
        to the initial seed on every `reset()` so repeated episodes are
        independent and replayable, matching Showdown's behavior.
        """
        if seed is not None:
            self.seed = int(seed)
        self.prng = Gen5PRNG((self.seed & 0xFFFF, (self.seed >> 16) & 0xFFFF, 0, 0))
        self.state = init_battle_state(
            team0, team1, self.game_data, self.seed, gen=self.gen
        )
        return self.observe(side=0)

    def observe(self, side: int = 0) -> Dict[str, np.ndarray]:
        assert self.state is not None
        us = state_to_universal_state(
            self.state, self.game_data, self.mappings, player_side=side
        )
        mask = get_action_mask(self.state, side, self.game_data)
        # Pad to 13-action space (Kakuna's DefaultActionSpace)
        illegal13 = np.ones(13, dtype=np.bool_)
        illegal13[:NUM_BATTLE_ACTIONS] = ~mask  # legal -> not illegal
        return build_kakuna_obs(us, illegal_actions=illegal13)

    def step(
        self, action0: int, action1: int, *, tera0: bool = False, tera1: bool = False
    ) -> Tuple[Dict[str, np.ndarray], float, float, bool]:
        """Step the env. If `state.phase` is FORCED_SWITCH, action0 must be a
        switch action (4..9); the engine handles it via `step_forced_switch`
        and stays on the same turn (no opponent action consumed).
        """
        assert self.state is not None
        if int(self.state.phase) == PHASE_FORCED_SWITCH:
            r0, r1, done = step_forced_switch_for_gen(
                self.gen,
                self.state,
                action0,
                side=0,
                game_data=self.game_data,
                move_effects=self.move_effects,
                type_chart=self.type_chart,
                gen5_prng=self.prng,
            )
        else:
            r0, r1, done = step_battle(
                self.gen,
                self.state,
                action0,
                action1,
                self.game_data,
                self.move_effects,
                self.type_chart,
                self.prng,
                wants_tera0=tera0 if self.profile.has_tera else False,
                wants_tera1=tera1 if self.profile.has_tera else False,
            )
        obs = self.observe(side=0)
        return obs, float(r0), float(r1), bool(done)

    def get_action_mask(self, side: int = 0) -> np.ndarray:
        return get_action_mask(self.state, side, self.game_data)
