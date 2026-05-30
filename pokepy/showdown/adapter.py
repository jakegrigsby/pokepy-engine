"""Adapter: drive pokepy.showdown.Battle for parity diffing and (Phase B) UniversalState.

Converts the parity harness's id-based team dict into engine pokemon sets,
runs battles with the same deterministic heuristic as ``parity_heuristic_e2e``,
and snapshots per-turn HP/status rows for Showdown comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pokepy.core.constants import NUM_BATTLE_ACTIONS
from pokepy.showdown.battle import Battle

_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")

# pokepy.showdown status id -> parity-row status code (matches the harness).
_STATUS_CODE = {"": 0, "brn": 1, "par": 2, "slp": 3, "frz": 4, "psn": 5, "tox": 6}


def team_dict_to_sets(team: Dict[str, Any], mappings) -> List[Dict[str, Any]]:
    """Convert the harness id-based team dict to engine pokemon sets."""
    species = team["species"]
    moves = team["moves"]
    abilities = team.get("abilities")
    items = team.get("items")
    levels = team.get("levels")
    evs = team.get("evs")
    ivs = team.get("ivs")
    natures = team.get("natures")
    genders = team.get("genders")
    tera = team.get("tera_types")

    sets: List[Dict[str, Any]] = []
    for i, sp in enumerate(species):
        pset: Dict[str, Any] = {
            "species": mappings.species_names.get(int(sp), str(sp)),
            "level": int(levels[i]) if levels is not None else 100,
        }
        move_names = []
        for mid in moves[i]:
            if int(mid) >= 0:
                move_names.append(mappings.move_names.get(int(mid), str(mid)))
        pset["moves"] = move_names

        if abilities is not None:
            abil_id = int(abilities[i])
            if abil_id > 0:
                pset["ability"] = mappings.ability_names.get(abil_id, "")
        if items is not None:
            item_id = int(items[i])
            if item_id > 0:
                pset["item"] = mappings.item_names.get(item_id, "")
        if evs is not None and i < len(evs):
            pset["evs"] = {_STAT_KEYS[j]: int(evs[i][j]) for j in range(6)}
        if ivs is not None and i < len(ivs):
            pset["ivs"] = {_STAT_KEYS[j]: int(ivs[i][j]) for j in range(6)}
        if natures is not None and i < len(natures):
            pset["nature"] = str(natures[i]).strip() or "Serious"
        if genders is not None and i < len(genders):
            pset["gender"] = str(genders[i]).strip().upper()[:1]
        if tera is not None and i < len(tera):
            pset["teraType"] = mappings.type_names.get(int(tera[i])) if hasattr(mappings, "type_names") else None
        sets.append(pset)
    return sets


def _status_code(pokemon) -> int:
    if pokemon is None:
        return 0
    return _STATUS_CODE.get(pokemon.status, 0)


def _active(side) -> Optional[Any]:
    return side.active[0] if side.active else None


def get_battle_action_mask(battle: Battle, side_n: int) -> List[bool]:
    """Legal 13-action mask: 4 moves + up to 6 switches (matches old packed engine)."""
    side = battle.sides[side_n]
    active = _active(side)
    mask = [False] * NUM_BATTLE_ACTIONS
    if active and active.hp > 0 and not active.fainted and active.is_active:
        for i, slot in enumerate(active.move_slots[:4]):
            if slot["pp"] > 0 and not slot.get("disabled"):
                mask[i] = True
    for i, pokemon in enumerate(side.pokemon):
        if i + 4 >= NUM_BATTLE_ACTIONS:
            break
        if not pokemon.fainted and not pokemon.is_active:
            mask[4 + i] = True
    return mask


def action_index_to_showdown_str(action_idx: int, *, forced: bool = False) -> str:
    if action_idx < 4:
        base = f"move {action_idx + 1}"
    else:
        base = f"switch {action_idx - 4 + 1}"
    return f"{base}+forced" if forced else base


def action_index_to_choice(battle: Battle, side_n: int, action_idx: int) -> Dict[str, Any]:
    side = battle.sides[side_n]
    if action_idx < 4:
        active = _active(side)
        slot = active.move_slots[action_idx]
        return {"choice": "move", "move": slot["id"]}
    team_idx = action_idx - 4
    return {"choice": "switch", "slot": team_idx}


def simple_heuristic_action(
    battle: Battle,
    side_n: int,
    mask: Optional[List[bool]] = None,
) -> int:
    if mask is None:
        mask = get_battle_action_mask(battle, side_n)
    legal = [i for i in range(NUM_BATTLE_ACTIONS) if mask[i]]
    if not legal:
        return 0
    moves = [a for a in legal if a < 4]
    if moves:
        return moves[0]
    return legal[0]


def simple_heuristic_forced_switch(battle: Battle, side_n: int) -> int:
    mask = get_battle_action_mask(battle, side_n)
    switches = [a for a in range(4, NUM_BATTLE_ACTIONS) if mask[a]]
    return switches[0] if switches else 4


def _side_needs_forced_switch(battle: Battle, side_n: int) -> bool:
    side = battle.sides[side_n]
    if battle.ended or side.pokemon_left() == 0:
        return False
    active = _active(side)
    if active is None:
        return True
    if active.fainted or active.hp <= 0:
        return True
    if not active.is_active:
        return True
    return False


def _forced_switch_mask(battle: Battle) -> int:
    """0 = p0 only, 1 = p1 only, 2 = both (matches old packed engine)."""
    p0 = _side_needs_forced_switch(battle, 0)
    p1 = _side_needs_forced_switch(battle, 1)
    if p0 and p1:
        return 2
    if p1:
        return 1
    if p0:
        return 0
    return -1


def _commit_forced_switch(battle: Battle, side_n: int, action_idx: int) -> None:
    side = battle.sides[side_n]
    team_idx = action_idx - 4
    incoming = side.pokemon[team_idx]
    pos = 0
    battle.actions.switch_in(incoming, pos)
    while battle.queue.list:
        action = battle.queue.shift()
        battle.run_action(action)
        if battle.ended:
            return
    battle.faint_messages()
    battle.check_win()


def _resolve_forced_switches(
    battle: Battle,
    p0_actions: List[str],
    p1_actions: List[str],
    choice_log: Optional[List[Tuple[str, str]]],
) -> None:
    while True:
        mask = _forced_switch_mask(battle)
        if mask < 0 or battle.ended:
            return
        if mask in (1, 2):
            fa1 = simple_heuristic_forced_switch(battle, 1)
            action1 = action_index_to_showdown_str(fa1, forced=True)
            p1_actions.append(action1)
            if choice_log is not None:
                choice_log.append(("p2", action1))
            _commit_forced_switch(battle, 1, fa1)
        if battle.ended:
            return
        mask = _forced_switch_mask(battle)
        if mask not in (0, 2):
            return
        fa0 = simple_heuristic_forced_switch(battle, 0)
        action0 = action_index_to_showdown_str(fa0, forced=True)
        p0_actions.append(action0)
        if choice_log is not None:
            choice_log.append(("p1", action0))
        _commit_forced_switch(battle, 0, fa0)


def _snapshot_row(battle: Battle, turn: int, a0: str, a1: str) -> Dict[str, Any]:
    p0 = _active(battle.sides[0])
    p1 = _active(battle.sides[1])
    return {
        "type": "normal",
        "turn": turn,
        "p0_hp": p0.hp if p0 else 0,
        "p0_max_hp": p0.maxhp if p0 else 0,
        "p0_status": _status_code(p0),
        "p1_hp": p1.hp if p1 else 0,
        "p1_max_hp": p1.maxhp if p1 else 0,
        "p1_status": _status_code(p1),
        "p0_action": a0,
        "p1_action": a1,
    }


def run_pokepy_battle(
    team0: Dict[str, Any],
    team1: Dict[str, Any],
    game_data,
    move_effects,
    mappings,
    type_chart,
    seed: int,
    n_turns: int,
    gen: int = 9,
    *,
    choice_log: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Drive the verbatim engine with the parity harness heuristic policy."""
    del game_data, move_effects, type_chart  # unused; kept for harness signature
    seed_tuple = (seed & 0xFFFF, (seed >> 16) & 0xFFFF, 0, 0)
    battle = Battle(
        gen=gen,
        seed=seed_tuple,
        p1_team=team_dict_to_sets(team0, mappings),
        p2_team=team_dict_to_sets(team1, mappings),
    )
    battle.start()

    rows: List[Dict[str, Any]] = []
    p0_actions: List[str] = []
    p1_actions: List[str] = []

    while len(rows) < int(n_turns) and not battle.ended:
        if _forced_switch_mask(battle) >= 0:
            _resolve_forced_switches(battle, p0_actions, p1_actions, choice_log)
            continue

        p0 = _active(battle.sides[0])
        p1 = _active(battle.sides[1])
        if p0 is None or p1 is None or p0.fainted or p1.fainted:
            break

        mask0 = get_battle_action_mask(battle, 0)
        mask1 = get_battle_action_mask(battle, 1)
        a0_idx = simple_heuristic_action(battle, 0, mask0)
        a1_idx = simple_heuristic_action(battle, 1, mask1)
        a0 = action_index_to_showdown_str(a0_idx)
        a1 = action_index_to_showdown_str(a1_idx)
        p0_actions.append(a0)
        p1_actions.append(a1)
        if choice_log is not None:
            choice_log.append(("p1", a0))
            choice_log.append(("p2", a1))

        played_turn = battle.turn
        battle.choose(
            action_index_to_choice(battle, 0, a0_idx),
            action_index_to_choice(battle, 1, a1_idx),
        )

        _resolve_forced_switches(battle, p0_actions, p1_actions, choice_log)

        if battle.turn <= 0:
            continue
        rows.append(_snapshot_row(battle, played_turn, a0, a1))

    return rows, p0_actions, p1_actions


def run_engine_battle(
    team0: Dict[str, Any],
    team1: Dict[str, Any],
    mappings,
    seed: int,
    n_turns: int,
    *,
    gen: int = 9,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Slice helper: same as run_pokepy_battle without unused harness args."""
    return run_pokepy_battle(
        team0,
        team1,
        None,
        None,
        mappings,
        None,
        seed,
        n_turns,
        gen=gen,
    )


def showdown_state_to_universal(battle: Battle, *, perspective: str = "p1"):
    """Produce a metamon ``UniversalState`` from a ported ``Battle`` (Phase B3)."""
    raise NotImplementedError(
        "showdown_state_to_universal is Phase B (todo B3). "
        "Use run_pokepy_battle for parity diffing until then."
    )
