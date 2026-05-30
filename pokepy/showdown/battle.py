"""Battle: the this-context + event dispatch + API shim + turn loop.

Near-verbatim port of the slice-relevant core of sim/battle.ts. Translated
effect callbacks run with ``this`` = this Battle (passed as the first Python
arg), so they read like the TypeScript source.

Scope of this Phase-A slice core: singles, gen9 base path, the dispatch engine
(run_event / single_event / find_event_handlers / resolve_priority /
speed_sort), the numeric API shim (modify/chainModify/randomizer/boost/damage/
heal/faint), and a turn loop that resolves a queue of move/switch actions in
speed order. Breadth (the full callback universe) is Phase B.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pokepy.showdown import util
from pokepy.showdown.battle_actions import BattleActions
from pokepy.showdown.battle_queue import ACTION_ORDER, Action, BattleQueue
from pokepy.showdown.dex import get_dex, to_id
from pokepy.showdown.field import Field
from pokepy.showdown.side import Side
from pokepy.showdown.pokemon import Pokemon
from pokepy.utils.gen5_prng import Gen5PRNG

_BIG_ORDER = 4294967296  # 2**32, Showdown's "no order" sentinel


class _EventCtx(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


class _Format:
    """Minimal Format/ruleTable stand-in for singles OU."""

    effectType = "Format"
    id = "format"
    name = "format"
    num = 0

    def __init__(self):
        self.handlers: Dict[str, Any] = {}


class _RuleTable:
    def has(self, _rule: str) -> bool:
        return False


class Battle:
    NOT_FAIL = ""

    def __init__(
        self,
        gen: int = 9,
        seed: Optional[Sequence[int]] = None,
        p1_team: Optional[List[Dict[str, Any]]] = None,
        p2_team: Optional[List[Dict[str, Any]]] = None,
        p1_name: str = "Player 1",
        p2_name: str = "Player 2",
    ):
        self.gen = int(gen)
        self.dex = get_dex(self.gen)
        self.format = _Format()
        self.format_data: Dict[str, Any] = {"id": "format"}
        self.rule_table = _RuleTable()
        self.game_type = "singles"
        self.active_per_half = 1

        if seed is None:
            seed = (12345, 12345, 12345, 12345)
        self.prng = Gen5PRNG(tuple(seed))
        self.initial_seed = tuple(seed)

        self.field = Field(self)
        self.sides: List[Side] = []
        if p1_team is not None and p2_team is not None:
            s1 = Side(self, 0, p1_team, p1_name)
            s2 = Side(self, 1, p2_team, p2_name)
            s1.foe = s2
            s2.foe = s1
            self.sides = [s1, s2]

        self.queue = BattleQueue(self)
        self.actions = BattleActions(self)

        # Dispatch context.
        self.event: _EventCtx = _EventCtx(id="", target=None, source=None, effect=None, modifier=1)
        self.effect: Any = None
        self.effect_state: Dict[str, Any] = {}
        self.event_depth = 0
        self.events: Optional[Dict[str, Any]] = None
        self.speed_order: List[int] = []

        # Move-in-progress context.
        self.active_move: Any = None
        self.active_pokemon: Optional[Pokemon] = None
        self.active_target: Optional[Pokemon] = None
        self.last_successful_move_this_turn: Any = None

        # Bookkeeping.
        self.turn = 0
        self.mid_turn = False
        self.started = False
        self.ended = False
        self.winner: Optional[str] = None
        self.log: List[str] = []
        self.sent_log_pos = 0
        self.input_log: List[str] = []
        self.faint_queue: List[Dict[str, Any]] = []
        self.quick_claw_roll: Optional[bool] = None

    # ================================================================== #
    # Numeric primitives (sim/battle.ts arithmetic).
    # ================================================================== #
    def trunc(self, num, bits: int = 0) -> int:
        return util.trunc(num, bits)

    def clamp_int_range(self, num, min_val=None, max_val=None) -> int:
        return util.clamp_int_range(num, min_val, max_val)

    def modify(self, value, numerator, denominator: int = 1) -> int:
        return util.modify(value, numerator, denominator)

    def chain(self, previous_mod, next_mod) -> float:
        return util.chain(previous_mod, next_mod)

    def chain_modify(self, numerator, denominator: int = 1):
        previous_mod = util.trunc(self.event.modifier * 4096)
        if isinstance(numerator, (list, tuple)):
            denominator = numerator[1]
            numerator = numerator[0]
        next_mod = util.trunc(numerator * 4096 / denominator)
        self.event.modifier = ((previous_mod * next_mod + 2048) >> 12) / 4096

    chainModify = chain_modify  # verbatim alias

    def final_modify(self, relay_var):
        relay_var = self.modify(relay_var, self.event.modifier)
        self.event.modifier = 1
        return relay_var

    finalModify = final_modify

    def randomizer(self, base_damage: int) -> int:
        return util.trunc(util.trunc(base_damage * (100 - self.random(16))) / 100)

    def get_category(self, move) -> str:
        if isinstance(move, str):
            return self.dex.moves.get(move).category or "Physical"
        return getattr(move, "category", None) or "Physical"

    getCategory = get_category

    # -- stat calc (sim/battle.ts:2358) -------------------------------- #
    def spread_modify(self, base_stats: Dict[str, int], pset: Dict[str, Any]) -> Dict[str, int]:
        return {st: self.stat_modify(base_stats, pset, st) for st in ("hp", "atk", "def", "spa", "spd", "spe")}

    def stat_modify(self, base_stats: Dict[str, int], pset: Dict[str, Any], stat_name: str) -> int:
        tr = util.trunc
        stat = base_stats[stat_name]
        ivs = pset.get("ivs") or {}
        evs = pset.get("evs") or {}
        iv = int(ivs.get(stat_name, 31))
        ev = int(evs.get(stat_name, 0))
        if self.gen <= 2:
            # Showdown sim/pokemon.ts: DVs are even IVs in gen1/2.
            iv = iv & 30
        level = int(pset.get("level", 100))
        if stat_name == "hp":
            if base_stats[stat_name] == 1:  # Shedinja
                return 1
            return tr(tr(2 * stat + iv + tr(ev / 4) + 100) * level / 100 + 10)
        stat = tr(tr(2 * stat + iv + tr(ev / 4)) * level / 100 + 5)
        nature = self.dex.natures.get(pset.get("nature", "Serious"))
        if nature and nature.plus == stat_name:
            stat = tr(tr(stat * 110, 16) / 100)
        elif nature and nature.minus == stat_name:
            stat = tr(tr(stat * 90, 16) / 100)
        return stat

    # ================================================================== #
    # PRNG shim (sim/prng.ts via Gen5PRNG).
    # ================================================================== #
    def random(self, from_val=None, to_val=None):
        return self.prng.random(from_val, to_val)

    def random_chance(self, numerator: int, denominator: int) -> bool:
        return self.prng.random_chance(numerator, denominator)

    randomChance = random_chance

    def sample(self, items: Sequence):
        if not items:
            raise ValueError("Cannot sample an empty array")
        return items[self.random(len(items))]

    # ================================================================== #
    # Speed sort + comparators (sim/battle.ts:404-460).
    # ================================================================== #
    @staticmethod
    def compare_priority(a: Dict[str, Any], b: Dict[str, Any]) -> int:
        return (
            -((b.get("order") or _BIG_ORDER) - (a.get("order") or _BIG_ORDER))
            or ((b.get("priority") or 0) - (a.get("priority") or 0))
            or ((b.get("speed") or 0) - (a.get("speed") or 0))
            or -((b.get("subOrder") or 0) - (a.get("subOrder") or 0))
            or -((b.get("effectOrder") or 0) - (a.get("effectOrder") or 0))
            or 0
        )

    @staticmethod
    def compare_left_to_right_order(a: Dict[str, Any], b: Dict[str, Any]) -> int:
        return (
            -((b.get("order") or _BIG_ORDER) - (a.get("order") or _BIG_ORDER))
            or ((b.get("priority") or 0) - (a.get("priority") or 0))
            or -((b.get("index") or 0) - (a.get("index") or 0))
            or 0
        )

    def speed_sort(self, lst: List[Any], comparator=None):
        """Selection sort that resolves speed ties via prng.shuffle (battle.ts:429)."""
        if comparator is None:
            comparator = self.compare_priority
        if len(lst) < 2:
            return
        sorted_n = 0
        while sorted_n + 1 < len(lst):
            next_indexes = [sorted_n]
            for i in range(sorted_n + 1, len(lst)):
                delta = comparator(self._cmp_view(lst[next_indexes[0]]), self._cmp_view(lst[i]))
                if delta < 0:
                    continue
                if delta > 0:
                    next_indexes = [i]
                if delta == 0:
                    next_indexes.append(i)
            for i in range(len(next_indexes)):
                index = next_indexes[i]
                if index != sorted_n + i:
                    lst[sorted_n + i], lst[index] = lst[index], lst[sorted_n + i]
            if len(next_indexes) > 1:
                self._prng_shuffle(lst, sorted_n, sorted_n + len(next_indexes))
            sorted_n += len(next_indexes)

    @staticmethod
    def _cmp_view(item) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        # speed-sorting Pokemon (eachEvent): expose {speed}
        return {"speed": getattr(item, "speed", 0)}

    def _prng_shuffle(self, items: List[Any], start: int, end: int):
        """Fisher-Yates matching sim/prng.ts shuffle (consumes frames)."""
        while start < end - 1:
            next_index = self.random(start, end)
            if start != next_index:
                items[start], items[next_index] = items[next_index], items[start]
            start += 1

    # ================================================================== #
    # Event dispatch (sim/battle.ts:465-1231).
    # ================================================================== #
    def init_effect_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data.setdefault("id", "")
        return data

    def get_callback(self, target, effect, callback_name: str):
        if not effect:
            return None
        handlers = getattr(effect, "handlers", None) or {}
        callback = handlers.get(callback_name)
        if (
            callback is None
            and isinstance(target, Pokemon)
            and self.gen >= 5
            and callback_name == "onSwitchIn"
            and not handlers.get("onAnySwitchIn")
            and getattr(effect, "effectType", None) in ("Ability", "Item")
        ):
            callback = handlers.get("onStart")
        return callback

    def resolve_priority(self, h: Dict[str, Any], callback_name: str) -> Dict[str, Any]:
        effect = h["effect"]
        handlers = getattr(effect, "handlers", None) or {}
        h["order"] = handlers.get(f"{callback_name}Order") or False
        h["priority"] = handlers.get(f"{callback_name}Priority") or 0
        h["subOrder"] = handlers.get(f"{callback_name}SubOrder") or 0
        if not h["subOrder"]:
            effect_type_order = {"Condition": 2, "Weather": 5, "Ability": 7, "Item": 8}
            h["subOrder"] = effect_type_order.get(getattr(effect, "effectType", None), 0)
        holder = h.get("effectHolder")
        if isinstance(holder, Pokemon) and hasattr(holder, "get_stat"):
            h["speed"] = holder.speed
            if callback_name.endswith("SwitchIn"):
                fpv = holder.get_field_position_value()
                idx = self.speed_order.index(fpv) if fpv in self.speed_order else 0
                h["speed"] -= idx / (self.active_per_half * 2)
        return h

    def find_pokemon_event_handlers(self, pokemon: Pokemon, callback_name: str, get_key=None) -> List[Dict[str, Any]]:
        handlers: List[Dict[str, Any]] = []
        status = pokemon.get_status()
        cb = self.get_callback(pokemon, status, callback_name)
        if cb is not None or (get_key and pokemon.status_state.get(get_key)):
            handlers.append(self.resolve_priority(
                {"effect": status, "callback": cb, "state": pokemon.status_state,
                 "end": pokemon.clear_status, "effectHolder": pokemon}, callback_name))
        for vid, vstate in list(pokemon.volatiles.items()):
            volatile = self.dex.conditions.get_by_id(vid)
            cb = self.get_callback(pokemon, volatile, callback_name)
            if cb is not None or (get_key and vstate.get(get_key)):
                handlers.append(self.resolve_priority(
                    {"effect": volatile, "callback": cb, "state": vstate,
                     "end": None, "effectHolder": pokemon}, callback_name))
        ability = pokemon.get_ability()
        cb = self.get_callback(pokemon, ability, callback_name)
        if cb is not None or (get_key and pokemon.ability_state.get(get_key)):
            handlers.append(self.resolve_priority(
                {"effect": ability, "callback": cb, "state": pokemon.ability_state,
                 "end": pokemon.clear_status, "effectHolder": pokemon}, callback_name))
        item = pokemon.get_item()
        cb = self.get_callback(pokemon, item, callback_name)
        if cb is not None or (get_key and pokemon.item_state.get(get_key)):
            handlers.append(self.resolve_priority(
                {"effect": item, "callback": cb, "state": pokemon.item_state,
                 "end": None, "effectHolder": pokemon}, callback_name))
        species = pokemon.base_species
        cb = self.get_callback(pokemon, species, callback_name)
        if cb is not None:
            handlers.append(self.resolve_priority(
                {"effect": species, "callback": cb, "state": pokemon.species_state,
                 "end": None, "effectHolder": pokemon}, callback_name))
        return handlers

    def find_side_event_handlers(self, side: Side, callback_name: str, get_key=None, custom_holder=None) -> List[Dict[str, Any]]:
        handlers: List[Dict[str, Any]] = []
        for cid, cdata in list(side.side_conditions.items()):
            cond = self.dex.conditions.get_by_id(cid)
            cb = self.get_callback(side, cond, callback_name)
            if cb is not None or (get_key and cdata.get(get_key)):
                handlers.append(self.resolve_priority(
                    {"effect": cond, "callback": cb, "state": cdata,
                     "end": None, "effectHolder": custom_holder or side}, callback_name))
        return handlers

    def find_field_event_handlers(self, field: Field, callback_name: str, get_key=None, custom_holder=None) -> List[Dict[str, Any]]:
        handlers: List[Dict[str, Any]] = []
        for pid, pdata in list(field.pseudo_weather.items()):
            cond = self.dex.conditions.get_by_id(pid)
            cb = self.get_callback(field, cond, callback_name)
            if cb is not None or (get_key and pdata.get(get_key)):
                handlers.append(self.resolve_priority(
                    {"effect": cond, "callback": cb, "state": pdata,
                     "end": None, "effectHolder": custom_holder or field}, callback_name))
        weather = field.get_weather()
        cb = self.get_callback(field, weather, callback_name)
        if cb is not None or (get_key and field.weather_state.get(get_key)):
            handlers.append(self.resolve_priority(
                {"effect": weather, "callback": cb, "state": field.weather_state,
                 "end": None, "effectHolder": custom_holder or field}, callback_name))
        terrain = field.get_terrain()
        cb = self.get_callback(field, terrain, callback_name)
        if cb is not None or (get_key and field.terrain_state.get(get_key)):
            handlers.append(self.resolve_priority(
                {"effect": terrain, "callback": cb, "state": field.terrain_state,
                 "end": None, "effectHolder": custom_holder or field}, callback_name))
        return handlers

    def find_battle_event_handlers(self, callback_name: str, get_key=None, custom_holder=None) -> List[Dict[str, Any]]:
        handlers: List[Dict[str, Any]] = []
        fmt = self.format
        cb = (fmt.handlers or {}).get(callback_name)
        if cb is not None or (get_key and self.format_data.get(get_key)):
            handlers.append(self.resolve_priority(
                {"effect": fmt, "callback": cb, "state": self.format_data,
                 "end": None, "effectHolder": custom_holder or self}, callback_name))
        return handlers

    def find_event_handlers(self, target, event_name: str, source=None) -> List[Dict[str, Any]]:
        handlers: List[Dict[str, Any]] = []
        if isinstance(target, list):
            for i, pokemon in enumerate(target):
                cur = self.find_event_handlers(pokemon, event_name, source)
                for h in cur:
                    h["target"] = pokemon
                    h["index"] = i
                handlers += cur
            return handlers
        should_bubble_down = isinstance(target, Side)
        prefixed = event_name not in ("BeforeTurn", "Update", "Weather", "WeatherChange", "TerrainChange")
        if isinstance(target, Pokemon) and (target.is_active or (source and getattr(source, "is_active", False))):
            handlers = self.find_pokemon_event_handlers(target, f"on{event_name}")
            if prefixed:
                for ally in target.alliesAndSelf():
                    handlers += self.find_pokemon_event_handlers(ally, f"onAlly{event_name}")
                    handlers += self.find_pokemon_event_handlers(ally, f"onAny{event_name}")
                for foe in target.foes():
                    handlers += self.find_pokemon_event_handlers(foe, f"onFoe{event_name}")
                    handlers += self.find_pokemon_event_handlers(foe, f"onAny{event_name}")
            target = target.side
        if source and prefixed and isinstance(source, Pokemon):
            handlers += self.find_pokemon_event_handlers(source, f"onSource{event_name}")
        if isinstance(target, Side):
            for side in self.sides:
                if should_bubble_down:
                    for active in side.active:
                        if not active:
                            continue
                        if side is target or side is target.ally_side:
                            handlers += self.find_pokemon_event_handlers(active, f"on{event_name}")
                        elif prefixed:
                            handlers += self.find_pokemon_event_handlers(active, f"onFoe{event_name}")
                        if prefixed:
                            handlers += self.find_pokemon_event_handlers(active, f"onAny{event_name}")
                if side.n < 2 or not side.ally_side:
                    if side is target or side is target.ally_side:
                        handlers += self.find_side_event_handlers(side, f"on{event_name}")
                    elif prefixed:
                        handlers += self.find_side_event_handlers(side, f"onFoe{event_name}")
                    if prefixed:
                        handlers += self.find_side_event_handlers(side, f"onAny{event_name}")
        handlers += self.find_field_event_handlers(self.field, f"on{event_name}")
        handlers += self.find_battle_event_handlers(f"on{event_name}")
        return handlers

    def single_event(self, eventid, effect, state, target, source=None, source_effect=None, relay_var=None, custom_callback=None):
        if self.event_depth >= 8:
            raise RuntimeError("Stack overflow")
        has_relay_var = True
        if relay_var is None:
            relay_var = True
            has_relay_var = False
        et = getattr(effect, "effectType", None)
        if et == "Status" and isinstance(target, Pokemon) and target.status != getattr(effect, "id", None):
            return relay_var
        handlers = getattr(effect, "handlers", None) or {}
        callback = custom_callback or handlers.get(f"on{eventid}")
        if callback is None:
            return relay_var
        parent_effect, parent_state, parent_event = self.effect, self.effect_state, self.event
        self.effect = effect
        self.effect_state = state or self.init_effect_state({})
        self.event = _EventCtx(id=eventid, target=target, source=source, effect=source_effect, modifier=1)
        self.event_depth += 1
        args = [target, source, source_effect]
        if has_relay_var:
            args.insert(0, relay_var)
        if callable(callback):
            return_val = callback(self, *args)
        else:
            return_val = callback
        self.event_depth -= 1
        self.effect, self.effect_state, self.event = parent_effect, parent_state, parent_event
        return relay_var if return_val is None else return_val

    singleEvent = single_event

    def run_event(self, eventid, target=None, source=None, source_effect=None, relay_var=None, on_effect=None, fast_exit=None):
        if self.event_depth >= 8:
            raise RuntimeError("Stack overflow")
        if not target:
            target = self
        effect_source = source if isinstance(source, Pokemon) else None
        handlers = self.find_event_handlers(target, eventid, effect_source)
        if on_effect and source_effect is not None:
            cb = (getattr(source_effect, "handlers", None) or {}).get(f"on{eventid}")
            if cb is not None:
                handlers.insert(0, self.resolve_priority(
                    {"effect": source_effect, "callback": cb, "state": self.init_effect_state({}),
                     "end": None, "effectHolder": target}, f"on{eventid}"))
        if eventid in ("Invulnerability", "TryHit", "DamagingHit", "EntryHazard"):
            handlers.sort(key=_cmp_key(self.compare_left_to_right_order))
        elif fast_exit:
            handlers.sort(key=_cmp_key(self.compare_left_to_right_order))
        else:
            self.speed_sort(handlers)
        has_relay_var = 1
        args = [target, source, source_effect]
        if relay_var is None:
            relay_var = True
            has_relay_var = 0
        else:
            args.insert(0, relay_var)
        parent_event = self.event
        self.event = _EventCtx(id=eventid, target=target, source=source, effect=source_effect, modifier=1)
        self.event_depth += 1
        for handler in handlers:
            effect = handler["effect"]
            effect_holder = handler.get("effectHolder")
            et = getattr(effect, "effectType", None)
            if et == "Status" and getattr(effect_holder, "status", None) != getattr(effect, "id", None):
                continue
            cb = handler.get("callback")
            if callable(cb):
                parent_effect, parent_state = self.effect, self.effect_state
                self.effect = effect
                self.effect_state = handler.get("state") or self.init_effect_state({})
                self.effect_state["target"] = effect_holder
                return_val = cb(self, *args)
                self.effect, self.effect_state = parent_effect, parent_state
            else:
                return_val = cb
            if return_val is not None:
                relay_var = return_val
                if not relay_var or fast_exit:
                    break
                if has_relay_var:
                    args[0] = relay_var
        self.event_depth -= 1
        if isinstance(relay_var, int) and not isinstance(relay_var, bool) and relay_var == abs(int(relay_var)):
            relay_var = self.modify(relay_var, self.event.modifier)
        self.event = parent_event
        return relay_var

    runEvent = run_event

    def priority_event(self, eventid, target, source=None, effect=None, relay_var=None, on_effect=None):
        return self.run_event(eventid, target, source, effect, relay_var, on_effect, True)

    priorityEvent = priority_event

    def each_event(self, eventid, effect=None, relay_var=None):
        actives = self.get_all_active()
        if not effect and self.effect:
            effect = self.effect
        self.speed_sort(actives, lambda a, b: (b.get("speed") or 0) - (a.get("speed") or 0))
        for pokemon in actives:
            self.run_event(eventid, pokemon, None, effect, relay_var)

    eachEvent = each_event

    # ================================================================== #
    # Suppression helpers.
    # ================================================================== #
    def suppressing_ability(self, target=None) -> bool:
        return False  # Mold Breaker / Neutralizing Gas: Phase B.

    suppressingAbility = suppressing_ability

    # ================================================================== #
    # API shim: logging + damage/heal/boost/faint.
    # ================================================================== #
    def add(self, *args):
        self.log.append("|" + "|".join(self._stringify(a) for a in args))

    def add_move(self, *args):
        self.add(*args)

    addMove = add_move

    def attr_last_move(self, *args):
        pass

    attrLastMove = attr_last_move

    def hint(self, *args, **kwargs):
        pass

    def debug(self, *args):
        pass

    @staticmethod
    def _stringify(a) -> str:
        if isinstance(a, Pokemon):
            return str(a)
        if isinstance(a, Side):
            return a.id
        return str(a)

    def set_active_move(self, move, pokemon=None, target=None):
        self.active_move = move
        self.active_pokemon = pokemon
        self.active_target = target

    setActiveMove = set_active_move

    def clear_active_move(self, failed=False):
        if self.active_move:
            if not failed:
                self.last_move = self.active_move
            self.active_move = None
            self.active_pokemon = None
            self.active_target = None

    clearActiveMove = clear_active_move

    def get_all_active(self, include_fainted=False) -> List[Pokemon]:
        out = []
        for side in self.sides:
            for p in side.active:
                if p and (include_fainted or not p.fainted):
                    out.append(p)
        return out

    getAllActive = get_all_active

    def boost(self, boosts: Dict[str, int], target=None, source=None, effect=None, is_secondary=False, is_self=False) -> bool:
        if target is None:
            target = self.event.target
        if not target or not target.hp:
            return False
        success = False
        for stat, amount in boosts.items():
            delta = target.boost_by({stat: amount})
            if delta != 0:
                success = True
                self.add("-boost" if amount > 0 else "-unboost", target, stat, abs(delta))
        return success

    def damage(self, amount, target=None, source=None, effect=None, instafaint=False):
        if target is None:
            target = self.event.target
        if not target or not target.hp:
            return 0
        if amount is None or amount is False:
            return amount
        amount = self.run_event("Damage", target, source, effect, amount)
        if not amount:
            return amount
        dealt = target.damage(amount, source, effect)
        if effect and getattr(effect, "effectType", None) == "Move":
            target.last_damage = dealt
        self.add("-damage", target, self._health(target))
        if not target.hp:
            self.faint(target, source, effect)
        return dealt

    def spread_damage(self, damages, targets=None, source=None, effect=None):
        # Slice: single-target wrapper.
        if not isinstance(damages, list):
            return self.damage(damages, targets, source, effect)
        return [self.damage(d, t, source, effect) for d, t in zip(damages, targets)]

    spreadDamage = spread_damage

    def direct_damage(self, amount, target=None, source=None, effect=None):
        if target is None:
            target = self.event.target
        if not target or not target.hp or amount <= 0:
            return 0
        amount = self.clamp_int_range(amount, 1)
        dealt = target.damage(amount, source, effect)
        self.add("-damage", target, self._health(target))
        if not target.hp:
            self.faint(target, source, effect)
        return dealt

    directDamage = direct_damage

    def heal(self, amount, target=None, source=None, effect=None):
        if target is None:
            target = self.event.target
        if amount and amount <= 1:
            amount = 1
        amount = self.trunc(amount)
        amount = self.run_event("TryHeal", target, source, effect, amount)
        if not amount:
            return amount
        if not target or not target.hp or not target.hp >= 0:
            return False
        if target.hp >= target.maxhp:
            return False
        final = target.heal(amount, source, effect)
        self.add("-heal", target, self._health(target))
        self.run_event("Heal", target, source, effect, final)
        return final

    @staticmethod
    def _health(target: Pokemon) -> str:
        if target.hp <= 0:
            return "0 fnt"
        return f"{target.hp}/{target.maxhp}"

    def faint(self, target: Pokemon, source=None, effect=None):
        if target.fainted or target.faint_queued:
            return False
        target.faint_queued = True
        self.faint_queue.append({"target": target, "source": source, "effect": effect})
        return True

    def faint_messages(self, last_first=False, force_check=False) -> bool:
        any_fainted = False
        while self.faint_queue:
            item = self.faint_queue.pop(0)
            target = item["target"]
            if target.fainted or target.hp > 0:
                continue
            target.fainted = True
            target.hp = 0
            target.status = ""
            target.is_active = False
            target.active = False
            if target.side.active and target.side.active[0] is target:
                pass
            self.add("faint", target)
            target.side.faint_counter += 1
            any_fainted = True
        return any_fainted

    faintMessages = faint_messages

    def check_win(self):
        sides_alive = [s for s in self.sides if s.pokemon_left() > 0]
        if len(sides_alive) <= 1:
            self.ended = True
            self.winner = sides_alive[0].name if sides_alive else None
        return self.ended

    checkWin = check_win

    # -- targeting (singles) ------------------------------------------- #
    def get_target(self, pokemon: Pokemon, move, target_loc, original_target=None):
        foe = pokemon.side.foe
        return foe.active[0] if foe and foe.active else None

    getTarget = get_target

    def get_random_target(self, pokemon: Pokemon, move):
        foe = pokemon.side.foe
        return foe.active[0] if foe and foe.active else None

    getRandomTarget = get_random_target

    # ================================================================== #
    # Turn loop (sim/battle.ts start / commitChoices / turnLoop / runAction).
    # ================================================================== #
    def get_action_speed(self, action: Action):
        """Set action.speed from its pokemon (sim/battle.ts:getActionSpeed)."""
        pokemon = action.get("pokemon")
        if pokemon is not None:
            action["speed"] = pokemon.get_action_speed()
        elif action.get("speed") is None:
            action["speed"] = 0
        return action

    def start(self):
        if self.started:
            return
        self.started = True
        self.add("gen", self.gen)
        self.add("tier", self.format.name)
        # runPickTeam: gen5+ team preview queues one 'team' action per mon per
        # side (all order 1 -> they tie in the first sort -> one shuffle frame).
        if self.gen >= 5:
            self._run_pick_team()
        self.queue.add_choice(Action(choice="start"))
        self.mid_turn = True
        if self.gen >= 5:
            # Team preview is its own commit; process the startup pseudo-turn now
            # so we land on "turn 1 awaiting moves" exactly like Showdown.
            self._commit()
        else:
            self.turn_loop()

    def _run_pick_team(self):
        for side in self.sides:
            for i, _pokemon in enumerate(side.pokemon):
                self.queue.add_choice(Action(choice="team", pokemon=side.pokemon[i], index=i,
                                             priority=-i))

    def _update_speed(self):
        for p in self.get_all_active():
            p.speed = p.get_action_speed()
        actives = self.get_all_active()
        self.speed_order = [p.get_field_position_value() for p in actives]

    def choose(self, p1_action: Dict[str, Any], p2_action: Dict[str, Any]):
        """Commit one move/switch decision per side (a turn's worth of choices)."""
        self.queue.clear()
        actions: List[Action] = []
        for side, action in ((self.sides[0], p1_action), (self.sides[1], p2_action)):
            actions.append(self._build_action(side, action))
        for a in actions:
            self.queue.add_choice(a)
        self._commit()

    def _build_action(self, side: Side, action: Dict[str, Any]) -> Action:
        pokemon = side.active[0]
        choice = action.get("choice", "move")
        if choice == "switch":
            return Action(choice="switch", pokemon=pokemon, slot=action.get("slot"))
        move = self.dex.moves.get(action.get("move"))
        priority = move.priority or 0
        priority = self.run_event("ModifyPriority", pokemon, None, move, priority)
        return Action(choice="move", pokemon=pokemon, move=move, moveid=move.id, priority=priority)

    def _commit(self):
        """sim/battle.ts commitChoices: updateSpeed, sort, turnLoop.

        The turn's choices were already resolved into ``queue.list`` by the
        caller (start/_run_pick_team or choose). Showdown saves any mid-turn
        remainder (empty at turn start) and re-sorts; we sort in place. Mid-turn
        switch saving is Phase B.
        """
        self._update_speed()
        self.queue.sort()
        self.turn_loop()

    def turn_loop(self):
        self.add("")
        if not self.mid_turn:
            self.queue.insert_choice(Action(choice="beforeTurn"))
            self.queue.add_choice(Action(choice="residual"))
            self.mid_turn = True
        while self.queue:
            action = self.queue.shift()
            self.run_action(action)
            if self.ended:
                return
        self.end_turn()
        self.mid_turn = False
        self.queue.clear()

    def run_action(self, action: Action):
        choice = action["choice"]
        residual_pokemon: List = []

        if choice == "start":
            self.add("start")
            for side in self.sides:
                for i in range(len(side.active)):
                    self.actions.switch_in(side.pokemon[i], i)
            self.mid_turn = True
        elif choice == "move":
            if not action["pokemon"].is_active:
                return False
            if action["pokemon"].fainted:
                return False
            self.actions.run_move(action["move"], action["pokemon"], action.get("targetLoc") or 0,
                                  target=action.get("originalTarget"))
        elif choice == "team":
            action["pokemon"].position = action["index"]
            return
        elif choice == "runSwitch":
            self.actions.run_switch(action["pokemon"])
        elif choice == "beforeTurn":
            self.each_event("BeforeTurn")
        elif choice == "switch" or choice == "instaswitch":
            self.actions.switch_in(self._switch_target(action), action["pokemon"].position)
        elif choice == "residual":
            self.add("")
            self.clear_active_move(True)
            self._update_speed()
            residual_pokemon = [(p, p.hp) for p in self.get_all_active()]
            self.field_event("Residual")
            if not self.ended:
                self.add("upkeep")

        # ---- common per-action tail (sim/battle.ts runAction tail) ---- #
        self.clear_active_move()
        self.faint_messages()
        if self.ended:
            return True

        peek = self.queue.peek()
        if not peek or (self.gen <= 3 and peek.get("choice") in ("move", "residual")):
            self.check_win()

        if self.gen >= 5 and choice != "start":
            self.each_event("Update")

        if self.gen < 5:
            self.each_event("Update")

        if self.gen >= 8 and peek and peek.get("choice") in ("move", "runDynamax"):
            self._update_speed()
            for qa in self.queue.list:
                if qa.get("pokemon"):
                    self.get_action_speed(qa)
            self.queue.sort()
        return False

    def _switch_target(self, action: Action) -> Pokemon:
        pokemon = action["pokemon"]
        slot = action.get("slot")
        return pokemon.side.pokemon[slot] if slot is not None else pokemon

    def field_event(self, eventid, targets=None):
        """sim/battle.ts fieldEvent: run handlers on field + all actives (speed sorted)."""
        get_key = "duration" if eventid == "Residual" else None
        handlers = self.find_field_event_handlers(self.field, f"onField{eventid}", get_key)
        for side in self.sides:
            if side.n < 2 or not side.ally_side:
                handlers += self.find_side_event_handlers(side, f"onSide{eventid}", get_key)
            for active in side.active:
                if not active:
                    continue
                if eventid == "SwitchIn":
                    handlers += self.find_pokemon_event_handlers(active, f"onAny{eventid}")
                if targets is not None and active not in targets:
                    continue
                handlers += self.find_pokemon_event_handlers(active, f"on{eventid}", get_key)
        self.speed_sort(handlers)
        for handler in handlers:
            cb = handler.get("callback")
            if not callable(cb):
                continue
            effect = handler["effect"]
            holder = handler.get("effectHolder")
            handler_eventid = eventid
            self.single_event(handler_eventid, effect, handler.get("state"), holder,
                              None, None, None, cb)
            self.faint_messages()
            if self.ended:
                return

    fieldEvent = field_event

    def end_turn(self):
        self.turn += 1
        self.last_successful_move_this_turn = None
        for side in self.sides:
            for pokemon in side.active:
                if not pokemon:
                    continue
                pokemon.move_this_turn = ""
                pokemon.move_this_turn_result = None
        self._update_speed()
        self.add("turn", self.turn)
        if self.gen == 2:
            self.quick_claw_roll = self.random_chance(60, 256)
        elif self.gen == 3:
            self.quick_claw_roll = self.random_chance(1, 5)


def _cmp_key(comparator):
    import functools

    return functools.cmp_to_key(comparator)
