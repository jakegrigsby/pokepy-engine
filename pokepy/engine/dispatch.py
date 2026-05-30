"""Event dispatch on the scalar bitpack battle state.

Structural port of sim/battle.ts runEvent/singleEvent/speedSort adapted for
``MultiFormatState`` + flat ``battle_state`` buffer. Effect handlers are
looked up via ``EffectRegistry`` and invoked with the bitpack signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from pokepy.core.gen_profile import GenProfile, profile_for_gen
from pokepy.core.state import MultiFormatState
from pokepy.engine.registry import DEFAULT_REGISTRY, EffectRegistry
from pokepy.utils.gen5_prng import Gen5PRNG

_BIG_ORDER = 4294967296


@dataclass
class EventContext:
    id: str = ""
    target: Any = None
    source: Any = None
    effect: Any = None
    modifier: Any = 1


@dataclass
class BitpackBattleContext:
    """Minimal Battle stand-in for scalar engine dispatch."""

    state: MultiFormatState
    battle: np.ndarray
    gen5_prng: Gen5PRNG
    game_data: Any
    move_effects: Any
    type_chart: np.ndarray
    profile: GenProfile
    registry: EffectRegistry = field(default_factory=lambda: DEFAULT_REGISTRY)
    event: EventContext = field(default_factory=EventContext)
    event_depth: int = 0
    speed_order: List[int] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    @property
    def gen(self) -> int:
        return int(self.profile.gen)

    # ------------------------------------------------------------------ #
    # PRNG + numeric API (sim/battle.ts)
    # ------------------------------------------------------------------ #
    def random(self, n: int) -> int:
        return int(self.gen5_prng.random(int(n)))

    def random_chance(self, num: int, denom: int) -> bool:
        return self.random(int(denom)) < int(num)

    def randomizer(self, base: int) -> int:
        """Gen3+ damage randomizer: 85-100%."""
        if self.gen <= 2:
            return int(base)
        roll = self.random(16)
        return max(1, (int(base) * (100 - roll)) // 100)

    def modify(self, value: int, modifier: int) -> int:
        if modifier == 4096:
            return int(value)
        return ((int(value) * int(modifier)) + 2047) >> 12

    def chain_modify(self, value: int, numerator: int, denominator: int) -> int:
        if denominator == 0:
            return int(value)
        modifier = (int(numerator) * 4096) // int(denominator)
        return self.modify(int(value), modifier)

    def boost(self, pokemon_offset: int, stat_index: int, delta: int) -> None:
        from pokepy.core.bitpack import apply_boost_to_packed

        poff = int(pokemon_offset)
        if stat_index == 5:
            self.battle[poff + 14] = np.int16(
                apply_boost_to_packed(int(self.battle[poff + 14]), 0, int(delta))
            )
        else:
            shift = stat_index * 4
            self.battle[poff + 13] = np.int16(
                apply_boost_to_packed(int(self.battle[poff + 13]), shift, int(delta))
            )

    def damage(self, pokemon_offset: int, amount: int) -> int:
        poff = int(pokemon_offset)
        hp = max(0, int(self.battle[poff + 1]) - int(amount))
        self.battle[poff + 1] = np.int16(hp)
        return hp

    def heal(self, pokemon_offset: int, amount: int) -> int:
        poff = int(pokemon_offset)
        max_hp = int(self.battle[poff + 2])
        hp = min(max_hp, int(self.battle[poff + 1]) + int(amount))
        self.battle[poff + 1] = np.int16(hp)
        return hp

    # ------------------------------------------------------------------ #
    # Speed sort (sim/battle.ts:429)
    # ------------------------------------------------------------------ #
    @staticmethod
    def compare_priority(a: Dict[str, Any], b: Dict[str, Any]) -> int:
        return (
            -((b.get("order") or _BIG_ORDER) - (a.get("order") or _BIG_ORDER))
            or ((b.get("priority") or 0) - (a.get("priority") or 0))
            or -((b.get("index") or 0) - (a.get("index") or 0))
            or 0
        )

    def _prng_shuffle(self, items: List[Any], start: int, end: int) -> None:
        while start < end - 1:
            next_index = self.random(end - start) + start
            if start != next_index:
                items[start], items[next_index] = items[next_index], items[start]
            start += 1

    def speed_sort(self, lst: List[Any], comparator: Optional[Callable] = None) -> None:
        if comparator is None:
            comparator = self.compare_priority
        if len(lst) < 2:
            return
        sorted_n = 0
        while sorted_n + 1 < len(lst):
            next_indexes = [sorted_n]
            for i in range(sorted_n + 1, len(lst)):
                delta = comparator(lst[next_indexes[0]], lst[i])
                if delta < 0:
                    continue
                if delta > 0:
                    next_indexes = [i]
                if delta == 0:
                    next_indexes.append(i)
            for i, index in enumerate(next_indexes):
                if index != sorted_n + i:
                    lst[sorted_n + i], lst[index] = lst[index], lst[sorted_n + i]
            if len(next_indexes) > 1:
                self._prng_shuffle(lst, sorted_n, sorted_n + len(next_indexes))
            sorted_n += len(next_indexes)

    # ------------------------------------------------------------------ #
    # Event dispatch
    # ------------------------------------------------------------------ #
    def find_event_handlers(
        self,
        target_offset: int,
        event_name: str,
        *,
        table: str = "ability",
        effect_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        handlers: List[Dict[str, Any]] = []
        poff = int(target_offset)
        if effect_id is None:
            if table == "ability":
                effect_id = int(self.battle[poff + 5])
            elif table == "item":
                effect_id = int(self.battle[poff + 6])
            else:
                effect_id = 0
        fn = self.registry.lookup(table, int(effect_id), event_name)
        if fn is not None:
            meta = self.registry.get_meta(table, int(effect_id), event_name)
            handlers.append(
                {
                    "callback": fn,
                    "effectHolder": poff,
                    "priority": int(meta.get("priority", 0)),
                    "order": meta.get("order", False),
                    "index": poff,
                }
            )
        # Attach generic handlers registered for this table+event.
        for entry in self.registry.generic_for(table, event_name):
            if entry["fn"] is fn:
                continue
            handlers.append(
                {
                    "callback": entry["fn"],
                    "effectHolder": poff,
                    "priority": entry.get("priority", 0),
                    "order": entry.get("order", False),
                    "index": poff,
                }
            )
        self.speed_sort(handlers)
        return handlers

    def single_event(
        self,
        eventid: str,
        effect: Any,
        state: Any,
        target_offset: int,
        source_offset: Optional[int] = None,
        *,
        relay_var: Any = None,
        custom_callback: Optional[Callable] = None,
    ) -> Any:
        if custom_callback is not None:
            return custom_callback(self, target_offset, source_offset, relay_var)
        handlers = self.find_event_handlers(int(target_offset), eventid)
        if not handlers:
            return relay_var
        self.event_depth += 1
        try:
            val = relay_var
            for h in handlers:
                cb = h.get("callback")
                if cb is None:
                    continue
                try:
                    val = cb(
                        self.battle,
                        int(h["effectHolder"]),
                        self.gen5_prng,
                        game_data=self.game_data,
                        move_effects=self.move_effects,
                        relay_var=val,
                        source_offset=source_offset,
                        ctx=self,
                    )
                except TypeError:
                    try:
                        val = cb(self.battle, int(h["effectHolder"]), self.gen5_prng)
                    except TypeError:
                        val = cb(self.battle, int(h["effectHolder"]))
                if val is False:
                    return False
            return val
        finally:
            self.event_depth -= 1

    def run_event(
        self,
        eventid: str,
        target_offset: Optional[int] = None,
        source_offset: Optional[int] = None,
        *,
        relay_var: Any = None,
        fast_exit: Optional[Callable[[Any], bool]] = None,
    ) -> Any:
        if target_offset is None:
            return relay_var
        val = self.single_event(
            eventid,
            None,
            None,
            int(target_offset),
            source_offset,
            relay_var=relay_var,
        )
        if fast_exit is not None and fast_exit(val):
            return val
        return val

    def each_event(self, eventid: str, callback: Callable[[int], None]) -> None:
        from pokepy.core.constants import (
            OFF_SIDE0,
            OFF_SIDE1,
            M_ACTIVE0,
            M_ACTIVE1,
            OFF_META,
            POKEMON_SIZE,
        )

        actives = [
            OFF_SIDE0 + int(self.battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE,
            OFF_SIDE1 + int(self.battle[OFF_META + M_ACTIVE1]) * POKEMON_SIZE,
        ]
        for poff in actives:
            if int(self.battle[poff + 1]) > 0:
                callback(poff)

    def field_event(self, eventid: str, *, relay_var: Any = None) -> Any:
        return relay_var


def make_context(
    state: MultiFormatState,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng: Gen5PRNG,
    *,
    profile: Optional[GenProfile] = None,
    registry: Optional[EffectRegistry] = None,
) -> BitpackBattleContext:
    prof = profile or profile_for_gen(int(getattr(state, "format_gen", 9) or 9))
    return BitpackBattleContext(
        state=state,
        battle=state.battle_state,
        gen5_prng=gen5_prng,
        game_data=game_data,
        move_effects=move_effects,
        type_chart=type_chart,
        profile=prof,
        registry=registry or DEFAULT_REGISTRY,
    )


__all__ = [
    "BitpackBattleContext",
    "EventContext",
    "make_context",
]
