"""Battle orchestrator — event loop mirroring sim/battle.ts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from pokepy.core.constants import (
    F_HAZARDS_0,
    F_HAZARDS_1,
    M_ACTIVE0,
    M_ACTIVE1,
    OFF_FIELD,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    Phase,
    STATUS_NAMES,
)
from pokepy.engine.switch_requests import (
    SwitchRequest,
    pokepy_action_from_slot,
    resolve_switch_choices_sync,
    slot_from_pokepy_action,
)
from pokepy.sim import dispatch
from pokepy.sim.helpers import (
    action_speed,
    active_slot,
    base_ability_for_offset,
    clear_side_switch_field,
    consume_endturn_quick_claw_roll,
    consume_startup_prng,
    count_alive_side,
    move_priority,
    set_forced_switch_phase,
    sync_showdown_order,
    terminal_rewards,
)
from pokepy.sim.queue import (
    ORDER_BEFORE_TURN,
    ORDER_MOVE,
    ORDER_RESIDUAL,
    ORDER_SWITCH,
    Action,
    BattleQueue,
)
from pokepy.sim.views import FieldView, PokemonView

if TYPE_CHECKING:
    from pokepy.core.gen_profile import GenProfile
    from pokepy.core.state import MultiFormatState

# comparePriority on queue/residual entry tuples:
# (order, priority, speed, subOrder, effectOrder)
PriorityEntry = Tuple[int, int, int, int, int]


def compare_priority_entries(a: PriorityEntry, b: PriorityEntry) -> int:
    """battle.ts:404-410 on action/residual handler tuples."""
    if a[0] != b[0]:
        return a[0] - b[0]
    if a[1] != b[1]:
        return b[1] - a[1]
    if a[2] != b[2]:
        return b[2] - a[2]
    if a[3] != b[3]:
        return a[3] - b[3]
    if a[4] != b[4]:
        return a[4] - b[4]
    return 0


# battle.ts:950-1015 resolvePriority subOrder defaults by effect type. We only
# need the relative ordering of the effect categories pokepy models.
SUBORDER_ABILITY = 7
SUBORDER_ITEM = 8
SUBORDER_SPECIES = 6
SUBORDER_DEFAULT = 0


def shuffle_in_place(
    rng, items: list, start: int = 0, end: int | None = None
) -> int | None:
    """prng.ts:155-162 Fisher-Yates shuffle; returns first draw for 2-way tie groups."""
    if end is None:
        end = len(items)
    first_draw: int | None = None
    while start < end - 1:
        next_index = int(rng(start, end))
        if first_draw is None and end - start == 2:
            first_draw = next_index
        if start != next_index:
            items[start], items[next_index] = items[next_index], items[start]
        start += 1
    return first_draw


def speed_sort_in_place(
    rng,
    items: list,
    comparator: Callable[[Any, Any], int],
) -> int | None:
    """battle.ts:429-459 selection sort + shuffle on ties.

    Returns the first 2-way tie shuffle draw (0 or 1) when the full list is
    a single tied pair, else None.
    """
    if len(items) < 2:
        return None
    tie_draw: int | None = None
    sorted_idx = 0
    while sorted_idx + 1 < len(items):
        next_indexes = [sorted_idx]
        for i in range(sorted_idx + 1, len(items)):
            delta = comparator(items[next_indexes[0]], items[i])
            if delta < 0:
                continue
            if delta > 0:
                next_indexes = [i]
            if delta == 0:
                next_indexes.append(i)
        for i, index in enumerate(next_indexes):
            target = sorted_idx + i
            if index != target:
                items[target], items[index] = items[index], items[target]
        if len(next_indexes) > 1:
            start = sorted_idx
            end = sorted_idx + len(next_indexes)
            draw = shuffle_in_place(rng, items, start, end)
            if (
                tie_draw is None
                and len(items) == 2
                and len(next_indexes) == 2
                and start == 0
            ):
                tie_draw = draw
        sorted_idx += len(next_indexes)
    return tie_draw


def consume_two_way_speed_tie(rng, speed_a: int, speed_b: int) -> None:
    """One Showdown eachEvent('Update') / runSwitch speedSort frame on a speed tie."""
    if int(speed_a) != int(speed_b):
        return
    entries: List[PriorityEntry] = [
        (0, 0, int(speed_a), 0, 0),
        (0, 0, int(speed_b), 0, 0),
    ]
    speed_sort_in_place(rng, entries, compare_priority_entries)


class SpeedSortTracker:
    """PRNG speedSort helper — uses battle.ts algorithms."""

    compare_priority = staticmethod(compare_priority_entries)

    def __init__(self, prng) -> None:
        self.prng = prng

    def random(self, *args):
        return self.prng.random(*args)

    def speed_sort_consume(self, entries: List[PriorityEntry]) -> int:
        items = list(entries)
        draw = speed_sort_in_place(self.random, items, self.compare_priority)
        return int(draw) if draw is not None else 0

    def each_event_update(self, active_speeds: List[int]) -> None:
        """eachEvent('Update') / eachEvent('BeforeTurn') — speed-only compare."""
        if len(active_speeds) < 2:
            return
        consume_two_way_speed_tie(self.random, active_speeds[0], active_speeds[1])

    def queue_sort(self, actions: List[PriorityEntry]) -> int:
        return self.speed_sort_consume(actions)


@dataclass
class EventHandler:
    callback: Callable[..., Any]
    priority: int = 0
    order: Optional[int] = None
    effect_holder: Any = None
    speed: int = 0
    sub_order: int = 0
    effect_order: int = 0
    index: int = 0


class Battle:
    """Synchronous Showdown-faithful battle event runner."""

    def __init__(
        self,
        state: "MultiFormatState",
        profile: "GenProfile",
        prng,
        *,
        game_data=None,
        move_effects=None,
        type_chart=None,
    ) -> None:
        self.state = state
        self.profile = profile
        self.prng = prng
        self.game_data = game_data
        self.move_effects = move_effects
        self.type_chart = type_chart
        self.battle = state.battle_state
        self.field = FieldView(self.battle)
        self.effect = None
        self.format_id = int(profile.format_id)
        self.queue = BattleQueue()
        self.max_turns = int(getattr(state, "max_turns", 200))
        self.pending_switch_sides: List[int] = []
        self.mid_turn_switch_pending: Dict[int, int] = {}
        self.wants_tera = (False, False)
        self._move_second_side: Optional[int] = None

    def random(self, *args):
        return self.prng.random(*args)

    def random_chance(self, num: int, denom: int) -> bool:
        return self.prng.random_chance(num, denom)

    @staticmethod
    def compare_priority(a: EventHandler, b: EventHandler) -> int:
        """battle.ts:404-410."""
        ao = a.order if a.order is not None else 4294967296
        bo = b.order if b.order is not None else 4294967296
        if ao != bo:
            return ao - bo
        if a.priority != b.priority:
            return b.priority - a.priority
        if a.speed != b.speed:
            return b.speed - a.speed
        if a.sub_order != b.sub_order:
            return a.sub_order - b.sub_order
        if a.effect_order != b.effect_order:
            return a.effect_order - b.effect_order
        return a.index - b.index

    def shuffle(self, items: list, start: int = 0, end: int | None = None) -> None:
        """prng.ts:155-162 Fisher-Yates shuffle."""
        shuffle_in_place(self.random, items, start, end)

    def speed_sort(
        self,
        items: list,
        comparator: Callable[[Any, Any], int] | None = None,
    ) -> None:
        """battle.ts:429-459 selection sort + shuffle on ties."""
        if len(items) < 2:
            return
        cmp_fn = comparator or self.compare_priority
        speed_sort_in_place(self.random, items, cmp_fn)

    def _status_name(self, mon: PokemonView) -> str:
        st = mon.status
        if 0 <= st < len(STATUS_NAMES):
            return STATUS_NAMES[st]
        return "nostatus"

    def _collect(
        self,
        handlers: List[EventHandler],
        effect_id: str,
        event_id: str,
        holder: Any,
        *,
        sub_order: int = SUBORDER_DEFAULT,
    ) -> None:
        """resolvePriority (battle.ts:950-1015): look up a registered handler
        for ``effect_id``/``event_id`` and append it with its priority/order
        plus the holder's speed and effect-category subOrder."""
        entry = dispatch.get_handlers(self.format_id, effect_id).get(event_id)
        if entry is None:
            return
        speed = int(getattr(holder, "spe", 0) or 0)
        handlers.append(
            EventHandler(
                callback=entry.handler,
                priority=entry.priority,
                order=entry.order,
                effect_holder=holder,
                speed=speed,
                sub_order=sub_order,
            )
        )

    def find_pokemon_event_handlers(
        self,
        mon: PokemonView,
        event_id: str,
    ) -> List[EventHandler]:
        """battle.ts:1097-1155 — status, ability, item, weather, residual.

        Abilities and items are keyed by their integer id (data-driven), so any
        ability/item that registers a handler participates automatically with no
        hardcoded id->name table.
        """
        handlers: List[EventHandler] = []
        self._collect(handlers, self._status_name(mon), event_id, mon)

        if self.profile.has_abilities and int(mon.ability) > 0:
            self._collect(
                handlers,
                str(int(mon.ability)),
                event_id,
                mon,
                sub_order=SUBORDER_ABILITY,
            )

        if self.profile.has_items and int(mon.item) > 0:
            self._collect(
                handlers,
                str(int(mon.item)),
                event_id,
                mon,
                sub_order=SUBORDER_ITEM,
            )
            self._collect(
                handlers,
                "berries",
                event_id,
                mon,
                sub_order=SUBORDER_ITEM,
            )

        # Per-mon residual bundle (salt cure, partial trap, aqua ring/ingrain,
        # yawn, weather healing) — each self-gates on the relevant volatile bit.
        self._collect(handlers, "monresidual", event_id, mon)

        if int(self.field.weather) > 0:
            self._collect(handlers, "weather", event_id, mon)

        # Entry hazards fire on the switching mon's SwitchIn event.
        if event_id == dispatch.SwitchIn:
            self._collect(handlers, "hazards", event_id, mon)

        return handlers

    def find_field_event_handlers(
        self,
        event_id: str,
        target: PokemonView | None = None,
    ) -> List[EventHandler]:
        """battle.ts:1181-1213 — pseudoWeather, weather, terrain (field-global)."""
        handlers: List[EventHandler] = []
        holder = target if target is not None else self.field
        if int(self.field.weather) > 0:
            self._collect(handlers, "fieldweather", event_id, holder)
        if int(self.field.terrain) > 0:
            self._collect(handlers, "fieldterrain", event_id, holder)
        return handlers

    def find_side_event_handlers(self, event_id: str) -> List[EventHandler]:
        """battle.ts:1216-1230 — side conditions (screens, hazards, etc.)."""
        return []

    def find_battle_event_handlers(self, event_id: str) -> List[EventHandler]:
        """battle.ts:1158-1178 — format/custom battle handlers."""
        return []

    def find_event_handlers(
        self,
        target: PokemonView,
        event_id: str,
        source: PokemonView | None = None,
    ) -> List[EventHandler]:
        """battle.ts:1035-1094 — gather pokemon + field + battle handlers."""
        handlers = self.find_pokemon_event_handlers(target, event_id)
        handlers.extend(self.find_field_event_handlers(event_id, target))
        handlers.extend(self.find_battle_event_handlers(event_id))
        return handlers

    def run_event(
        self,
        event_id: str,
        target: PokemonView | None = None,
        source: PokemonView | None = None,
        effect=None,
        relay_var: Any = True,
        *,
        left_to_right: bool = False,
        **kwargs,
    ) -> Any:
        """battle.ts:655-712 / 758-795 relayVar semantics."""
        if target is None and source is None:
            return relay_var
        tgt = target or source
        assert tgt is not None
        handlers = self.find_event_handlers(tgt, event_id, source)
        for i, h in enumerate(handlers):
            h.index = i
        if not left_to_right:
            self.speed_sort(handlers)
        result = relay_var
        for h in handlers:
            cb = h.callback
            ret = cb(
                self,
                h.effect_holder,
                source,
                effect,
                result,
                **kwargs,
            )
            if ret is False or ret is None:
                return ret
            if ret is True:
                continue
            if isinstance(ret, (int, float)) and isinstance(result, (int, float)):
                result = ret
            elif ret is not True and ret is not None:
                result = ret
        return result

    def single_event(
        self,
        event_id: str,
        effect_id: str,
        target: PokemonView,
        *,
        source: PokemonView | None = None,
        relay_var: Any = True,
        **kwargs,
    ) -> Any:
        """battle.ts:571-651."""
        entry = dispatch.get_handlers(self.format_id, effect_id).get(event_id)
        if entry is None:
            return relay_var
        return entry.handler(self, target, source, None, relay_var, **kwargs)

    def each_event(self, event_id: str, *, relay_var: Any = True) -> None:
        """battle.ts:465-475."""
        actives: List[PokemonView] = []
        for side in (0, 1):
            mon = self.field.active(side)
            if mon.hp > 0 and not mon.fainted:
                actives.append(mon)
        wrappers = [
            EventHandler(
                callback=lambda *a, **k: None,
                speed=m.spe,
                effect_holder=m,
            )
            for m in actives
        ]
        self.speed_sort(wrappers, comparator=lambda a, b: b.speed - a.speed)
        for w in wrappers:
            self.run_event(event_id, target=w.effect_holder, relay_var=relay_var)

    def field_event(self, event_id: str) -> None:
        """battle.ts:484-567 — Residual / SwitchIn field pass.

        Honors ``event_id`` (no longer hardcoded to Residual). Speed-sorts the
        active mons, dispatches the event per-mon, and runs ``faint_messages``
        after each handler block. Field-level duration decrements run after the
        per-mon pass for Residual.
        """
        actives: List[PokemonView] = []
        for side in (0, 1):
            mon = self.field.active(side)
            if mon.hp > 0 and not mon.fainted:
                actives.append(mon)
        wrappers = [
            EventHandler(callback=lambda *a, **k: None, speed=m.spe, effect_holder=m)
            for m in actives
        ]
        self.speed_sort(wrappers, comparator=lambda a, b: b.speed - a.speed)
        for w in wrappers:
            mon = w.effect_holder
            if mon.hp <= 0:
                continue
            self.run_event(event_id, target=mon, relay_var=True)
            self.faint_messages()
        if event_id == dispatch.Residual:
            self._residual_field_decrements()

    def _residual_field_decrements(self) -> None:
        """Decrement field-wide duration counters (Trick Room, screens, weather,
        terrain) at end of turn. battle.ts residual handles these as field/side
        conditions with onResidual + onEnd; pokepy stores them as packed counters."""
        from pokepy.effects.end_of_turn import (
            decrement_screens,
            decrement_terrain,
            decrement_trick_room,
            decrement_weather,
        )

        decrement_trick_room(self.battle)
        decrement_screens(self.battle)
        decrement_weather(self.battle)
        decrement_terrain(self.battle)

    def check_fainted(self) -> None:
        """battle.ts:2537-2545 — flag any active mon at 0 HP as fainted and
        request a replacement switch for its side."""
        for side in (0, 1):
            mon = self.field.active(side)
            if mon.hp <= 0:
                if not mon.fainted:
                    mon.fainted = True
                self._request_switch(side)

    def faint_messages(self) -> None:
        """battle.ts:2548-2624 — drain pending faints (deferred faint queue).

        pokepy applies HP changes directly to the packed buffer, so rather than
        a separate queue we scan the active slots and flag/enqueue any that have
        reached 0 HP."""
        self.check_fainted()

    def get_all_active(self) -> List[PokemonView]:
        out: List[PokemonView] = []
        for side in (0, 1):
            mon = self.field.active(side)
            if mon.hp > 0 and not mon.fainted:
                out.append(mon)
        return out

    def maybe_set_status_speedsort(self) -> None:
        """setStatus speedSort when tied actives (pokemon.ts:1724)."""
        actives = self.get_all_active()
        if len(actives) < 2:
            return
        speeds = [m.spe for m in actives]
        if len(set(speeds)) == 1:
            self.speed_sort(
                actives,
                comparator=lambda a, b: b.spe - a.spe,
            )

    # ------------------------------------------------------------------
    # Turn loop — battle.ts turnLoop / commitChoices / runAction
    # ------------------------------------------------------------------

    def _resolve_action(self, side: int, action: int) -> Action:
        """Build a queue Action from a pokepy action index (0-3 move, 4-9 switch)."""
        act = int(action)
        if act >= 4:
            slot = max(0, min(5, act - 4))
            return Action(
                choice="switch",
                order=ORDER_SWITCH,
                side=int(side),
                switch_slot=slot,
                speed=action_speed(self.battle, side, self.profile),
            )
        prio = move_priority(
            self.battle,
            self.state,
            side,
            act,
            self.game_data,
            self.prng,
        )
        fractional = 0.0
        priority = int(prio)
        if isinstance(prio, float):
            priority = int(prio)
            fractional = float(prio) - float(priority)
        return Action(
            choice="move",
            order=ORDER_MOVE,
            side=int(side),
            move_slot=act,
            priority=priority,
            fractional_priority=fractional,
            speed=action_speed(self.battle, side, self.profile),
        )

    def commit_choices(self, action0: int, action1: int) -> None:
        """battle.ts:3019 commitChoices — build queue and speedSort."""
        self.queue.clear()
        self._move_second_side = None
        if self.profile.has_tera:
            from pokepy.effects.tera import activate_terastallization, side_can_tera

            if self.wants_tera[0] and side_can_tera(self.battle, 0):
                activate_terastallization(
                    self.battle, 0, team_tera=self.state.team_tera
                )
            if self.wants_tera[1] and side_can_tera(self.battle, 1):
                activate_terastallization(self.battle, 1, team_tera=self.state.opp_tera)
        self.queue.add_choice(self._resolve_action(0, action0))
        self.queue.add_choice(self._resolve_action(1, action1))
        self.queue.sort(self.speed_sort)

    def _insert_before_turn(self) -> None:
        before = Action(choice="beforeTurn", order=ORDER_BEFORE_TURN)
        self.queue.insert_choice(before, 0)

    def _append_residual(self) -> None:
        self.queue.add_choice(Action(choice="residual", order=ORDER_RESIDUAL))

    def run_action(self, action: Action) -> None:
        """battle.ts:2686-2864 runAction dispatch + shared postamble."""
        if action.choice == "beforeTurn":
            self.each_event(dispatch.BeforeTurn)
            if self.profile.gen < 5:
                self.each_event(dispatch.Update)
            return
        if action.choice == "switch":
            self.run_switch_action(action.side, action.switch_slot)
            self._after_action()
            return
        if action.choice == "move":
            from pokepy.sim import moves

            is_second = self._move_second_side is not None
            self._move_second_side = action.side
            moves.run_move(
                self,
                action.side,
                action.move_slot,
                is_second=is_second,
            )
            self._after_action()
            return
        if action.choice == "residual":
            self.run_residual()
            self._after_action()

    def _after_action(self) -> None:
        """battle.ts:2856-2935 postamble — process faints and queue switch
        requests for any side whose active fainted this action."""
        self.faint_messages()

    def run_switch_action(self, side: int, slot: int) -> None:
        self.switch_in(side, slot, mid_turn=True)

    def switch_in(
        self,
        side: int,
        slot: int,
        *,
        mid_turn: bool = False,
        forced: bool = False,
    ) -> None:
        """Mirror runSwitch / switchIn (battle-actions.ts)."""
        import pokepy.effects as fx

        side = int(side)
        slot = max(0, min(5, int(slot)))
        side_base = OFF_SIDE0 if side == 0 else OFF_SIDE1
        active_meta = M_ACTIVE0 if side == 0 else M_ACTIVE1
        opp_meta = M_ACTIVE1 if side == 0 else M_ACTIVE0
        side_order = self.state.side_order0 if side == 0 else self.state.side_order1
        hazards_off = F_HAZARDS_0 if side == 0 else F_HAZARDS_1

        target_off = side_base + slot * POKEMON_SIZE
        if int(self.battle[target_off + 1]) <= 0 or (
            int(self.battle[target_off + 15]) & 1
        ):
            for i in range(6):
                so = side_base + i * POKEMON_SIZE
                if (
                    int(self.battle[so + 1]) > 0
                    and (int(self.battle[so + 15]) & 1) == 0
                ):
                    slot = i
                    target_off = so
                    break

        old_active = active_slot(self.battle, side)
        old_off = side_base + old_active * POKEMON_SIZE
        if int(self.battle[old_off + 1]) > 0:
            fx.apply_regenerator_on_switch_out(self.battle, old_off, True)
            fx.apply_natural_cure_on_switch_out(self.battle, old_off, True)

        self.battle[OFF_META + active_meta] = slot
        sync_showdown_order(side_order, slot)
        clear_side_switch_field(self.battle, side)

        fx.reset_incoming_switch_state(
            self.battle,
            target_off,
            self.game_data,
            base_ability=base_ability_for_offset(self.state, target_off),
            state=self.state,
        )

        # Entry hazards are applied via the SwitchIn event (the "hazards"
        # handler in field.py), not imperatively here.
        opp_side = 1 - side
        opp_off = (OFF_SIDE1 if side == 0 else OFF_SIDE0) + active_slot(
            self.battle, opp_side
        ) * POKEMON_SIZE

        if self.profile.has_abilities and int(self.battle[target_off + 1]) > 0:
            fx.apply_switch_in_ability(
                self.battle,
                target_off,
                opp_off,
                side == 0,
                gen5_prng=self.prng,
                has_terrain=self.profile.has_terrain,
                ability_weather_limited=self.profile.ability_weather_limited,
            )

        self.run_event(dispatch.SwitchIn, target=PokemonView(self.battle, target_off))
        self.each_event(dispatch.Update)

        if int(self.battle[target_off + 1]) <= 0:
            self._request_switch(side)

    def _request_switch(self, side: int) -> None:
        if side not in self.pending_switch_sides:
            self.pending_switch_sides.append(int(side))

    def run_residual(self) -> None:
        """battle.ts:2832-2838 residual action — single owner via fieldEvent.

        All end-of-turn effects are registered onResidual handlers fired through
        ``field_event(Residual)``; the legacy ``apply_end_of_turn_effects`` path
        is gone (it double-applied status/weather/speed-boost)."""
        import pokepy.effects as fx
        from pokepy.core.constants import OFF_SIDE0, OFF_SIDE1, POKEMON_SIZE
        from pokepy.sim.helpers import active_slot

        self.field_event(dispatch.Residual)
        a0 = active_slot(self.battle, 0)
        a1 = active_slot(self.battle, 1)
        p0 = OFF_SIDE0 + a0 * POKEMON_SIZE
        p1 = OFF_SIDE1 + a1 * POKEMON_SIZE
        fx.apply_leech_seed_damage(self.battle, p0, p1, self.game_data)
        fx.process_perish_song(self.battle, p0, p1)
        fx.decrement_taunt_encore(self.battle, self.prng)
        fx.decrement_confusion(self.battle)
        fx.clear_protect_at_turn_end(self.battle)
        fx.clear_volatile_turn_effects(self.battle)

    def end_turn(self) -> Tuple[np.float32, np.float32, bool]:
        """battle.ts endTurn — increment turn, check win, set forced-switch phase."""
        consume_endturn_quick_claw_roll(self.profile, self.prng)
        self.state.turn = np.int16(int(self.state.turn) + 1)
        reward0, reward1, done, winner = terminal_rewards(self.state, self.max_turns)
        self.state.done = np.bool_(done)
        self.state.winner = np.int8(winner)
        if not done and self.pending_switch_sides:
            sides = tuple(sorted(set(self.pending_switch_sides)))
            set_forced_switch_phase(self.state, sides)
        elif not done:
            self.state.phase = np.int8(Phase.BATTLE)
            self.state.forced_switch_side = np.int8(-1)
        return reward0, reward1, done

    def turn_loop(self) -> Tuple[np.float32, np.float32, bool]:
        """Execute one full turn synchronously."""
        gen = self.turn_loop_iter()
        try:
            req = next(gen)
            while True:
                choices = resolve_switch_choices_sync(
                    self.state,
                    self.battle,
                    req,
                    side_order0=self.state.side_order0,
                    side_order1=self.state.side_order1,
                )
                for side, slot in choices.items():
                    self.switch_in(side, slot, mid_turn=True)
                req = gen.send(choices)
        except StopIteration as stop:
            return stop.value

    def turn_loop_iter(
        self,
        *,
        resolve_mid_turn_switch0=None,
    ) -> Generator[SwitchRequest, Dict[int, int], Tuple[np.float32, np.float32, bool]]:
        """Generator yielding SwitchRequest for mid-turn / double-KO replacements."""
        self._insert_before_turn()
        self._append_residual()
        while self.queue.list:
            action = self.queue.shift()
            assert action is not None
            self.run_action(action)
            if self.pending_switch_sides:
                sides = tuple(sorted(set(self.pending_switch_sides)))
                self.pending_switch_sides.clear()
                choices = yield SwitchRequest(sides)
                for side in sides:
                    slot = int(choices.get(side, 0))
                    if side == 0 and resolve_mid_turn_switch0 is not None:
                        slot = slot_from_pokepy_action(
                            int(resolve_mid_turn_switch0(self.state))
                        )
                    self.switch_in(side, slot, mid_turn=True)
        result = self.end_turn()
        return result

    def execute_forced_switch(
        self, action: int, side: int
    ) -> Tuple[np.float32, np.float32, bool]:
        """Forced switch after KO/pivot — no turn advance."""
        slot = slot_from_pokepy_action(int(action))
        self.switch_in(int(side), slot, forced=True)
        reward0, reward1, done, winner = terminal_rewards(self.state, self.max_turns)
        self.state.done = np.bool_(done)
        self.state.winner = np.int8(winner)
        if not done:
            self.state.phase = np.int8(Phase.BATTLE)
            self.state.forced_switch_side = np.int8(-1)
        return reward0, reward1, done
