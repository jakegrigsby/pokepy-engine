"""Unit tests for Battle event engine core."""

from __future__ import annotations

from pokepy.core.gen_profile import GEN1_PROFILE
from pokepy.sim.battle import (
    Battle,
    SpeedSortTracker,
    compare_priority_entries,
    consume_two_way_speed_tie,
    speed_sort_in_place,
)
from pokepy.sim.queue import Action
from pokepy.utils.gen5_prng import Gen5PRNG


class _State:
    battle_state = __import__("numpy").zeros(256, dtype=__import__("numpy").int16)


def test_speed_sort_two_way_tie_one_frame():
    prng = Gen5PRNG((999, 999, 0, 0))
    tracker = SpeedSortTracker(prng)
    a = (200, 0, 100, 0, 0)
    b = (200, 0, 100, 0, 0)
    result = tracker.queue_sort([a, b])
    assert result in (0, 1)


def test_speed_sort_matches_showdown_shuffle_bounds():
    """3-way tie at index 0 consumes random(0,3) then random(1,3)."""
    prng = Gen5PRNG((424242, 424242, 0, 0))
    draws: list[tuple[int, int]] = []

    def recording_rng(start, end):
        val = prng.random(start, end)
        draws.append((int(start), int(end)))
        return val

    entries = [(0, 0, 80, 0, 0), (0, 0, 80, 0, 0), (0, 0, 80, 0, 0)]
    speed_sort_in_place(recording_rng, entries, compare_priority_entries)
    assert draws == [(0, 3), (1, 3)]


def test_consume_two_way_speed_tie_skips_unequal():
    prng = Gen5PRNG((999, 999, 0, 0))
    calls = 0

    def counting_rng(*args):
        nonlocal calls
        calls += 1
        return prng.random(*args)

    consume_two_way_speed_tie(counting_rng, 100, 80)
    assert calls == 0


def test_consume_two_way_speed_tie_one_frame():
    prng = Gen5PRNG((999, 999, 0, 0))
    calls = 0

    def counting_rng(*args):
        nonlocal calls
        calls += 1
        return prng.random(*args)

    consume_two_way_speed_tie(counting_rng, 100, 100)
    assert calls == 1


def test_run_event_relay_short_circuit():
    from pokepy.core.constants import OFF_SIDE0, STATUS_SLEEP
    from pokepy.sim import dispatch
    from pokepy.sim.views import PokemonView

    prng = Gen5PRNG((1, 2, 3, 4))
    battle = Battle(_State(), GEN1_PROFILE, prng)

    def blocker(b, mon, src, eff, relay, **kw):
        return False

    dispatch.register(
        GEN1_PROFILE.format_id, "slp", dispatch.BeforeMove, blocker, priority=10
    )
    mon = PokemonView(_State().battle_state, OFF_SIDE0)
    mon.status = STATUS_SLEEP
    result = battle.run_event(dispatch.BeforeMove, target=mon, relay_var=True)
    assert result is False


def test_battle_speed_sort_on_actions():
    prng = Gen5PRNG((1, 1, 1, 1))
    battle = Battle(_State(), GEN1_PROFILE, prng)
    fast = Action(choice="move", order=200, speed=120, side=1)
    slow = Action(choice="move", order=200, speed=80, side=0)
    items = [slow, fast]
    battle.speed_sort(
        items,
        comparator=lambda a, b: compare_priority_entries(
            (a.order, a.priority, a.speed, a.sub_order, a.effect_order),
            (b.order, b.priority, b.speed, b.sub_order, b.effect_order),
        ),
    )
    assert items[0].speed == 120
