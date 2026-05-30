"""Unit tests for sim scaffold (views, queue, dispatch)."""

from __future__ import annotations

import numpy as np

from pokepy.core.constants import (
    OFF_FIELD,
    OFF_SIDE0,
    POKEMON_SIZE,
    STATE_SIZE,
    STATUS_SLEEP,
)
from pokepy.core import bitpack
from pokepy.sim.queue import Action, BattleQueue, ORDER_MOVE
from pokepy.sim.views import FieldView, PokemonView


def test_pokemon_view_roundtrip():
    battle = np.zeros(STATE_SIZE, dtype=np.int16)
    base = OFF_SIDE0
    battle[base + 1] = 120
    battle[base + 2] = 200
    battle[base + 12] = bitpack.set_status(STATUS_SLEEP, 3)
    battle[base + 13] = 0x6666

    mon = PokemonView(battle, base)
    assert mon.hp == 120
    assert mon.max_hp == 200
    assert mon.status == STATUS_SLEEP
    assert mon.status_turns == 3

    mon.hp = 100
    mon.status_turns = 2
    assert battle[base + 1] == 100
    assert bitpack.get_status_turns(int(battle[base + 12])) == 2


def test_field_turn():
    battle = np.zeros(STATE_SIZE, dtype=np.int16)
    fld = FieldView(battle)
    fld.turn = 5
    assert int(battle[OFF_FIELD + 3]) == 5


def test_queue_sort_faster_first():
    q = BattleQueue()
    q.add_choice(Action(choice="move", order=ORDER_MOVE, speed=80, side=0))
    q.add_choice(Action(choice="move", order=ORDER_MOVE, speed=120, side=1))
    q.sort(lambda items: None)
    assert q.list[0].speed == 120
    assert q.list[1].speed == 80
