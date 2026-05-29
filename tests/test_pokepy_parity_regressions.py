from __future__ import annotations

import sys
from pathlib import Path

from pokepy.data.loader import load_game_data, load_id_mappings, load_move_effect_data
from pokepy.data.team_pool import get_team as pool_get_team
from pokepy.data.type_charts import MODERN_TYPE_CHART
from pokepy.engine.action_mask import get_battle_action_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from parity_heuristic_e2e import (
    run_pokepy_battle,
    run_showdown,
    simple_heuristic_action,
    simple_heuristic_forced_switch,
    team_to_showdown_packed,
)


def test_luster_purge_has_showdown_secondary():
    mappings = load_id_mappings()
    me = load_move_effect_data()

    move_id = mappings.move_to_idx["lusterpurge"]
    assert int(me.stat_target[move_id]) == 1
    assert int(me.stat_chance[move_id]) == 50
    assert [int(x) for x in me.stat_changes[move_id]] == [0, 0, 0, -1, 0, 0, 0]


def test_freezing_glare_has_showdown_secondary():
    from pokepy.core.constants import STATUS_FREEZE

    mappings = load_id_mappings()
    me = load_move_effect_data()

    move_id = mappings.move_to_idx["freezingglare"]
    assert int(me.effect_type[move_id]) != 0
    assert int(me.status[move_id]) == STATUS_FREEZE
    assert int(me.status_chance[move_id]) == 10


def test_battle5_switch_turn_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(39303),
        pool_get_team(25661),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        275121931,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[15]["p1_hp"] == 277
    assert normal_rows[16]["p0_hp"] == 277
    assert normal_rows[17]["p0_hp"] == 184
    assert normal_rows[17]["p1_hp"] == 394


def test_battle6_defog_hidden_prng_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(41987),
        pool_get_team(22519),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1074497556,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[12]["p0_hp"] == 330
    assert normal_rows[12]["p0_status"] == 4
    assert normal_rows[12]["p1_hp"] == 350


def test_battle77_freezing_glare_secondary_roll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42830),
        pool_get_team(976),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1628908287,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 17
    assert normal_rows[1]["p0_hp"] == 257
    assert normal_rows[1]["p1_hp"] == 239
    assert normal_rows[2]["p1_status"] == 1
    assert normal_rows[17]["p0_hp"] == 15
    assert normal_rows[17]["p1_hp"] == 0
    assert normal_rows[13]["p0_status"] == 0


def test_battle81_neutralizing_gas_suppresses_contrary_self_drop_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(32492),
        pool_get_team(31437),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        181342798,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[22]["p0_hp"] == 281
    assert normal_rows[22]["p1_hp"] == 255
    assert normal_rows[23]["p0_hp"] == 282
    assert normal_rows[23]["p1_hp"] == 102
    assert normal_rows[24]["p0_hp"] == 260
    assert normal_rows[24]["p1_hp"] == 323


def test_battle83_multihit_ko_stops_extra_damage_rolls_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(2080),
        pool_get_team(8972),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1060837206,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 17
    assert normal_rows[2]["p0_hp"] == 286
    assert normal_rows[2]["p1_hp"] == 325
    assert normal_rows[3]["p0_hp"] == 286
    assert normal_rows[3]["p1_hp"] == 197
    assert normal_rows[4]["p0_hp"] == 165
    assert normal_rows[4]["p1_hp"] == 197
    assert normal_rows[16]["p0_hp"] == 170
    assert normal_rows[16]["p1_hp"] == 285
    assert normal_rows[17]["p0_hp"] == 95
    assert normal_rows[17]["p1_hp"] == 0


def test_battle90_switch_in_resets_non_tera_species_types_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(14737),
        pool_get_team(1140),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        325340136,
        35,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 30
    assert normal_rows[12]["p0_hp"] == 357
    assert normal_rows[13]["p1_hp"] == 265
    assert normal_rows[14]["p0_hp"] == 210
    assert normal_rows[15]["p0_hp"] == 63
    assert normal_rows[16]["p0_hp"] == 292
    assert normal_rows[16]["p1_hp"] == 318


def test_battle92_harvest_replays_berry_update_hooks_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(23355),
        pool_get_team(39136),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        139493164,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 27
    assert normal_rows[18]["p0_hp"] == 315
    assert normal_rows[19]["p0_hp"] == 31
    assert normal_rows[20]["p0_hp"] == 381
    assert normal_rows[20]["p1_hp"] == 123
    assert normal_rows[21]["p1_hp"] == 512
    assert normal_rows[22]["p0_hp"] == 301
    assert normal_rows[22]["p1_hp"] == 200


def test_battle98_hazard_ko_after_pivot_cancels_slower_move_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(27250),
        pool_get_team(26033),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1824884135,
        55,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 47
    assert normal_rows[10]["p0_hp"] == 355
    assert normal_rows[10]["p1_hp"] == 299
    assert normal_rows[11]["p0_hp"] == 275
    assert normal_rows[11]["p1_hp"] == 99
    assert normal_rows[11]["p1_status"] == 1
    assert normal_rows[12]["p0_hp"] == 307
    assert normal_rows[12]["p1_hp"] == 299
    assert normal_rows[44]["p1_hp"] == 299
    assert normal_rows[45]["p0_hp"] == 43
    assert normal_rows[45]["p1_hp"] == 117
    assert normal_rows[45]["p1_status"] == 1
    assert normal_rows[46]["p0_hp"] == 147
    assert normal_rows[46]["p1_hp"] == 70
    assert normal_rows[46]["p1_status"] == 1


def test_battle88_pivot_user_life_orb_hp_persists_through_bench_reentry_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49484),
        pool_get_team(29053),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        960007121,
        29,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Dragapult uses U-turn with Life Orb earlier, then re-enters on turn 22.
    # The outgoing bench snapshot must keep that Life Orb recoil so Stealth Rock
    # brings it back at 185/317 rather than the stale 216/317 row.
    assert len(normal_rows) == 29
    assert normal_rows[21]["p0_hp"] == 5
    assert normal_rows[22]["p0_action"] == "move 1+forced:switch 5"
    assert normal_rows[22]["p0_hp"] == 185
    assert normal_rows[22]["p0_max_hp"] == 317
    assert normal_rows[23]["p0_hp"] == 196


def test_battle137_poison_puppeteer_and_self_ko_notarget_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(11482),
        pool_get_team(13964),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        283086057,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 50
    assert normal_rows[2]["p0_hp"] == 238
    assert normal_rows[2]["p0_status"] == 6
    assert normal_rows[7]["p0_hp"] == 56
    assert normal_rows[7]["p0_status"] == 6
    assert normal_rows[8]["p0_active"] == 3
    assert normal_rows[8]["p0_hp"] == 404
    assert normal_rows[8]["p1_hp"] == 268
    assert normal_rows[9]["p0_hp"] == 320
    assert normal_rows[9]["p1_hp"] == 94
    assert normal_rows[9]["p1_status"] == 1
    assert normal_rows[10]["p0_hp"] == 266
    assert normal_rows[10]["p1_active"] == 0
    assert normal_rows[50]["p0_hp"] == 0
    assert normal_rows[50]["p1_hp"] == 185


def test_battle152_alluring_voice_callback_secondary_roll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(7410),
        pool_get_team(31698),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1805590445,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 44
    assert normal_rows[17]["p0_hp"] == 181
    assert normal_rows[17]["p1_hp"] == 299
    assert normal_rows[44]["p0_hp"] == 290
    assert normal_rows[44]["p1_hp"] == 0


def test_battle154_avalanche_only_doubles_when_hit_by_target_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(37441),
        pool_get_team(20955),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        13982335,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 25
    assert normal_rows[11]["p0_hp"] == 240
    assert normal_rows[12]["p1_hp"] == 421
    assert normal_rows[25]["p0_hp"] == 0
    assert normal_rows[25]["p1_hp"] == 78


def test_battle205_team_preview_speed_tie_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(31978),
        pool_get_team(12489),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        572225617,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 22
    assert normal_rows[15]["p0_hp"] == 354
    assert normal_rows[15]["p1_hp"] == 122
    assert normal_rows[17]["p0_hp"] == 368
    assert normal_rows[17]["p1_hp"] == 254
    assert normal_rows[18]["p0_hp"] == 112
    assert normal_rows[18]["p1_hp"] == 156
    assert normal_rows[20]["p0_hp"] == 156
    assert normal_rows[20]["p1_hp"] == 122


def test_battle101_ceaseless_edge_empty_secondary_roll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(44808),
        pool_get_team(27141),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        301182625,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 19
    assert normal_rows[4]["p0_hp"] == 1
    assert normal_rows[4]["p1_hp"] == 55
    assert normal_rows[8]["p0_hp"] == 9
    assert normal_rows[8]["p1_hp"] == 399
    assert normal_rows[10]["p0_hp"] == 338
    assert normal_rows[10]["p1_hp"] == 261
    assert normal_rows[14]["p0_hp"] == 83
    assert normal_rows[14]["p1_hp"] == 74
    assert normal_rows[19]["p0_hp"] == 83
    assert normal_rows[19]["p1_hp"] == 0


def test_battle111_neutralizing_gas_damage_and_same_turn_confusion_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(41489),
        pool_get_team(23001),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1735706819,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 55
    assert normal_rows[10]["p0_hp"] == 349
    assert normal_rows[10]["p1_hp"] == 275
    assert normal_rows[23]["p0_hp"] == 18
    assert normal_rows[23]["p1_hp"] == 177
    assert normal_rows[32]["p0_hp"] == 383
    assert normal_rows[32]["p1_hp"] == 102
    assert normal_rows[42]["p0_hp"] == 194
    assert normal_rows[42]["p1_hp"] == 436
    assert normal_rows[43]["p0_hp"] == 139
    assert normal_rows[43]["p1_hp"] == 375
    assert normal_rows[44]["p0_hp"] == 139
    assert normal_rows[44]["p1_hp"] == 294
    assert normal_rows[46]["p0_hp"] == 83
    assert normal_rows[46]["p1_hp"] == 197
    assert normal_rows[55]["p0_hp"] == 146
    assert normal_rows[55]["p1_hp"] == 0


def test_battle128_same_terrain_switch_does_not_refresh_turns_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(1595),
        pool_get_team(6137),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        145261943,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 25
    assert normal_rows[10]["p0_hp"] == 235
    assert normal_rows[10]["p1_hp"] == 404
    assert normal_rows[13]["p0_hp"] == 199
    assert normal_rows[13]["p1_hp"] == 355
    assert normal_rows[14]["p0_hp"] == 285
    assert normal_rows[14]["p1_hp"] == 380
    assert normal_rows[20]["p0_hp"] == 74
    assert normal_rows[20]["p1_hp"] == 55
    assert normal_rows[23]["p0_hp"] == 20
    assert normal_rows[23]["p1_hp"] == 0


def test_battle118_cached_action_speed_update_frames_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(48468),
        pool_get_team(49618),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        325429550,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 34
    assert normal_rows[15]["p0_hp"] == 371
    assert normal_rows[15]["p1_hp"] == 371
    assert normal_rows[18]["p0_hp"] == 296
    assert normal_rows[18]["p1_hp"] == 300
    assert normal_rows[27]["p0_hp"] == 11
    assert normal_rows[27]["p1_hp"] == 132
    assert normal_rows[28]["p0_hp"] == 67
    assert normal_rows[28]["p1_hp"] == 64
    assert normal_rows[30]["p0_hp"] == 211
    assert normal_rows[30]["p1_hp"] == 534
    assert normal_rows[31]["p0_hp"] == 139
    assert normal_rows[31]["p0_status"] == 1
    assert normal_rows[31]["p1_hp"] == 348
    assert normal_rows[34]["p0_hp"] == 0
    assert normal_rows[34]["p1_hp"] == 142


def test_battle2_cached_action_speed_refresh_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4297),
        pool_get_team(34868),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        432652533,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 25
    assert normal_rows[18]["p0_hp"] == 121
    assert normal_rows[18]["p1_hp"] == 301
    assert normal_rows[19]["p0_hp"] == 245
    assert normal_rows[19]["p1_hp"] == 112
    assert normal_rows[20]["p0_hp"] == 121
    assert normal_rows[20]["p1_hp"] == 323
    assert normal_rows[21]["p0_hp"] == 382
    assert normal_rows[21]["p1_hp"] == 323
    assert normal_rows[22]["p0_hp"] == 382
    assert normal_rows[22]["p1_hp"] == 243
    assert normal_rows[23]["p0_hp"] == 382
    assert normal_rows[23]["p1_hp"] == 47


def test_battle34_immediate_defender_ability_state_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(40244),
        pool_get_team(368),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1711206424,
        35,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 31
    assert normal_rows[14]["p0_hp"] == 291
    assert normal_rows[14]["p1_hp"] == 28
    assert normal_rows[15]["p0_hp"] == 19
    assert normal_rows[15]["p1_hp"] == 51
    assert normal_rows[16]["p0_hp"] == 63
    assert normal_rows[16]["p1_hp"] == 337
    assert normal_rows[17]["p0_hp"] == 240
    assert normal_rows[17]["p1_hp"] == 387
    assert normal_rows[18]["p0_hp"] == 194
    assert normal_rows[18]["p1_hp"] == 51
    assert normal_rows[31]["p0_hp"] == 76
    assert normal_rows[31]["p1_hp"] == 0


def test_battle156_switch_in_restores_native_type_for_libero_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(9073),
        pool_get_team(46651),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1585542025,
        45,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 40
    assert normal_rows[31]["p0_hp"] == 28
    assert normal_rows[31]["p1_hp"] == 202
    assert normal_rows[32]["p0_hp"] == 28
    assert normal_rows[32]["p1_hp"] == 167
    assert normal_rows[37]["p0_hp"] == 28
    assert normal_rows[37]["p1_hp"] == 257
    assert normal_rows[38]["p0_hp"] == 362
    assert normal_rows[38]["p1_hp"] == 178
    assert normal_rows[39]["p0_hp"] == 272
    assert normal_rows[39]["p1_hp"] == 178
    assert normal_rows[40]["p0_hp"] == 272
    assert normal_rows[40]["p1_hp"] == 0


def test_battle637_defog_hazard_order_and_pressure_notarget_pp_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(44399),
        pool_get_team(153),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1669365125,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 69
    assert normal_rows[61]["p0_hp"] == 14
    assert normal_rows[61]["p1_hp"] == 514
    assert normal_rows[64]["p0_hp"] == 7
    assert normal_rows[64]["p1_hp"] == 514
    assert normal_rows[65]["p0_hp"] == 157
    assert normal_rows[65]["p1_hp"] == 468
    assert normal_rows[65]["p0_action"] == "move 1+pivot:switch 2"
    assert normal_rows[66]["p0_hp"] == 113
    assert normal_rows[66]["p1_hp"] == 500
    assert normal_rows[66]["p0_action"] == "switch 2+forced:switch 2"
    assert normal_rows[67]["p0_hp"] == 57
    assert normal_rows[67]["p1_hp"] == 366
    assert normal_rows[68]["p0_hp"] == 29
    assert normal_rows[68]["p1_hp"] == 292
    assert normal_rows[68]["p1_action"] == "move 4"
    assert normal_rows[69]["p0_hp"] == 0
    assert normal_rows[69]["p1_hp"] == 47
    assert normal_rows[69]["p1_action"] == "move 1"


def test_battle121_first_mover_after_move_secondary_hp_order_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(44672),
        pool_get_team(5799),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1114239844,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 55
    assert normal_rows[4]["p0_hp"] == 152
    assert normal_rows[4]["p1_hp"] == 196
    assert normal_rows[10]["p0_hp"] == 46
    assert normal_rows[10]["p1_hp"] == 357
    assert normal_rows[14]["p0_hp"] == 394
    assert normal_rows[14]["p1_hp"] == 380
    assert normal_rows[15]["p0_hp"] == 394
    assert normal_rows[15]["p1_hp"] == 318
    assert normal_rows[41]["p0_hp"] == 394
    assert normal_rows[41]["p1_hp"] == 318
    assert normal_rows[48]["p0_hp"] == 315
    assert normal_rows[48]["p1_hp"] == 245
    assert normal_rows[50]["p0_hp"] == 259
    assert normal_rows[50]["p1_hp"] == 141
    assert normal_rows[55]["p0_hp"] == 0
    assert normal_rows[55]["p1_hp"] == 294


def test_battle47_life_orb_after_drain_and_static_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(10729),
        pool_get_team(13827),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        877308582,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 21
    assert normal_rows[6]["p0_hp"] == 71
    assert normal_rows[6]["p1_hp"] == 275
    assert normal_rows[6]["p1_status"] == 0
    assert normal_rows[7]["p0_hp"] == 273
    assert normal_rows[7]["p1_hp"] == 275
    assert normal_rows[7]["p1_status"] == 0
    assert normal_rows[10]["p0_hp"] == 301
    assert normal_rows[10]["p1_hp"] == 197
    assert normal_rows[10]["p1_status"] == 0
    assert normal_rows[11]["p0_hp"] == 275
    assert normal_rows[11]["p1_hp"] == 249
    assert normal_rows[11]["p1_status"] == 0
    assert normal_rows[12]["p0_hp"] == 168
    assert normal_rows[12]["p1_hp"] == 275
    assert normal_rows[12]["p1_status"] == 2
    assert normal_rows[13]["p0_hp"] == 61
    assert normal_rows[13]["p1_hp"] == 275
    assert normal_rows[13]["p1_status"] == 2
    assert normal_rows[20]["p0_hp"] == 116
    assert normal_rows[20]["p1_hp"] == 136
    assert normal_rows[20]["p1_status"] == 2
    assert normal_rows[21]["p0_hp"] == 0
    assert normal_rows[21]["p1_hp"] == 193
    assert normal_rows[21]["p1_status"] == 2


def test_choice_lock_out_of_pp_uses_locked_slot_for_struggle():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4297),
        pool_get_team(34868),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        432652533,
        65,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}
    assert normal_rows[61]["p0_hp"] == 164
    assert normal_rows[61]["p1_hp"] == 318
    assert normal_rows[62]["p0_hp"] == 321
    assert normal_rows[62]["p1_hp"] == 295


def test_battle7_hazard_ko_notarget_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(18539),
        pool_get_team(9127),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1990212659,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 17
    assert normal_rows[15]["p0_hp"] == 258
    assert normal_rows[15]["p1_hp"] == 299
    assert normal_rows[16]["p1_hp"] == 384
    assert normal_rows[17]["p0_hp"] == 0
    assert normal_rows[17]["p1_hp"] == 300


def test_battle9_terminal_hazard_switch_chain_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(41138),
        pool_get_team(27271),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        952224741,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 48
    assert normal_rows[45]["p0_hp"] == 187
    assert normal_rows[45]["p1_hp"] == 2
    assert normal_rows[48]["p0_hp"] == 127
    assert normal_rows[48]["p1_hp"] == 0


def test_battle10_confusion_onbeforemove_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(22522),
        pool_get_team(11361),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        197860369,
        11,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 11
    assert normal_rows[2]["p0_hp"] == 183
    assert normal_rows[2]["p1_hp"] == 291
    assert normal_rows[3]["p0_hp"] == 205
    assert normal_rows[3]["p1_hp"] == 135
    assert normal_rows[4]["p0_hp"] == 297
    assert normal_rows[4]["p1_hp"] == 105
    assert normal_rows[11]["p0_hp"] == 293
    assert normal_rows[11]["p1_hp"] == 0


def test_battle12_protosynthesis_speed_order_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42914),
        pool_get_team(41381),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        594336918,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[7]["p0_hp"] == 389
    assert normal_rows[7]["p1_hp"] == 301
    assert normal_rows[9]["p0_hp"] == 361
    assert normal_rows[9]["p1_hp"] == 268
    assert normal_rows[11]["p0_hp"] == 331
    assert normal_rows[11]["p1_hp"] == 231
    assert normal_rows[44]["p0_hp"] == 221
    assert normal_rows[44]["p1_hp"] == 258


def test_battle14_hyperspace_fury_and_recoil_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(35026),
        pool_get_team(17726),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        145857092,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[4]["p0_hp"] == 400
    assert normal_rows[4]["p1_hp"] == 434
    assert normal_rows[6]["p0_hp"] == 260
    assert normal_rows[6]["p1_hp"] == 123
    assert normal_rows[11]["p0_hp"] == 138
    assert normal_rows[11]["p1_hp"] == 434
    assert normal_rows[12]["p0_hp"] == 89
    assert normal_rows[12]["p1_hp"] == 216


def test_battle16_prng_and_beat_up_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(33895),
        pool_get_team(38919),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1631871624,
        52,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[13]["p0_hp"] == 371
    assert normal_rows[13]["p1_hp"] == 319
    assert normal_rows[14]["p0_hp"] == 310
    assert normal_rows[14]["p1_hp"] == 3
    assert normal_rows[15]["p0_hp"] == 392
    assert normal_rows[15]["p1_hp"] == 3
    assert normal_rows[49]["p0_hp"] == 142
    assert normal_rows[49]["p1_hp"] == 33
    assert normal_rows[50]["p0_hp"] == 142
    assert normal_rows[50]["p1_hp"] == 3
    assert normal_rows[52]["p0_hp"] == 142
    assert normal_rows[52]["p1_hp"] == 0


def test_battle18_future_sight_offfield_item_suppression_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(24889),
        pool_get_team(2190),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1173749072,
        66,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[4]["p0_hp"] == 246
    assert normal_rows[4]["p1_hp"] == 315
    assert normal_rows[21]["p0_hp"] == 391
    assert normal_rows[21]["p1_hp"] == 209
    assert normal_rows[38]["p0_hp"] == 186
    assert normal_rows[38]["p1_hp"] == 56
    assert normal_rows[44]["p0_hp"] == 304
    assert normal_rows[44]["p1_hp"] == 295
    assert normal_rows[46]["p0_hp"] == 404
    assert normal_rows[46]["p1_hp"] == 274


def test_battle19_flip_turn_and_futuresight_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(7714),
        pool_get_team(37168),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1466836457,
        46,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 34
    assert normal_rows[5]["p0_hp"] == 283
    assert normal_rows[5]["p1_hp"] == 317
    assert normal_rows[8]["p0_hp"] == 303
    assert normal_rows[8]["p1_hp"] == 528
    assert normal_rows[21]["p0_hp"] == 434
    assert normal_rows[21]["p1_hp"] == 63
    assert normal_rows[31]["p0_hp"] == 283
    assert normal_rows[31]["p1_hp"] == 534
    assert normal_rows[32]["p0_hp"] == 280
    assert normal_rows[32]["p1_hp"] == 288
    assert normal_rows[32]["p1_status"] == 5
    assert normal_rows[33]["p0_hp"] == 277
    assert normal_rows[33]["p1_hp"] == 32
    assert normal_rows[34]["p0_hp"] == 258
    assert normal_rows[34]["p1_hp"] == 0
    assert normal_rows[34]["p1_status"] == 0


def test_battle20_switch_order_confusion_and_ko_update_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(46126),
        pool_get_team(37238),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        787359109,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 60
    assert normal_rows[5]["p0_hp"] == 257
    assert normal_rows[5]["p0_status"] == 1
    assert normal_rows[5]["p1_hp"] == 69
    assert normal_rows[9]["p0_hp"] == 284
    assert normal_rows[9]["p1_hp"] == 393
    assert normal_rows[20]["p0_hp"] == 318
    assert normal_rows[20]["p1_hp"] == 369
    assert normal_rows[39]["p0_hp"] == 284
    assert normal_rows[39]["p1_hp"] == 127
    assert normal_rows[40]["p1_hp"] == 345
    assert normal_rows[41]["p1_action"] == "switch 4"
    assert normal_rows[41]["p1_hp"] == 316
    assert normal_rows[43]["p1_hp"] == 210
    assert normal_rows[60]["p0_hp"] == 166
    assert normal_rows[60]["p1_hp"] == 0


def test_battle24_contact_ko_notarget_prng_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(23147),
        pool_get_team(6496),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1474200732,
        26,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 26
    assert normal_rows[15]["p0_hp"] == 196
    assert normal_rows[15]["p1_hp"] == 52
    assert normal_rows[16]["p0_hp"] == 133
    assert normal_rows[16]["p1_hp"] == 236
    assert normal_rows[18]["p0_hp"] == 105
    assert normal_rows[18]["p1_hp"] == 97


def test_battle26_status_immunity_and_phaze_queue_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, p0_actions, p1_actions = run_pokepy_battle(
        pool_get_team(28224),
        pool_get_team(33490),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2019291965,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Thunder Wave into Ground should not add a stray accuracy frame, and
    # phazing drags should not be serialized into the replay queue as fake
    # player switch choices.
    assert p0_actions[:10] == [
        "move 2",
        "move 1",
        "move 4",
        "move 1",
        "move 4",
        "switch 5",
        "switch 5",
        "move 1",
        "switch 4",
        "switch 6",
    ]
    assert p1_actions[:10] == ["move 2"] + ["move 1"] * 9
    assert normal_rows[8]["p0_action"] == "move 1"
    assert normal_rows[8]["p0_hp"] == 258
    assert normal_rows[9]["p0_action"] == "switch 4"
    assert normal_rows[9]["p0_hp"] == 379


def test_battle27_inline_item_switch_and_terminal_uturn_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(21857),
        pool_get_team(8034),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1788162810,
        22,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 22
    assert normal_rows[12]["p0_hp"] == 58
    assert normal_rows[12]["p1_hp"] == 104
    assert normal_rows[12]["p1_max_hp"] == 291
    assert normal_rows[21]["p0_hp"] == 230
    assert normal_rows[21]["p0_max_hp"] == 382
    assert normal_rows[21]["p1_hp"] == 24
    assert normal_rows[22]["p0_hp"] == 155
    assert normal_rows[22]["p0_max_hp"] == 382
    assert normal_rows[22]["p1_hp"] == 0


def test_battle8_inline_eject_button_cancellation_regression_rows_match():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(39078),
        pool_get_team(32193),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        864178267,
        55,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 55
    assert normal_rows[7]["p1_hp"] == 321
    assert normal_rows[7]["p1_max_hp"] == 321
    assert normal_rows[8]["p1_hp"] == 341
    assert normal_rows[8]["p1_max_hp"] == 341
    assert normal_rows[55]["p0_hp"] == 0
    assert normal_rows[55]["p1_hp"] == 159


def test_battle32_freeze_immunity_and_libero_timing_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(11977),
        pool_get_team(34124),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1367435608,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[12]["p0_hp"] == 342
    assert normal_rows[12]["p1_hp"] == 237
    assert normal_rows[12]["p1_status"] == 0
    assert normal_rows[19]["p0_hp"] == 301
    assert normal_rows[19]["p1_hp"] == 167
    assert normal_rows[20]["p0_active"] == 4
    assert normal_rows[20]["p0_hp"] == 271
    assert normal_rows[20]["p1_hp"] == 167
    assert normal_rows[21]["p0_hp"] == 202
    assert normal_rows[21]["p1_hp"] == 167
    assert normal_rows[25]["p0_hp"] == 46
    assert normal_rows[25]["p1_hp"] == 0


def test_battle33_inline_knock_off_item_order_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(6987),
        pool_get_team(41636),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        429299595,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 23
    assert normal_rows[10]["p0_hp"] == 282
    assert normal_rows[10]["p1_hp"] == 399
    assert normal_rows[11]["p0_hp"] == 238
    assert normal_rows[11]["p1_hp"] == 237
    assert normal_rows[12]["p0_hp"] == 254
    assert normal_rows[12]["p1_hp"] == 130
    assert normal_rows[13]["p0_hp"] == 269
    assert normal_rows[13]["p1_hp"] == 25
    assert normal_rows[23]["p0_hp"] == 332
    assert normal_rows[23]["p1_hp"] == 0


def test_battle34_good_as_gold_terrain_and_switch_update_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(40244),
        pool_get_team(368),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1711206424,
        32,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 31
    assert normal_rows[14]["p0_hp"] == 291
    assert normal_rows[14]["p1_hp"] == 28
    assert normal_rows[15]["p0_hp"] == 19
    assert normal_rows[15]["p1_hp"] == 51
    assert normal_rows[16]["p0_hp"] == 63
    assert normal_rows[16]["p1_hp"] == 337
    assert normal_rows[17]["p0_hp"] == 240
    assert normal_rows[17]["p1_hp"] == 387
    assert normal_rows[31]["p0_hp"] == 76
    assert normal_rows[31]["p1_hp"] == 0


def test_battle36_rapid_spin_faster_onafterhit_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(23578),
        pool_get_team(35258),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        595114158,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 48
    assert normal_rows[12]["p0_hp"] == 130
    assert normal_rows[12]["p1_hp"] == 511
    assert normal_rows[13]["p0_hp"] == 130
    assert normal_rows[13]["p1_hp"] == 472
    assert normal_rows[18]["p0_hp"] == 130
    assert normal_rows[18]["p1_hp"] == 374
    assert normal_rows[22]["p0_hp"] == 130
    assert normal_rows[22]["p1_hp"] == 303
    assert normal_rows[23]["p0_hp"] == 130
    assert normal_rows[23]["p1_hp"] == 283
    assert normal_rows[36]["p0_hp"] == 130
    assert normal_rows[36]["p1_hp"] == 75
    assert normal_rows[47]["p0_hp"] == 125
    assert normal_rows[47]["p1_hp"] == 89
    assert normal_rows[48]["p0_hp"] == 125
    assert normal_rows[48]["p1_hp"] == 0


def test_battle37_fixed_multihit_count_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(39036),
        pool_get_team(27782),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        985514123,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 29
    assert normal_rows[12]["p0_hp"] == 60
    assert normal_rows[12]["p1_hp"] == 322
    assert normal_rows[13]["p0_hp"] == 79
    assert normal_rows[13]["p1_hp"] == 279
    assert normal_rows[14]["p0_hp"] == 98
    assert normal_rows[14]["p1_hp"] == 322
    assert normal_rows[15]["p0_hp"] == 117
    assert normal_rows[15]["p1_hp"] == 249
    assert normal_rows[16]["p0_hp"] == 264
    assert normal_rows[16]["p1_hp"] == 146
    assert normal_rows[16]["p0_status"] == 6
    assert normal_rows[29]["p0_hp"] == 394
    assert normal_rows[29]["p1_hp"] == 0


def test_battle28_charge_turn_residual_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(31488),
        pool_get_team(35013),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        208792607,
        45,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 41
    assert normal_rows[33]["p0_hp"] == 97
    assert normal_rows[33]["p1_hp"] == 264
    assert normal_rows[36]["p0_hp"] == 334
    assert normal_rows[36]["p1_hp"] == 307
    assert normal_rows[37]["p0_hp"] == 273
    assert normal_rows[37]["p1_hp"] == 91
    assert normal_rows[38]["p0_hp"] == 8
    assert normal_rows[38]["p1_hp"] == 236
    assert normal_rows[41]["p0_hp"] == 8
    assert normal_rows[41]["p1_hp"] == 0


def test_battle38_seed_sower_and_terminal_forced_switch_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(25280),
        pool_get_team(28437),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        80020205,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 23
    assert normal_rows[14]["p0_hp"] == 248
    assert normal_rows[14]["p1_hp"] == 315
    assert normal_rows[15]["p0_hp"] == 166
    assert normal_rows[15]["p1_hp"] == 311
    assert normal_rows[16]["p0_hp"] == 66
    assert normal_rows[16]["p1_hp"] == 307
    assert normal_rows[23]["p0_hp"] == 0
    assert normal_rows[23]["p1_hp"] == 273


def test_battle39_pressure_and_lockedmove_restart_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(6989),
        pool_get_team(12274),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        245951460,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 65
    assert normal_rows[29]["p0_hp"] == 129
    assert normal_rows[29]["p1_hp"] == 514
    assert normal_rows[51]["p0_hp"] == 235
    assert normal_rows[51]["p1_hp"] == 156
    assert normal_rows[52]["p0_hp"] == 227
    assert normal_rows[52]["p1_hp"] == 156
    assert normal_rows[53]["p0_hp"] == 220
    assert normal_rows[53]["p1_hp"] == 156
    assert normal_rows[54]["p0_hp"] == 243
    assert normal_rows[54]["p1_hp"] == 113
    assert normal_rows[65]["p0_hp"] == 250
    assert normal_rows[65]["p1_hp"] == 0


def test_battle126_charge_lock_followup_switch_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(18999),
        pool_get_team(13123),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        11396850,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 23
    assert normal_rows[13]["p0_hp"] == 63
    assert normal_rows[13]["p1_hp"] == 358
    assert normal_rows[14]["p0_action"] == "move 1+forced:switch 4"
    assert normal_rows[14]["p0_hp"] == 279
    assert normal_rows[14]["p1_hp"] == 335
    assert normal_rows[17]["p0_action"] == "move 2+forced:switch 5"
    assert normal_rows[17]["p1_action"] == "move 1+forced:switch 6"
    assert normal_rows[17]["p0_hp"] == 258
    assert normal_rows[17]["p1_hp"] == 404
    assert normal_rows[23]["p0_hp"] == 0
    assert normal_rows[23]["p1_hp"] == 312
    assert normal_rows[23]["p1_status"] == 5


def test_battle140_crash_notarget_and_sparkling_aria_secondary_roll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(46375),
        pool_get_team(34713),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1645376471,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 18
    assert normal_rows[4]["p1_action"] == "move 3+forced:switch 2"
    assert normal_rows[4]["p0_hp"] == 391
    assert normal_rows[4]["p1_hp"] == 364
    assert normal_rows[5]["p0_hp"] == 242
    assert normal_rows[5]["p1_hp"] == 364
    assert normal_rows[6]["p0_action"] == "move 2+forced:switch 3"
    assert normal_rows[6]["p0_hp"] == 315
    assert normal_rows[6]["p1_hp"] == 28
    assert normal_rows[7]["p1_action"] == "move 1+forced:switch 3"
    assert normal_rows[7]["p0_hp"] == 315
    assert normal_rows[7]["p1_hp"] == 384
    assert normal_rows[8]["p0_hp"] == 133
    assert normal_rows[8]["p1_hp"] == 311
    assert normal_rows[9]["p0_action"] == "move 4+forced:switch 2"
    assert normal_rows[9]["p0_hp"] == 391
    assert normal_rows[9]["p1_hp"] == 173
    assert normal_rows[18]["p0_hp"] == 141
    assert normal_rows[18]["p1_hp"] == 0


def test_battle149_partial_trap_source_faint_and_fickle_beam_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(47781),
        pool_get_team(26051),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        615138588,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 29
    assert normal_rows[25]["p0_action"] == "move 3"
    assert normal_rows[25]["p0_hp"] == 36
    assert normal_rows[25]["p0_max_hp"] == 424
    assert normal_rows[25]["p1_action"] == "move 1+forced:switch 5"
    assert normal_rows[25]["p1_hp"] == 12
    assert normal_rows[25]["p1_max_hp"] == 323
    assert normal_rows[26]["p0_action"] == "move 2"
    assert normal_rows[26]["p0_hp"] == 62
    assert normal_rows[26]["p0_max_hp"] == 424
    assert normal_rows[26]["p1_action"] == "switch 6"
    assert normal_rows[26]["p1_hp"] == 258
    assert normal_rows[26]["p1_max_hp"] == 415
    assert normal_rows[27]["p0_action"] == "move 2+forced:switch 3"
    assert normal_rows[27]["p0_hp"] == 332
    assert normal_rows[27]["p0_max_hp"] == 379
    assert normal_rows[27]["p1_action"] == "move 2"
    assert normal_rows[27]["p1_hp"] == 258
    assert normal_rows[27]["p1_max_hp"] == 415
    assert normal_rows[28]["p0_action"] == "move 1"
    assert normal_rows[28]["p0_hp"] == 201
    assert normal_rows[28]["p0_max_hp"] == 379
    assert normal_rows[28]["p1_action"] == "move 3+forced:switch 6"
    assert normal_rows[28]["p1_hp"] == 12
    assert normal_rows[28]["p1_max_hp"] == 323
    assert normal_rows[29]["p0_action"] == "move 1"
    assert normal_rows[29]["p0_hp"] == 12
    assert normal_rows[29]["p0_max_hp"] == 379
    assert normal_rows[29]["p1_action"] == "move 4"
    assert normal_rows[29]["p1_hp"] == 0
    assert normal_rows[29]["p1_max_hp"] == 323


def test_battle155_eject_button_cancels_slower_move_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(48311),
        pool_get_team(8298),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1279991636,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 21
    assert normal_rows[18]["p0_action"] == "move 2+forced:switch 3"
    assert normal_rows[18]["p0_hp"] == 279
    assert normal_rows[18]["p0_max_hp"] == 318
    assert normal_rows[18]["p1_action"] == "move 2"
    assert normal_rows[18]["p1_hp"] == 206
    assert normal_rows[18]["p1_max_hp"] == 371
    assert normal_rows[19]["p0_action"] == "move 3+forced:switch 6"
    assert normal_rows[19]["p0_hp"] == 16
    assert normal_rows[19]["p0_max_hp"] == 301
    assert normal_rows[19]["p1_action"] == "move 1"
    assert normal_rows[19]["p1_hp"] == 206
    assert normal_rows[19]["p1_max_hp"] == 371
    assert normal_rows[20]["p0_action"] == "move 2+forced:switch 6"
    assert normal_rows[20]["p0_hp"] == 113
    assert normal_rows[20]["p0_max_hp"] == 318
    assert normal_rows[20]["p1_action"] == "move 1"
    assert normal_rows[20]["p1_hp"] == 206
    assert normal_rows[20]["p1_max_hp"] == 371
    assert normal_rows[21]["p0_action"] == "move 3"
    assert normal_rows[21]["p0_hp"] == 0
    assert normal_rows[21]["p0_max_hp"] == 318
    assert normal_rows[21]["p1_action"] == "move 1"
    assert normal_rows[21]["p1_hp"] == 206
    assert normal_rows[21]["p1_max_hp"] == 371


def test_battle158_healing_wish_showdown_row_status_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(31715),
        pool_get_team(39238),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1818874729,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 44
    assert normal_rows[32]["p0_action"] == "move 3"
    assert normal_rows[32]["p0_hp"] == 318
    assert normal_rows[32]["p0_max_hp"] == 318
    assert normal_rows[32]["p0_status"] == 0
    assert normal_rows[32]["p1_action"] == "switch 6"
    assert normal_rows[32]["p1_hp"] == 281
    assert normal_rows[32]["p1_max_hp"] == 321
    assert normal_rows[33]["p0_action"] == "move 2+forced:switch 2"
    assert normal_rows[33]["p0_hp"] == 463
    assert normal_rows[33]["p0_max_hp"] == 463
    assert normal_rows[33]["p0_status"] == 0
    assert normal_rows[33]["p1_action"] == "switch 6"
    assert normal_rows[33]["p1_hp"] == 391
    assert normal_rows[33]["p1_max_hp"] == 391
    assert normal_rows[34]["p0_action"] == "move 2"
    assert normal_rows[34]["p0_hp"] == 463
    assert normal_rows[34]["p0_max_hp"] == 463
    assert normal_rows[34]["p0_status"] == 1
    assert normal_rows[34]["p1_action"] == "switch 6"
    assert normal_rows[34]["p1_hp"] == 281
    assert normal_rows[34]["p1_max_hp"] == 321
    assert normal_rows[44]["p0_action"] == "move 1"
    assert normal_rows[44]["p0_hp"] == 0
    assert normal_rows[44]["p0_max_hp"] == 444
    assert normal_rows[44]["p1_action"] == "move 2"
    assert normal_rows[44]["p1_hp"] == 281
    assert normal_rows[44]["p1_max_hp"] == 321


def test_battle158_showdown_runner_clears_healing_wish_status():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    team0 = pool_get_team(31715)
    team1 = pool_get_team(39238)
    battle_seed = 1818874729
    seed_tuple = (battle_seed & 0xFFFF, (battle_seed >> 16) & 0xFFFF, 0, 0)

    _, p0_actions, p1_actions = run_pokepy_battle(
        team0,
        team1,
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        battle_seed,
        50,
    )

    show_rows, raw_data, meta = run_showdown(
        seed_tuple,
        team_to_showdown_packed(team0, mappings),
        team_to_showdown_packed(team1, mappings),
        p0_actions,
        p1_actions,
        50,
        timeout_s=120,
    )

    assert meta["returncode"] == 0
    assert not meta["timeout"]
    assert meta["error"] is None
    assert raw_data is not None
    assert show_rows

    by_turn = {row["turn"]: row for row in show_rows}
    assert by_turn[33]["p0_hp"] == 463
    assert by_turn[33]["p0_max_hp"] == 463
    assert by_turn[33]["p0_status"] == 0
    assert by_turn[34]["p0_hp"] == 463
    assert by_turn[34]["p0_max_hp"] == 463
    assert by_turn[34]["p0_status"] == 1


def test_battle172_struggle_does_not_trigger_libero_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(44081),
        pool_get_team(3327),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1020958852,
        45,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 39
    assert normal_rows[12]["p0_action"] == "move 1"
    assert normal_rows[12]["p0_hp"] == 226
    assert normal_rows[12]["p0_max_hp"] == 301
    assert normal_rows[12]["p0_status"] == 0
    assert normal_rows[12]["p1_action"] == "switch 4"
    assert normal_rows[12]["p1_hp"] == 169
    assert normal_rows[12]["p1_max_hp"] == 379
    assert normal_rows[12]["p1_status"] == 0
    assert normal_rows[13]["p0_action"] == "move 1+forced:switch 2"
    assert normal_rows[13]["p0_hp"] == 353
    assert normal_rows[13]["p0_max_hp"] == 403
    assert normal_rows[13]["p0_status"] == 0
    assert normal_rows[13]["p1_action"] == "move 1"
    assert normal_rows[13]["p1_hp"] == 109
    assert normal_rows[13]["p1_max_hp"] == 379
    assert normal_rows[13]["p1_status"] == 0
    assert normal_rows[14]["p0_action"] == "move 1"
    assert normal_rows[14]["p0_hp"] == 378
    assert normal_rows[14]["p0_max_hp"] == 403
    assert normal_rows[14]["p0_status"] == 0
    assert normal_rows[14]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[14]["p1_hp"] == 121
    assert normal_rows[14]["p1_max_hp"] == 321
    assert normal_rows[14]["p1_status"] == 0
    assert normal_rows[39]["p0_action"] == "move 4"
    assert normal_rows[39]["p0_hp"] == 294
    assert normal_rows[39]["p0_max_hp"] == 391
    assert normal_rows[39]["p0_status"] == 0
    assert normal_rows[39]["p1_action"] == "move 4"
    assert normal_rows[39]["p1_hp"] == 0
    assert normal_rows[39]["p1_max_hp"] == 387
    assert normal_rows[39]["p1_status"] == 0


def test_battle173_miracle_berry_cures_early_status_before_slower_move_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(29721),
        pool_get_team(17893),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        313904651,
        35,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 27
    assert normal_rows[15]["p0_action"] == "move 4"
    assert normal_rows[15]["p0_hp"] == 107
    assert normal_rows[15]["p0_max_hp"] == 351
    assert normal_rows[15]["p0_status"] == 0
    assert normal_rows[15]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[15]["p1_hp"] == 273
    assert normal_rows[15]["p1_max_hp"] == 434
    assert normal_rows[15]["p1_status"] == 1
    assert normal_rows[16]["p0_action"] == "move 4"
    assert normal_rows[16]["p0_hp"] == 107
    assert normal_rows[16]["p0_max_hp"] == 351
    assert normal_rows[16]["p0_status"] == 0
    assert normal_rows[16]["p1_action"] == "switch 2"
    assert normal_rows[16]["p1_hp"] == 364
    assert normal_rows[16]["p1_max_hp"] == 386
    assert normal_rows[16]["p1_status"] == 0
    assert normal_rows[17]["p0_action"] == "move 1"
    assert normal_rows[17]["p0_hp"] == 56
    assert normal_rows[17]["p0_max_hp"] == 351
    assert normal_rows[17]["p0_status"] == 0
    assert normal_rows[17]["p1_action"] == "move 2+forced:switch 2"
    assert normal_rows[17]["p1_hp"] == 273
    assert normal_rows[17]["p1_max_hp"] == 434
    assert normal_rows[17]["p1_status"] == 1
    assert normal_rows[18]["p0_action"] == "move 4"
    assert normal_rows[18]["p0_hp"] == 56
    assert normal_rows[18]["p0_max_hp"] == 351
    assert normal_rows[18]["p0_status"] == 0
    assert normal_rows[18]["p1_action"] == "switch 5"
    assert normal_rows[18]["p1_hp"] == 304
    assert normal_rows[18]["p1_max_hp"] == 380
    assert normal_rows[18]["p1_status"] == 0
    assert normal_rows[27]["p0_action"] == "move 2"
    assert normal_rows[27]["p0_hp"] == 297
    assert normal_rows[27]["p0_max_hp"] == 385
    assert normal_rows[27]["p0_status"] == 0
    assert normal_rows[27]["p1_action"] == "move 1"
    assert normal_rows[27]["p1_hp"] == 0
    assert normal_rows[27]["p1_max_hp"] == 380
    assert normal_rows[27]["p1_status"] == 0


def test_battle178_double_faint_replacement_intimidate_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(28870),
        pool_get_team(7680),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        978517948,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[13]["p0_action"] == "switch 6+forced:switch 2"
    assert normal_rows[13]["p0_hp"] == 405
    assert normal_rows[13]["p0_max_hp"] == 524
    assert normal_rows[13]["p1_action"] == "move 3+forced:switch 2"
    assert normal_rows[13]["p1_hp"] == 319
    assert normal_rows[13]["p1_max_hp"] == 319
    assert normal_rows[14]["p0_action"] == "move 1"
    assert normal_rows[14]["p0_hp"] == 318
    assert normal_rows[14]["p0_max_hp"] == 524
    assert normal_rows[14]["p1_action"] == "move 2"
    assert normal_rows[14]["p1_hp"] == 71
    assert normal_rows[14]["p1_max_hp"] == 319
    assert normal_rows[15]["p0_action"] == "move 1"
    assert normal_rows[15]["p0_hp"] == 186
    assert normal_rows[15]["p0_max_hp"] == 524
    assert normal_rows[15]["p1_action"] == "move 4+forced:switch 3"
    assert normal_rows[15]["p1_hp"] == 291
    assert normal_rows[15]["p1_max_hp"] == 364


def test_battle181_switch_in_restores_base_ability_and_reactive_trace_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(8732),
        pool_get_team(41549),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        114636827,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[30]["p0_action"] == "move 4"
    assert normal_rows[30]["p0_hp"] == 45
    assert normal_rows[30]["p0_max_hp"] == 339
    assert normal_rows[30]["p1_action"] == "move 3+forced:switch 6"
    assert normal_rows[30]["p1_hp"] == 277
    assert normal_rows[30]["p1_max_hp"] == 277
    assert normal_rows[31]["p0_action"] == "switch 2+forced:switch 2"
    assert normal_rows[31]["p0_hp"] == 45
    assert normal_rows[31]["p0_max_hp"] == 339
    assert normal_rows[31]["p1_action"] == "move 2"
    assert normal_rows[31]["p1_hp"] == 277
    assert normal_rows[31]["p1_max_hp"] == 277
    assert normal_rows[32]["p0_action"] == "move 1"
    assert normal_rows[32]["p0_hp"] == 0
    assert normal_rows[32]["p0_max_hp"] == 339
    assert normal_rows[32]["p1_action"] == "move 2"
    assert normal_rows[32]["p1_hp"] == 82
    assert normal_rows[32]["p1_max_hp"] == 277


def test_battle184_growth_uses_effective_sun_weather_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(41410),
        pool_get_team(15899),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2114031675,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[2]["p0_action"] == "move 2"
    assert normal_rows[2]["p0_hp"] == 301
    assert normal_rows[2]["p0_max_hp"] == 301
    assert normal_rows[2]["p1_action"] == "switch 6"
    assert normal_rows[2]["p1_hp"] == 379
    assert normal_rows[2]["p1_max_hp"] == 379
    assert normal_rows[3]["p0_action"] == "move 3"
    assert normal_rows[3]["p0_hp"] == 171
    assert normal_rows[3]["p0_max_hp"] == 301
    assert normal_rows[3]["p1_action"] == "move 4"
    assert normal_rows[3]["p1_hp"] == 209
    assert normal_rows[3]["p1_max_hp"] == 379
    assert normal_rows[4]["p0_hp"] == 115
    assert normal_rows[4]["p1_hp"] == 45


def test_battle187_notarget_status_move_does_not_clear_hazards_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(2173),
        pool_get_team(24654),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        389764866,
        100,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[58]["p0_action"] == "switch 3"
    assert normal_rows[58]["p0_hp"] == 127
    assert normal_rows[58]["p1_action"] == "move 4"
    assert normal_rows[58]["p1_hp"] == 295
    assert normal_rows[59]["p0_action"] == "move 2"
    assert normal_rows[59]["p0_hp"] == 127
    assert normal_rows[59]["p1_action"] == "switch 2"
    assert normal_rows[59]["p1_hp"] == 256
    assert normal_rows[60]["p0_action"] == "switch 3+forced:switch 4"
    assert normal_rows[60]["p0_hp"] == 421
    assert normal_rows[60]["p0_max_hp"] == 449
    assert normal_rows[60]["p1_action"] == "move 3"
    assert normal_rows[60]["p1_hp"] == 256
    assert normal_rows[60]["p1_max_hp"] == 400
    assert normal_rows[61]["p0_hp"] == 375
    assert normal_rows[61]["p1_hp"] == 170
    assert normal_rows[62]["p0_hp"] == 329
    assert normal_rows[62]["p1_hp"] == 170


def test_battle35_pivot_hazard_ko_does_not_cancel_foe_side_move_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(39346),
        pool_get_team(39018),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1427756342,
        120,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[53]["p0_action"] == "move 2"
    assert normal_rows[53]["p0_hp"] == 300
    assert normal_rows[53]["p1_action"] == "switch 5"
    assert normal_rows[53]["p1_hp"] == 321
    assert normal_rows[54]["p0_action"] == "move 2"
    assert normal_rows[54]["p0_hp"] == 252
    assert normal_rows[54]["p1_action"] == "move 1+forced:switch 5+die_switch:switch 5"
    assert normal_rows[54]["p1_hp"] == 321
    assert normal_rows[55]["p0_action"] == "move 2"
    assert normal_rows[55]["p0_hp"] == 206
    assert normal_rows[55]["p1_action"] == "move 1+forced:switch 3"
    assert normal_rows[55]["p1_hp"] == 226
    assert normal_rows[55]["p1_max_hp"] == 317
    assert normal_rows[56]["p0_hp"] == 202
    assert normal_rows[56]["p1_hp"] == 226


def test_battle188_clanging_scales_selfboost_updates_before_body_press_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(44455),
        pool_get_team(11837),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        489776228,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[6]["p0_action"] == "switch 2"
    assert normal_rows[6]["p0_hp"] == 325
    assert normal_rows[6]["p1_action"] == "move 4"
    assert normal_rows[6]["p1_hp"] == 231
    assert normal_rows[7]["p0_action"] == "move 3"
    assert normal_rows[7]["p0_hp"] == 195
    assert normal_rows[7]["p1_action"] == "move 2"
    assert normal_rows[7]["p1_hp"] == 41
    assert normal_rows[7]["p1_max_hp"] == 291
    assert normal_rows[8]["p0_hp"] == 65
    assert normal_rows[8]["p1_action"] == "move 2+forced:switch 6"
    assert normal_rows[8]["p1_hp"] == 162
    assert normal_rows[8]["p1_max_hp"] == 364


def test_battle189_magician_steals_helmet_from_fainted_target_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(12469),
        pool_get_team(32128),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1226712778,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[8]["p0_action"] == "move 2"
    assert normal_rows[8]["p0_hp"] == 364
    assert normal_rows[8]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[8]["p1_hp"] == 301
    assert normal_rows[9]["p0_action"] == "move 4+forced:switch 6"
    assert normal_rows[9]["p0_hp"] == 251
    assert normal_rows[9]["p1_action"] == "move 3"
    assert normal_rows[9]["p1_hp"] == 251
    assert normal_rows[9]["p1_max_hp"] == 301
    assert normal_rows[10]["p0_hp"] == 251
    assert normal_rows[10]["p1_hp"] == 434


def test_battle192_liquid_voice_rewrites_psychic_noise_to_water_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(46290),
        pool_get_team(26187),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1918878899,
        6,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[1]["p0_action"] == "move 2"
    assert normal_rows[1]["p0_hp"] == 339
    assert normal_rows[1]["p1_action"] == "switch 4"
    assert normal_rows[1]["p1_hp"] == 321
    assert normal_rows[2]["p0_action"] == "move 1"
    assert normal_rows[2]["p0_hp"] == 285
    assert normal_rows[2]["p1_action"] == "move 1"
    assert normal_rows[2]["p1_hp"] == 216
    assert normal_rows[3]["p0_hp"] == 232
    assert normal_rows[3]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[3]["p1_hp"] == 341


def test_battle196_sticky_web_contrary_switchin_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(41856),
        pool_get_team(42178),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1719482721,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[11]["p0_hp"] == 21
    assert normal_rows[11]["p1_action"] == "move 2+forced:switch 2"
    assert normal_rows[11]["p1_hp"] == 255
    assert normal_rows[12]["p0_action"] == "move 2+forced:switch 4"
    assert normal_rows[12]["p0_hp"] == 250
    assert normal_rows[12]["p1_action"] == "move 3"
    assert normal_rows[12]["p1_hp"] == 291
    assert normal_rows[13]["p0_hp"] == 274
    assert normal_rows[13]["p1_hp"] == 255


def test_battle199_cute_charm_contact_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(38973),
        pool_get_team(31390),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1379722329,
        70,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[47]["p0_hp"] == 195
    assert normal_rows[47]["p1_hp"] == 166
    assert normal_rows[48]["p0_action"] == "move 3"
    assert normal_rows[48]["p0_hp"] == 3
    assert normal_rows[48]["p1_action"] == "move 4+forced:switch 2"
    assert normal_rows[48]["p1_hp"] == 386
    assert normal_rows[49]["p0_hp"] == 3
    assert normal_rows[49]["p1_hp"] == 287


def test_battle200_deferred_absorb_after_struggle_recoil_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(30814),
        pool_get_team(38949),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1587082674,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[48]["p0_hp"] == 392
    assert normal_rows[48]["p1_hp"] == 299
    assert normal_rows[49]["p0_action"] == "move 4"
    assert normal_rows[49]["p0_hp"] == 353
    assert normal_rows[49]["p1_action"] == "move 2"
    assert normal_rows[49]["p1_hp"] == 298
    assert normal_rows[50]["p0_hp"] == 309
    assert normal_rows[50]["p1_hp"] == 297


def test_battle21_steam_eruption_secondary_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(48375),
        pool_get_team(20542),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        699704628,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 57
    assert normal_rows[23]["p0_hp"] == 319
    assert normal_rows[23]["p1_hp"] == 129
    assert normal_rows[24]["p0_hp"] == 404
    assert normal_rows[24]["p1_hp"] == 129
    assert normal_rows[50]["p0_hp"] == 350
    assert normal_rows[50]["p1_hp"] == 391
    assert normal_rows[51]["p0_hp"] == 293
    assert normal_rows[51]["p1_hp"] == 391
    assert normal_rows[52]["p0_hp"] == 243
    assert normal_rows[52]["p1_hp"] == 391
    assert normal_rows[55]["p0_hp"] == 76
    assert normal_rows[55]["p1_hp"] == 391
    assert normal_rows[56]["p0_hp"] == 20
    assert normal_rows[56]["p1_hp"] == 391
    assert normal_rows[57]["p0_hp"] == 0
    assert normal_rows[57]["p1_hp"] == 391


def test_battle29_switch_protect_and_forced_chain_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(15618),
        pool_get_team(38392),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1787264314,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 23
    assert normal_rows[11]["p0_hp"] == 242
    assert normal_rows[11]["p1_hp"] == 332
    assert normal_rows[22]["p0_action"] == "switch 3+forced:switch 3+forced:switch 2"
    assert normal_rows[22]["p0_hp"] == 141
    assert normal_rows[22]["p1_hp"] == 382
    assert normal_rows[23]["p0_hp"] == 0
    assert normal_rows[23]["p1_hp"] == 183
    assert normal_rows[23]["p1_status"] == 1


def test_battle41_slow_status_accuracy_fallback_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(23554),
        pool_get_team(42754),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1213835295,
        120,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 44
    assert normal_rows[1]["p0_hp"] == 1
    assert normal_rows[1]["p1_hp"] == 241
    assert normal_rows[2]["p0_hp"] == 301
    assert normal_rows[2]["p1_hp"] == 241
    assert normal_rows[14]["p0_hp"] == 387
    assert normal_rows[14]["p1_hp"] == 532
    assert normal_rows[15]["p0_hp"] == 307
    assert normal_rows[15]["p1_hp"] == 84
    assert normal_rows[15]["p1_status"] == 1
    assert normal_rows[44]["p0_hp"] == 114
    assert normal_rows[44]["p1_hp"] == 0


def test_battle42_multihit_noncrit_hit_does_not_inherit_first_hit_crit():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(3959),
        pool_get_team(38249),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1233945443,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 19
    assert normal_rows[6]["p0_hp"] == 11
    assert normal_rows[6]["p1_hp"] == 291
    assert normal_rows[7]["p0_hp"] == 378
    assert normal_rows[7]["p1_hp"] == 79
    assert normal_rows[8]["p0_hp"] == 378
    assert normal_rows[8]["p1_hp"] == 291
    assert normal_rows[19]["p0_hp"] == 201
    assert normal_rows[19]["p1_hp"] == 0


def test_battle44_paradox_best_stat_uses_stage_adjusted_stats():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4528),
        pool_get_team(27960),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1707061845,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 51
    assert normal_rows[4]["p0_hp"] == 132
    assert normal_rows[4]["p1_hp"] == 305
    assert normal_rows[5]["p0_hp"] == 317
    assert normal_rows[5]["p1_hp"] == 152
    assert normal_rows[6]["p0_hp"] == 166
    assert normal_rows[6]["p1_hp"] == 173
    assert normal_rows[51]["p0_hp"] == 0
    assert normal_rows[51]["p1_hp"] == 183


def test_battle45_tri_attack_secondary_frames_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(15197),
        pool_get_team(30141),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        66180795,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 43
    assert normal_rows[35]["p0_hp"] == 27
    assert normal_rows[35]["p1_hp"] == 242
    assert normal_rows[36]["p0_hp"] == 371
    assert normal_rows[36]["p1_hp"] == 184
    assert normal_rows[37]["p0_hp"] == 170
    assert normal_rows[37]["p1_hp"] == 64
    assert normal_rows[38]["p0_hp"] == 170
    assert normal_rows[38]["p1_hp"] == 160
    assert normal_rows[42]["p0_hp"] == 128
    assert normal_rows[42]["p1_hp"] == 69
    assert normal_rows[43]["p0_hp"] == 5
    assert normal_rows[43]["p1_hp"] == 0


def test_battle40_multiscale_sitrus_and_berserk_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(21980),
        pool_get_team(33420),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1405375275,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[6]["p0_hp"] == 400
    assert normal_rows[6]["p1_hp"] == 323
    assert normal_rows[7]["p0_hp"] == 504
    assert normal_rows[7]["p1_hp"] == 247
    assert normal_rows[22]["p0_hp"] == 303
    assert normal_rows[22]["p1_hp"] == 177
    assert normal_rows[23]["p0_hp"] == 177
    assert normal_rows[23]["p1_hp"] == 181
    assert normal_rows[24]["p0_hp"] == 329
    assert normal_rows[24]["p1_hp"] == 181
    assert normal_rows[25]["p0_hp"] == 354
    assert normal_rows[25]["p1_hp"] == 132
    assert normal_rows[29]["p0_hp"] == 0
    assert normal_rows[29]["p1_hp"] == 132


def test_battle47_confusion_priority_and_notarget_status_chain_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(10729),
        pool_get_team(13827),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        877308582,
        24,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[16]["p0_hp"] == 281
    assert normal_rows[16]["p1_hp"] == 33
    assert normal_rows[16]["p1_status"] == 2
    assert normal_rows[17]["p0_hp"] == 39
    assert normal_rows[17]["p1_hp"] == 196
    assert normal_rows[17]["p1_status"] == 2
    assert normal_rows[18]["p0_hp"] == 261
    assert normal_rows[18]["p1_hp"] == 176
    assert normal_rows[19]["p0_hp"] == 241
    assert normal_rows[19]["p1_hp"] == 176
    assert normal_rows[20]["p0_hp"] == 116
    assert normal_rows[20]["p1_hp"] == 136
    assert normal_rows[21]["p0_hp"] == 0
    assert normal_rows[21]["p1_hp"] == 193
    assert normal_rows[21]["p1_status"] == 2


def test_battle49_booster_energy_not_preactive_on_voluntary_switch_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(11696),
        pool_get_team(41083),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        125204184,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 15
    assert normal_rows[9]["p0_hp"] == 202
    assert normal_rows[9]["p1_hp"] == 41
    assert normal_rows[10]["p0_hp"] == 202
    assert normal_rows[10]["p1_hp"] == 41
    assert normal_rows[11]["p0_hp"] == 202
    assert normal_rows[11]["p1_hp"] == 142
    assert normal_rows[12]["p0_hp"] == 202
    assert normal_rows[12]["p1_hp"] == 41
    assert normal_rows[15]["p0_hp"] == 202
    assert normal_rows[15]["p1_hp"] == 0


def test_battle50_water_bubble_toxic_chain_and_same_turn_stat_drop_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42783),
        pool_get_team(14069),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1970642946,
        85,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 80
    assert normal_rows[16]["p0_hp"] == 36
    assert normal_rows[16]["p1_hp"] == 115
    assert normal_rows[18]["p0_hp"] == 315
    assert normal_rows[18]["p1_hp"] == 34
    assert normal_rows[18]["p1_status"] == 6
    assert normal_rows[19]["p0_hp"] == 338
    assert normal_rows[19]["p1_hp"] == 377
    assert normal_rows[20]["p0_hp"] == 151
    assert normal_rows[20]["p1_hp"] == 259
    assert normal_rows[21]["p0_hp"] == 28
    assert normal_rows[21]["p1_hp"] == 151
    assert normal_rows[22]["p0_hp"] == 36
    assert normal_rows[22]["p1_hp"] == 151
    assert normal_rows[80]["p0_hp"] == 0
    assert normal_rows[80]["p1_hp"] == 89


def test_battle51_partial_trap_primary_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(14679),
        pool_get_team(21715),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1421454891,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 18
    assert normal_rows[10]["p0_hp"] == 386
    assert normal_rows[10]["p1_hp"] == 411
    assert normal_rows[11]["p0_hp"] == 143
    assert normal_rows[11]["p1_hp"] == 411
    assert normal_rows[13]["p0_hp"] == 143
    assert normal_rows[13]["p1_hp"] == 232
    assert normal_rows[16]["p0_hp"] == 162
    assert normal_rows[16]["p1_hp"] == 315
    assert normal_rows[17]["p0_hp"] == 27
    assert normal_rows[17]["p1_hp"] == 33
    assert normal_rows[18]["p0_hp"] == 27
    assert normal_rows[18]["p1_hp"] == 0


def test_battle55_psychic_noise_heal_block_residual_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(40701),
        pool_get_team(16060),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        358571615,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 19
    assert normal_rows[8]["p0_hp"] == 82
    assert normal_rows[8]["p1_hp"] == 16
    assert normal_rows[9]["p0_hp"] == 51
    assert normal_rows[9]["p1_hp"] == 338
    assert normal_rows[12]["p0_hp"] == 387
    assert normal_rows[12]["p1_hp"] == 423
    assert normal_rows[14]["p0_hp"] == 266
    assert normal_rows[14]["p0_status"] == 1
    assert normal_rows[17]["p0_hp"] == 41
    assert normal_rows[17]["p1_hp"] == 423
    assert normal_rows[19]["p0_hp"] == 0
    assert normal_rows[19]["p1_hp"] == 423


def test_battle58_harvest_residual_roll_and_restore_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(34838),
        pool_get_team(23093),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1539871430,
        45,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 38
    assert normal_rows[34]["p0_hp"] == 359
    assert normal_rows[34]["p1_hp"] == 56
    assert normal_rows[35]["p0_hp"] == 135
    assert normal_rows[35]["p1_hp"] == 148
    assert normal_rows[36]["p0_hp"] == 56
    assert normal_rows[36]["p1_hp"] == 382
    assert normal_rows[37]["p0_hp"] == 56
    assert normal_rows[37]["p1_hp"] == 177
    assert normal_rows[38]["p0_hp"] == 0
    assert normal_rows[38]["p1_hp"] == 177


def test_battle59_charge_turn_cancellation_and_switch_reset_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(8063),
        pool_get_team(45031),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1075985461,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 20
    assert normal_rows[17]["p0_hp"] == 301
    assert normal_rows[17]["p1_hp"] == 317
    assert normal_rows[18]["p0_hp"] == 301
    assert normal_rows[18]["p1_hp"] == 227
    assert normal_rows[19]["p0_hp"] == 214
    assert normal_rows[19]["p1_hp"] == 227
    assert normal_rows[20]["p0_hp"] == 54
    assert normal_rows[20]["p1_hp"] == 0


def test_battle78_confusion_self_hit_blocks_status_preapply_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(24638),
        pool_get_team(35973),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1424421636,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 17
    assert normal_rows[4]["p0_hp"] == 272
    assert normal_rows[4]["p1_hp"] == 323
    assert normal_rows[5]["p0_hp"] == 222
    assert normal_rows[5]["p1_hp"] == 317
    assert normal_rows[6]["p0_hp"] == 41
    assert normal_rows[6]["p1_hp"] == 317
    assert normal_rows[8]["p0_hp"] == 129
    assert normal_rows[8]["p1_hp"] == 317
    assert normal_rows[10]["p0_hp"] == 222
    assert normal_rows[10]["p1_hp"] == 158
    assert normal_rows[17]["p0_hp"] == 0
    assert normal_rows[17]["p1_hp"] == 158


def test_battle79_temper_flare_previous_move_failed_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(21604),
        pool_get_team(15433),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1347135477,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 45
    assert normal_rows[8]["p0_hp"] == 132
    assert normal_rows[8]["p1_hp"] == 171
    assert normal_rows[9]["p0_hp"] == 132
    assert normal_rows[9]["p1_hp"] == 394
    assert normal_rows[10]["p0_hp"] == 132
    assert normal_rows[11]["p1_hp"] == 464
    assert normal_rows[10]["p1_hp"] == 464
    assert normal_rows[11]["p0_hp"] == 299
    assert normal_rows[11]["p1_hp"] == 464
    assert normal_rows[18]["p0_hp"] == 40
    assert normal_rows[18]["p1_hp"] == 278
    assert normal_rows[31]["p0_hp"] == 182
    assert normal_rows[31]["p1_hp"] == 342
    assert normal_rows[44]["p0_hp"] == 182
    assert normal_rows[44]["p1_hp"] == 394
    assert normal_rows[45]["p0_hp"] == 0
    assert normal_rows[45]["p1_hp"] == 380


def test_battle60_showdown_runner_drops_stale_switch_when_trapped():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    team0 = pool_get_team(46902)
    team1 = pool_get_team(7615)
    battle_seed = 1068008206
    seed_tuple = (battle_seed & 0xFFFF, (battle_seed >> 16) & 0xFFFF, 0, 0)

    _, p0_actions, p1_actions = run_pokepy_battle(
        team0,
        team1,
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        battle_seed,
        80,
    )

    show_rows, raw_data, meta = run_showdown(
        seed_tuple,
        team_to_showdown_packed(team0, mappings),
        team_to_showdown_packed(team1, mappings),
        p0_actions,
        p1_actions,
        80,
        timeout_s=120,
    )

    assert meta["returncode"] == 0
    assert not meta["timeout"]
    assert meta["error"] is None
    assert raw_data is not None
    assert show_rows


def test_battle60_source_switch_trap_cleanup_and_lockedmove_immunity_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(46902),
        pool_get_team(7615),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1068008206,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 55
    assert normal_rows[3]["p0_hp"] == 211
    assert normal_rows[3]["p1_hp"] == 195
    assert normal_rows[4]["p0_hp"] == 211
    assert normal_rows[4]["p1_hp"] == 341
    assert normal_rows[5]["p0_hp"] == 326
    assert normal_rows[5]["p1_hp"] == 341
    assert normal_rows[47]["p0_hp"] == 323
    assert normal_rows[47]["p1_hp"] == 279
    assert normal_rows[48]["p0_hp"] == 326
    assert normal_rows[48]["p1_hp"] == 279
    assert normal_rows[49]["p0_hp"] == 350
    assert normal_rows[49]["p1_hp"] == 267
    assert normal_rows[50]["p0_hp"] == 374
    assert normal_rows[50]["p1_hp"] == 23
    assert normal_rows[51]["p0_hp"] == 386
    assert normal_rows[51]["p1_hp"] == 280
    assert normal_rows[55]["p0_hp"] == 386
    assert normal_rows[55]["p1_hp"] == 0


def test_battle93_lead_booster_energy_preserves_encoded_best_stat_regression():
    from pokepy.core.constants import (
        OFF_META,
        OFF_SIDE0,
        POKEMON_SIZE,
        M_ACTIVE0,
        FLAG_BOOSTER_ENERGY_ACTIVE,
    )
    from pokepy.env import init_battle_state

    gd = load_game_data()

    state = init_battle_state(
        pool_get_team(4136), pool_get_team(13792), gd, seed=1045090807
    )

    active0 = int(state.battle_state[OFF_META + M_ACTIVE0])
    lead0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
    lead0_flags = int(state.battle_state[lead0_off + 15])

    assert lead0_flags & FLAG_BOOSTER_ENERGY_ACTIVE
    assert (lead0_flags & 0x6010) == 0x2010


def test_battle93_simultaneous_lead_booster_energy_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4136),
        pool_get_team(13792),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1045090807,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 47
    assert normal_rows[1]["p0_hp"] == 110
    assert normal_rows[1]["p1_hp"] == 289
    assert normal_rows[2]["p0_hp"] == 110
    assert normal_rows[2]["p1_hp"] == 85
    assert normal_rows[3]["p0_hp"] == 110
    assert normal_rows[3]["p1_hp"] == 304
    assert normal_rows[47]["p0_hp"] == 290
    assert normal_rows[47]["p1_hp"] == 0


def test_battle62_fixed_maxhp_recoil_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(8310),
        pool_get_team(19051),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        512746651,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 28
    assert normal_rows[6]["p0_hp"] == 159
    assert normal_rows[6]["p1_hp"] == 130
    assert normal_rows[11]["p0_hp"] == 404
    assert normal_rows[11]["p1_hp"] == 12
    assert normal_rows[13]["p0_hp"] == 78
    assert normal_rows[13]["p1_hp"] == 4
    assert normal_rows[28]["p0_hp"] == 0
    assert normal_rows[28]["p1_hp"] == 209


def test_battle63_self_ko_does_not_false_trigger_moxie_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(15075),
        pool_get_team(34185),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1353521562,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 35
    assert normal_rows[27]["p0_hp"] == 364
    assert normal_rows[27]["p1_hp"] == 150
    assert normal_rows[28]["p0_hp"] == 125
    assert normal_rows[28]["p1_hp"] == 169
    assert normal_rows[34]["p0_hp"] == 97
    assert normal_rows[34]["p1_hp"] == 85
    assert normal_rows[35]["p0_hp"] == 0
    assert normal_rows[35]["p1_hp"] == 85


def test_battle64_deferred_protect_and_failed_roost_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(30359),
        pool_get_team(18090),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2060924130,
        140,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 120
    assert normal_rows[10]["p0_hp"] == 384
    assert normal_rows[10]["p1_hp"] == 114
    assert normal_rows[11]["p0_hp"] == 384
    assert normal_rows[11]["p1_hp"] == 158
    assert normal_rows[12]["p0_hp"] == 384
    assert normal_rows[12]["p1_hp"] == 202
    assert normal_rows[49]["p0_hp"] == 384
    assert normal_rows[49]["p1_hp"] == 354
    assert normal_rows[50]["p0_hp"] == 384
    assert normal_rows[50]["p1_hp"] == 354
    assert normal_rows[51]["p0_hp"] == 384
    assert normal_rows[51]["p1_hp"] == 354
    assert normal_rows[52]["p0_hp"] == 384
    assert normal_rows[52]["p1_hp"] == 354
    assert normal_rows[120]["p0_hp"] == 0
    assert normal_rows[120]["p1_hp"] == 218


def test_battle65_phazing_switchin_timing_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4382),
        pool_get_team(17129),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        253415745,
        100,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 88
    assert normal_rows[70]["p0_hp"] == 329
    assert normal_rows[70]["p1_hp"] == 66
    assert normal_rows[71]["p0_hp"] == 265
    assert normal_rows[71]["p1_hp"] == 514
    assert normal_rows[72]["p0_hp"] == 195
    assert normal_rows[72]["p1_hp"] == 58
    assert normal_rows[73]["p0_hp"] == 119
    assert normal_rows[73]["p1_hp"] == 412
    assert normal_rows[88]["p0_hp"] == 0
    assert normal_rows[88]["p1_hp"] == 514


def test_battle67_toxic_reentry_and_terminal_switch_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(45429),
        pool_get_team(24753),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1502609628,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 24
    assert normal_rows[20]["p0_hp"] == 224
    assert normal_rows[20]["p1_hp"] == 19
    assert normal_rows[21]["p0_hp"] == 193
    assert normal_rows[21]["p1_hp"] == 278
    assert normal_rows[22]["p0_hp"] == 131
    assert normal_rows[22]["p1_hp"] == 95
    assert normal_rows[23]["p0_hp"] == 38
    assert normal_rows[23]["p1_hp"] == 83
    assert normal_rows[24]["p0_hp"] == 0
    assert normal_rows[24]["p1_hp"] == 0


def test_battle69_future_sight_offfield_boost_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(48458),
        pool_get_team(13176),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1672354832,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 57
    assert normal_rows[53]["p0_hp"] == 108
    assert normal_rows[53]["p1_hp"] == 339
    assert normal_rows[54]["p0_hp"] == 41
    assert normal_rows[54]["p1_hp"] == 363
    assert normal_rows[55]["p0_hp"] == 281
    assert normal_rows[55]["p1_hp"] == 230
    assert normal_rows[56]["p0_hp"] == 104
    assert normal_rows[56]["p1_hp"] == 79
    assert normal_rows[57]["p0_hp"] == 76
    assert normal_rows[57]["p1_hp"] == 0


def test_battle71_eject_button_cancels_pivot_selfswitch_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, p0_actions, _ = run_pokepy_battle(
        pool_get_team(22468),
        pool_get_team(36845),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        584634302,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 21
    assert normal_rows[5]["p0_hp"] == 321
    assert normal_rows[5]["p1_hp"] == 219
    assert normal_rows[5]["p0_active"] == 3
    assert normal_rows[6]["p0_hp"] == 370
    assert normal_rows[6]["p1_hp"] == 183
    assert normal_rows[6]["p0_active"] == 5
    assert normal_rows[7]["p0_hp"] == 370
    assert normal_rows[7]["p1_hp"] == 162
    assert normal_rows[21]["p0_hp"] == 178
    assert normal_rows[21]["p1_hp"] == 0
    assert p0_actions[:7] == [
        "move 4",
        "move 4",
        "move 4",
        "switch 4",
        "move 4",
        "move 4",
        "switch 6",
    ]


def test_battle72_throat_chop_secondary_frame_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(3962),
        pool_get_team(4819),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        959976128,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 36
    assert normal_rows[15]["p0_hp"] == 513
    assert normal_rows[15]["p1_hp"] == 217
    assert normal_rows[16]["p0_hp"] == 513
    assert normal_rows[16]["p1_hp"] == 351
    assert normal_rows[17]["p0_hp"] == 513
    assert normal_rows[17]["p1_hp"] == 301
    assert normal_rows[18]["p0_hp"] == 513
    assert normal_rows[18]["p1_hp"] == 190
    assert normal_rows[19]["p0_hp"] == 513
    assert normal_rows[19]["p1_hp"] == 85
    assert normal_rows[20]["p0_hp"] == 513
    assert normal_rows[20]["p1_hp"] == 321
    assert normal_rows[36]["p0_hp"] == 318
    assert normal_rows[36]["p1_hp"] == 0


def test_battle73_neutralizing_gas_and_switch_absorb_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(45130),
        pool_get_team(6378),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        978772129,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 61
    assert normal_rows[1]["p0_hp"] == 187
    assert normal_rows[10]["p0_hp"] == 403
    assert normal_rows[52]["p0_hp"] == 304
    assert normal_rows[52]["p1_hp"] == 158
    assert normal_rows[53]["p0_hp"] == 87
    assert normal_rows[53]["p1_hp"] == 43
    assert normal_rows[54]["p0_hp"] == 253
    assert normal_rows[54]["p1_hp"] == 61
    assert normal_rows[55]["p0_hp"] == 270
    assert normal_rows[55]["p1_hp"] == 158
    assert normal_rows[57]["p0_hp"] == 394
    assert normal_rows[57]["p1_hp"] == 136
    assert normal_rows[58]["p0_hp"] == 345
    assert normal_rows[58]["p1_hp"] == 136
    assert normal_rows[61]["p0_hp"] == 0
    assert normal_rows[61]["p1_hp"] == 4


def test_battle74_pivot_first_impression_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(35201),
        pool_get_team(10118),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1549659416,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 27
    assert normal_rows[14]["p0_hp"] == 107
    assert normal_rows[14]["p1_hp"] == 284
    assert normal_rows[14]["p1_active"] == 2
    assert normal_rows[15]["p0_hp"] == 362
    assert normal_rows[15]["p0_max_hp"] == 362
    assert normal_rows[15]["p0_active"] == 3
    assert normal_rows[15]["p1_hp"] == 284
    assert normal_rows[16]["p0_hp"] == 362
    assert normal_rows[16]["p1_hp"] == 288
    assert normal_rows[17]["p0_hp"] == 186
    assert normal_rows[17]["p0_status"] == 2
    assert normal_rows[17]["p1_hp"] == 288
    assert normal_rows[27]["p0_hp"] == 0
    assert normal_rows[27]["p1_hp"] == 173


def test_battle84_fixed_maxhp_recoil_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(9411),
        pool_get_team(16493),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1418625644,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 19
    assert normal_rows[3]["p0_hp"] == 51
    assert normal_rows[3]["p1_hp"] == 144
    assert normal_rows[19]["p0_hp"] == 0
    assert normal_rows[19]["p1_hp"] == 67


def test_choice_lock_mask_keeps_zero_pp_locked_move_selected():
    from pokepy.core.constants import (
        F_CHOICE_LOCK_0,
        OFF_SIDE0,
        OFF_SIDE1,
        M_ACTIVE0,
        OFF_FIELD,
        OFF_META,
        PHASE_FORCED_SWITCH,
        POKEMON_SIZE,
    )
    from pokepy.engine.battle_gen9 import step_battle_gen9, step_forced_switch
    from pokepy.env import init_battle_state
    from pokepy.utils.gen5_prng import Gen5PRNG

    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    t0 = 4297
    t1 = 34868
    battle_seed = 432652533

    state = init_battle_state(
        pool_get_team(t0), pool_get_team(t1), gd, seed=battle_seed
    )
    prng = Gen5PRNG((battle_seed & 0xFFFF, (battle_seed >> 16) & 0xFFFF, 0, 0))

    battle_arr = state.battle_state
    for slot in range(6):
        spe0 = int(battle_arr[OFF_SIDE0 + slot * POKEMON_SIZE + 11])
        spe1 = int(battle_arr[OFF_SIDE1 + slot * POKEMON_SIZE + 11])
        if spe0 == spe1:
            prng.random(0, 2)
    if int(battle_arr[OFF_SIDE0 + 11]) == int(battle_arr[OFF_SIDE1 + 11]):
        prng.random(0, 2)
        prng.random(0, 2)
        prng.random(0, 2)

    def run_forced_switch_chain():
        while int(state.phase) == PHASE_FORCED_SWITCH:
            a0 = simple_heuristic_forced_switch(
                state, 0, gd, mappings, MODERN_TYPE_CHART
            )
            step_forced_switch(
                state,
                a0,
                side=0,
                game_data=gd,
                move_effects=me,
                type_chart=MODERN_TYPE_CHART,
                gen5_prng=prng,
            )

    for _ in range(62):
        if int(state.phase) == PHASE_FORCED_SWITCH:
            run_forced_switch_chain()
        else:
            a0 = simple_heuristic_action(state, 0, gd, mappings, MODERN_TYPE_CHART)
            a1 = simple_heuristic_action(state, 1, gd, mappings, MODERN_TYPE_CHART)
            _, _, done = step_battle_gen9(
                state, a0, a1, gd, me, MODERN_TYPE_CHART, prng
            )
            if not done and int(state.phase) == PHASE_FORCED_SWITCH:
                run_forced_switch_chain()

    active0 = int(state.battle_state[OFF_META + M_ACTIVE0])
    assert active0 == 1
    assert int(state.battle_state[OFF_FIELD + F_CHOICE_LOCK_0]) == 2
    assert int(state.team_pp[active0, 2]) == 0

    mask = get_battle_action_mask(state, 0, gd)
    assert mask.tolist()[:4] == [False, False, True, False]


def test_battle76_mid_turn_pivot_resyncs_post_hit_target_regression(monkeypatch):
    from parity_heuristic_e2e import _deterministic_auto_switch
    from pokepy.core.constants import (
        ITEM_LEFTOVERS,
        M_ACTIVE1,
        OFF_META,
        OFF_SIDE0,
        OFF_SIDE1,
        PHASE_FORCED_SWITCH,
        POKEMON_SIZE,
    )
    from pokepy.engine.battle_gen9 import step_battle_gen9, step_forced_switch
    from pokepy.env import init_battle_state
    from pokepy.utils.gen5_prng import Gen5PRNG
    import pokepy.effects as fx

    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    battle_seed = 994727115
    state = init_battle_state(
        pool_get_team(27345), pool_get_team(8838), gd, seed=battle_seed
    )
    prng = Gen5PRNG((battle_seed & 0xFFFF, (battle_seed >> 16) & 0xFFFF, 0, 0))

    monkeypatch.setattr(fx, "auto_switch", _deterministic_auto_switch)

    battle_arr = state.battle_state
    for slot in range(6):
        spe0 = int(battle_arr[OFF_SIDE0 + slot * POKEMON_SIZE + 11])
        spe1 = int(battle_arr[OFF_SIDE1 + slot * POKEMON_SIZE + 11])
        if spe0 == spe1:
            prng.random(0, 2)
    if int(battle_arr[OFF_SIDE0 + 11]) == int(battle_arr[OFF_SIDE1 + 11]):
        prng.random(0, 2)
        prng.random(0, 2)
        prng.random(0, 2)

    def run_forced_switch_chain():
        while int(state.phase) == PHASE_FORCED_SWITCH:
            a0 = simple_heuristic_forced_switch(
                state, 0, gd, mappings, MODERN_TYPE_CHART
            )
            step_forced_switch(
                state,
                a0,
                side=0,
                game_data=gd,
                move_effects=me,
                type_chart=MODERN_TYPE_CHART,
                gen5_prng=prng,
            )

    saw_low_hp_corviknight = False
    for _ in range(16):
        if int(state.phase) == PHASE_FORCED_SWITCH:
            run_forced_switch_chain()
        else:
            a0 = simple_heuristic_action(state, 0, gd, mappings, MODERN_TYPE_CHART)
            a1 = simple_heuristic_action(state, 1, gd, mappings, MODERN_TYPE_CHART)
            _, _, done = step_battle_gen9(
                state,
                a0,
                a1,
                gd,
                me,
                MODERN_TYPE_CHART,
                prng,
                resolve_mid_turn_switch0=lambda st: simple_heuristic_forced_switch(
                    st, 0, gd, mappings, MODERN_TYPE_CHART
                ),
            )
            if not done and int(state.phase) == PHASE_FORCED_SWITCH:
                run_forced_switch_chain()

        corviknight_off = OFF_SIDE1 + 5 * POKEMON_SIZE
        if int(state.battle_state[corviknight_off + 1]) == 44:
            saw_low_hp_corviknight = True
        if (
            saw_low_hp_corviknight
            and int(state.battle_state[OFF_META + M_ACTIVE1]) == 0
        ):
            break

    corviknight_off = OFF_SIDE1 + 5 * POKEMON_SIZE
    assert saw_low_hp_corviknight
    assert int(state.battle_state[OFF_META + M_ACTIVE1]) == 0
    assert int(state.battle_state[corviknight_off + 1]) == 44
    assert int(state.battle_state[corviknight_off + 6]) == ITEM_LEFTOVERS


def test_battle88_disguise_bust_does_not_feed_recoil_regression(monkeypatch):
    from parity_heuristic_e2e import _deterministic_auto_switch
    from pokepy.core.constants import (
        M_ACTIVE0,
        M_ACTIVE1,
        OFF_META,
        OFF_SIDE0,
        OFF_SIDE1,
        PHASE_FORCED_SWITCH,
        POKEMON_SIZE,
    )
    from pokepy.engine.battle_gen9 import step_battle_gen9, step_forced_switch
    from pokepy.env import init_battle_state
    from pokepy.utils.gen5_prng import Gen5PRNG
    import pokepy.effects as fx

    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    battle_seed = 960007121
    state = init_battle_state(
        pool_get_team(49484), pool_get_team(29053), gd, seed=battle_seed
    )
    prng = Gen5PRNG((battle_seed & 0xFFFF, (battle_seed >> 16) & 0xFFFF, 0, 0))

    monkeypatch.setattr(fx, "auto_switch", _deterministic_auto_switch)

    battle_arr = state.battle_state
    for slot in range(6):
        spe0 = int(battle_arr[OFF_SIDE0 + slot * POKEMON_SIZE + 11])
        spe1 = int(battle_arr[OFF_SIDE1 + slot * POKEMON_SIZE + 11])
        if spe0 == spe1:
            prng.random(0, 2)
    if int(battle_arr[OFF_SIDE0 + 11]) == int(battle_arr[OFF_SIDE1 + 11]):
        prng.random(0, 2)
        prng.random(0, 2)
        prng.random(0, 2)

    def run_forced_switch_chain():
        while int(state.phase) == PHASE_FORCED_SWITCH:
            a0 = simple_heuristic_forced_switch(
                state, 0, gd, mappings, MODERN_TYPE_CHART
            )
            step_forced_switch(
                state,
                a0,
                side=0,
                game_data=gd,
                move_effects=me,
                type_chart=MODERN_TYPE_CHART,
                gen5_prng=prng,
            )

    for _ in range(27):
        if int(state.phase) == PHASE_FORCED_SWITCH:
            run_forced_switch_chain()
        else:
            a0 = simple_heuristic_action(state, 0, gd, mappings, MODERN_TYPE_CHART)
            a1 = simple_heuristic_action(state, 1, gd, mappings, MODERN_TYPE_CHART)
            _, _, done = step_battle_gen9(
                state,
                a0,
                a1,
                gd,
                me,
                MODERN_TYPE_CHART,
                prng,
                resolve_mid_turn_switch0=lambda st: simple_heuristic_forced_switch(
                    st, 0, gd, mappings, MODERN_TYPE_CHART
                ),
            )
            if not done and int(state.phase) == PHASE_FORCED_SWITCH:
                run_forced_switch_chain()

    active0 = int(state.battle_state[OFF_META + M_ACTIVE0])
    active1 = int(state.battle_state[OFF_META + M_ACTIVE1])
    assert active0 == 3
    assert active1 == 3
    assert int(state.battle_state[OFF_SIDE0 + active0 * POKEMON_SIZE + 1]) == 106
    assert int(state.battle_state[OFF_SIDE1 + active1 * POKEMON_SIZE + 1]) == 18


def test_battle124_rest_sleep_talk_and_wake_turn_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(20510),
        pool_get_team(18682),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1903305990,
        85,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[40]["p0_hp"] == 471
    assert normal_rows[40]["p0_status"] == 3
    assert normal_rows[40]["p1_hp"] == 225

    assert normal_rows[61]["p0_hp"] == 489
    assert normal_rows[61]["p0_status"] == 3
    assert normal_rows[61]["p1_active"] == 4
    assert normal_rows[61]["p1_hp"] == 225

    assert normal_rows[62]["p0_hp"] == 474
    assert normal_rows[62]["p0_status"] == 3
    assert normal_rows[62]["p1_active"] == 2
    assert normal_rows[62]["p1_hp"] == 435
    assert normal_rows[62]["p1_status"] == 1

    assert normal_rows[63]["p0_hp"] == 474
    assert normal_rows[63]["p0_status"] == 0
    assert normal_rows[63]["p1_active"] == 3
    assert normal_rows[63]["p1_hp"] == 302

    assert normal_rows[77]["p0_hp"] == 348
    assert normal_rows[77]["p1_active"] == 3
    assert normal_rows[77]["p1_hp"] == 138

    assert normal_rows[78]["p0_hp"] == 222
    assert normal_rows[78]["p1_active"] == 2
    assert normal_rows[78]["p1_hp"] == 67
    assert normal_rows[78]["p1_status"] == 1


def test_battle202_eject_pack_mixed_boosts_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(16154),
        pool_get_team(25711),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2076581196,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert "forced:switch" in normal_rows[5]["p1_action"]
    assert normal_rows[5]["p1_active"] == 3
    assert normal_rows[5]["p1_hp"] == 264
    assert normal_rows[5]["p1_max_hp"] == 301
    assert normal_rows[5]["p1_status"] == 5


def test_battle203_red_card_single_choice_drag_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42878),
        pool_get_team(41684),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        993854069,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[54]["p0_active"] == 4
    assert normal_rows[54]["p0_hp"] == 157
    assert normal_rows[54]["p0_max_hp"] == 301
    assert normal_rows[54]["p0_status"] == 5
    assert normal_rows[54]["p1_hp"] == 22

    assert "forced:switch" in normal_rows[55]["p0_action"]
    assert normal_rows[55]["p0_active"] == 4
    assert normal_rows[55]["p0_hp"] == 157
    assert normal_rows[55]["p0_max_hp"] == 301
    assert normal_rows[55]["p0_status"] == 5
    assert normal_rows[55]["p1_hp"] == 22


def test_battle204_future_sight_active_source_item_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(39206),
        pool_get_team(19254),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1473529150,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert len(normal_rows) == 45
    assert normal_rows[23]["p0_hp"] == 393
    assert normal_rows[23]["p1_hp"] == 399
    assert normal_rows[24]["p0_hp"] == 393
    assert normal_rows[24]["p1_hp"] == 399
    assert normal_rows[25]["p0_hp"] == 393
    assert normal_rows[25]["p1_hp"] == 277
    assert normal_rows[26]["p0_hp"] == 393
    assert normal_rows[26]["p1_hp"] == 277
    assert normal_rows[28]["p0_hp"] == 393
    assert normal_rows[28]["p1_hp"] == 148


def test_battle206_tied_double_switch_update_frame_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(3056),
        pool_get_team(6988),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        931188965,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[17]["p0_hp"] == 114
    assert normal_rows[17]["p1_hp"] == 264
    assert normal_rows[18]["p0_hp"] == 317
    assert normal_rows[18]["p1_hp"] == 360
    assert normal_rows[19]["p0_hp"] == 317
    assert normal_rows[19]["p1_hp"] == 45
    assert normal_rows[20]["p0_hp"] == 317
    assert normal_rows[20]["p1_hp"] == 227
    assert normal_rows[21]["p0_hp"] == 137
    assert normal_rows[21]["p1_hp"] == 227


def test_battle210_white_herb_immediate_after_move_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36847),
        pool_get_team(16374),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        931786189,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[23]["p0_hp"] == 234
    assert normal_rows[23]["p1_hp"] == 261
    assert normal_rows[24]["p0_hp"] == 266
    assert normal_rows[24]["p1_hp"] == 80
    assert normal_rows[24]["p1_max_hp"] == 261
    assert normal_rows[25]["p0_hp"] == 88
    assert normal_rows[25]["p1_hp"] == 17
    assert normal_rows[25]["p1_max_hp"] == 321


def test_battle213_light_clay_reflect_duration_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(47442),
        pool_get_team(33542),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1967842387,
        100,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[49]["p0_hp"] == 295
    assert normal_rows[49]["p1_hp"] == 341
    assert normal_rows[50]["p0_hp"] == 226
    assert normal_rows[50]["p1_hp"] == 299
    assert normal_rows[51]["p0_hp"] == 134
    assert normal_rows[51]["p1_hp"] == 193
    assert normal_rows[52]["p0_hp"] == 41
    assert normal_rows[52]["p1_hp"] == 91


def test_battle214_toxic_chain_uturn_preroll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42709),
        pool_get_team(24045),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1654185559,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[22]["p0_hp"] == 301
    assert normal_rows[22]["p0_status"] == 0
    assert normal_rows[22]["p1_hp"] == 317
    assert normal_rows[23]["p0_hp"] == 175
    assert normal_rows[23]["p0_status"] == 6
    assert normal_rows[23]["p1_hp"] == 434
    assert normal_rows[24]["p0_hp"] == 102
    assert normal_rows[24]["p0_status"] == 6
    assert normal_rows[24]["p1_hp"] == 434
    assert normal_rows[25]["p0_hp"] == 360
    assert normal_rows[25]["p1_hp"] == 44
    assert normal_rows[26]["p0_hp"] == 112
    assert normal_rows[26]["p1_hp"] == 173
    assert normal_rows[27]["p0_hp"] == 0
    assert normal_rows[27]["p1_hp"] == 55


def test_battle215_triple_arrows_dual_secondary_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(16418),
        pool_get_team(23024),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1149837455,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[42]["p0_hp"] == 380
    assert normal_rows[42]["p1_hp"] == 66
    assert normal_rows[43]["p0_hp"] == 3
    assert normal_rows[43]["p1_hp"] == 319
    assert normal_rows[44]["p0_hp"] == 303
    assert normal_rows[44]["p0_max_hp"] == 303
    assert normal_rows[44]["p1_hp"] == 123


def test_battle217_future_sight_focus_sash_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(32629),
        pool_get_team(10117),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1727418297,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[28]["p0_hp"] == 307
    assert normal_rows[28]["p1_hp"] == 394
    assert normal_rows[29]["p0_hp"] == 1
    assert normal_rows[29]["p0_max_hp"] == 307
    assert normal_rows[29]["p1_hp"] == 192
    assert normal_rows[30]["p0_hp"] == 1
    assert normal_rows[30]["p0_max_hp"] == 307
    assert normal_rows[30]["p1_hp"] == 209
    assert normal_rows[31]["p0_hp"] == 1
    assert normal_rows[31]["p1_hp"] == 0


def test_battle222_grassy_terrain_skips_phantom_force_charge_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(25165),
        pool_get_team(43035),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1973408810,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert "forced:switch" in normal_rows[19]["p1_action"]
    assert normal_rows[19]["p1_active"] == 3
    assert normal_rows[19]["p1_hp"] == 381

    assert normal_rows[20]["p0_hp"] == 329
    assert normal_rows[20]["p1_action"] == "move 4"
    assert normal_rows[20]["p1_hp"] == 139
    assert normal_rows[20]["p1_max_hp"] == 381

    assert normal_rows[21]["p0_hp"] == 100
    assert normal_rows[21]["p1_action"] == "move 4"
    assert normal_rows[21]["p1_hp"] == 139


def test_battle224_hurricane_confusion_focus_sash_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(31810),
        pool_get_team(4754),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1335459691,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[14]["p0_active"] == 2
    assert normal_rows[14]["p0_hp"] == 51
    assert normal_rows[14]["p1_hp"] == 38

    assert "forced:switch" in normal_rows[15]["p1_action"]
    assert normal_rows[15]["p1_active"] == 2
    assert normal_rows[15]["p1_hp"] == 302

    assert normal_rows[16]["p0_hp"] == 51
    assert normal_rows[16]["p1_action"] == "move 4"
    assert normal_rows[16]["p1_hp"] == 0
    assert normal_rows[16]["p1_max_hp"] == 302


def test_battle225_throat_chop_choice_lock_disablemove_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36285),
        pool_get_team(29039),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        181447805,
        35,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[26]["p0_active"] == 4
    assert normal_rows[26]["p0_action"] == "move 4"
    assert normal_rows[26]["p0_hp"] == 255
    assert normal_rows[26]["p1_hp"] == 294

    assert normal_rows[27]["p0_hp"] == 137
    assert normal_rows[27]["p1_hp"] == 326

    assert normal_rows[28]["p0_hp"] == 29
    assert normal_rows[28]["p1_hp"] == 358

    assert "forced:switch" in normal_rows[29]["p0_action"]
    assert normal_rows[29]["p0_hp"] == 226
    assert normal_rows[29]["p1_hp"] == 390


def test_battle230_slower_secondary_uses_projected_target_hp_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49832),
        pool_get_team(39755),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2000104913,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[17]["p0_action"] == "move 4+forced:switch 6"
    assert normal_rows[17]["p0_hp"] == 246
    assert normal_rows[17]["p1_hp"] == 164

    assert normal_rows[18]["p0_action"] == "switch 4+forced:switch 4"
    assert normal_rows[18]["p0_hp"] == 246
    assert normal_rows[18]["p1_hp"] == 164

    assert normal_rows[19]["p0_action"] == "move 2"
    assert normal_rows[19]["p0_hp"] == 46
    assert normal_rows[19]["p1_action"] == "move 1"
    assert normal_rows[19]["p1_hp"] == 20

    assert normal_rows[20]["p0_hp"] == 46
    assert normal_rows[20]["p1_action"] == "move 1+forced:switch 3"
    assert normal_rows[20]["p1_active"] == 2
    assert normal_rows[20]["p1_hp"] == 394


def test_battle232_eject_button_forced_switch_skips_update_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, p0_actions, _ = run_pokepy_battle(
        pool_get_team(36092),
        pool_get_team(29503),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        107688083,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[10]["p0_action"] == "move 3+forced:switch 4"
    assert normal_rows[10]["p0_active"] == 3
    assert normal_rows[10]["p0_hp"] == 363
    assert normal_rows[10]["p1_hp"] == 532

    assert normal_rows[11]["p0_action"] == "move 4"
    assert normal_rows[11]["p0_hp"] == 343
    assert normal_rows[11]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[11]["p1_active"] == 3
    assert normal_rows[11]["p1_hp"] == 315

    assert normal_rows[12]["p0_action"] == "switch 3"
    assert normal_rows[12]["p0_active"] == 2
    assert normal_rows[12]["p0_hp"] == 210
    assert normal_rows[12]["p1_hp"] == 315

    assert p0_actions[10:13] == [
        "move 3",
        "switch 4",
        "move 4",
    ]


def test_battle233_player_eject_button_switch_completes_before_residual_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, p0_actions, _ = run_pokepy_battle(
        pool_get_team(4752),
        pool_get_team(26027),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1323205765,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[7]["p0_action"] == "move 4"
    assert normal_rows[7]["p0_active"] == 4
    assert normal_rows[7]["p0_hp"] == 321
    assert normal_rows[7]["p1_action"] == "switch 2"
    assert normal_rows[7]["p1_active"] == 1
    assert normal_rows[7]["p1_hp"] == 323

    assert normal_rows[8]["p0_action"] == "move 1+pivot:switch 2+forced:switch 2"
    assert normal_rows[8]["p0_active"] == 4
    assert normal_rows[8]["p0_hp"] == 321
    assert normal_rows[8]["p1_action"] == "move 3"
    assert normal_rows[8]["p1_active"] == 1
    assert normal_rows[8]["p1_hp"] == 43

    assert p0_actions[8:11] == [
        "move 1",
        "switch 2",
        "switch 2",
    ]

    assert normal_rows[9]["p0_action"] == "move 1+pivot:switch 2"
    assert normal_rows[9]["p0_active"] == 1
    assert normal_rows[9]["p0_hp"] == 92
    assert normal_rows[9]["p1_action"] == "move 3+forced:switch 6"
    assert normal_rows[9]["p1_active"] == 0
    assert normal_rows[9]["p1_hp"] == 89


def test_battle77_midturn_pivot_and_item_switch_queue_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, p0_actions, _ = run_pokepy_battle(
        pool_get_team(42830),
        pool_get_team(976),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1628908287,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[6]["p0_action"] == "move 1+pivot:switch 3+forced:switch 3"
    assert normal_rows[6]["p0_active"] == 3
    assert normal_rows[6]["p0_hp"] == 281
    assert normal_rows[6]["p1_action"] == "move 1"
    assert normal_rows[6]["p1_active"] == 1
    assert normal_rows[6]["p1_hp"] == 204

    assert p0_actions[7:10] == [
        "move 1",
        "switch 3",
        "switch 3",
    ]


def test_battle24_voluntary_switch_then_inline_forced_switch_queue_rebase_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, p0_actions, _ = run_pokepy_battle(
        pool_get_team(23147),
        pool_get_team(6496),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1474200732,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[32]["p0_action"] == "switch 5+forced:switch 2"
    assert normal_rows[32]["p0_active"] == 4
    assert normal_rows[32]["p0_hp"] == 190
    assert normal_rows[32]["p1_action"] == "move 3"
    assert normal_rows[32]["p1_active"] == 1
    assert normal_rows[32]["p1_hp"] == 277

    assert p0_actions[34:36] == [
        "switch 5",
        "switch 2",
    ]

    assert normal_rows[39]["p0_action"] == "move 1+forced:switch 2"
    assert normal_rows[39]["p0_active"] == 2
    assert normal_rows[39]["p0_hp"] == 391
    assert normal_rows[39]["p1_action"] == "move 2"
    assert normal_rows[39]["p1_active"] == 4
    assert normal_rows[39]["p1_hp"] == 303


def test_battle242_ceaseless_edge_second_layer_survives_faster_user_faint_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36061),
        pool_get_team(4385),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1048387382,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[9]["p0_action"] == "move 3+forced:switch 5"
    assert normal_rows[9]["p0_active"] == 4
    assert normal_rows[9]["p0_hp"] == 235
    assert normal_rows[9]["p1_action"] == "move 2"
    assert normal_rows[9]["p1_active"] == 1
    assert normal_rows[9]["p1_hp"] == 60

    assert normal_rows[12]["p0_action"] == "move 1+forced:switch 4"
    assert normal_rows[12]["p0_active"] == 3
    assert normal_rows[12]["p0_hp"] == 299
    assert normal_rows[12]["p1_action"] == "move 2"
    assert normal_rows[12]["p1_active"] == 2
    assert normal_rows[12]["p1_hp"] == 243


def test_battle250_switch_weather_change_and_weather_residual_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(48064),
        pool_get_team(7389),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        185239393,
        12,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[3]["p0_action"] == "switch 2"
    assert normal_rows[3]["p0_active"] == 1
    assert normal_rows[3]["p0_hp"] == 124
    assert normal_rows[3]["p0_max_hp"] == 349
    assert normal_rows[3]["p1_action"] == "move 1"
    assert normal_rows[3]["p1_hp"] == 339

    assert normal_rows[4]["p0_action"] == "move 3"
    assert normal_rows[4]["p0_hp"] == 124
    assert normal_rows[4]["p1_action"] == "switch 6"
    assert normal_rows[4]["p1_active"] == 0
    assert normal_rows[4]["p1_hp"] == 256

    assert normal_rows[7]["p0_action"] == "switch 4"
    assert normal_rows[7]["p0_active"] == 3
    assert normal_rows[7]["p0_hp"] == 101
    assert normal_rows[7]["p1_action"] == "move 1"
    assert normal_rows[7]["p1_hp"] == 339


def test_battle251_dragon_energy_hp_scaled_power_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(46806),
        pool_get_team(27750),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        940391759,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[23]["p0_hp"] == 317
    assert normal_rows[23]["p1_hp"] == 340
    assert normal_rows[24]["p0_action"] == "switch 6+forced:switch 3"
    assert normal_rows[24]["p0_hp"] == 404
    assert normal_rows[24]["p1_hp"] == 340
    assert normal_rows[25]["p0_action"] == "move 4"
    assert normal_rows[25]["p0_hp"] == 236
    assert normal_rows[25]["p1_action"] == "move 1"
    assert normal_rows[26]["p0_hp"] == 72
    assert normal_rows[26]["p1_hp"] == 340


def test_battle253_side1_pivot_contact_preroll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36484),
        pool_get_team(8799),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1187543565,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[22]["p0_hp"] == 112
    assert normal_rows[22]["p1_hp"] == 241
    assert normal_rows[23]["p0_hp"] == 43
    assert normal_rows[23]["p1_action"] == "move 3+forced:switch 6+die_switch:switch 6"
    assert normal_rows[23]["p1_hp"] == 194
    assert normal_rows[24]["p0_action"] == "move 3+forced:switch 3"
    assert normal_rows[24]["p0_hp"] == 421
    assert normal_rows[24]["p1_hp"] == 38
    assert normal_rows[24]["p1_status"] == 0
    assert normal_rows[25]["p0_hp"] == 91
    assert normal_rows[25]["p1_hp"] == 0


def test_battle264_steel_beam_mind_blown_recoil_on_miss_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36129),
        pool_get_team(41269),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        926213119,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[3]["p0_action"] == "move 4"
    assert normal_rows[3]["p0_hp"] == 322
    assert normal_rows[3]["p1_hp"] == 326
    assert normal_rows[4]["p0_action"] == "move 3+forced:switch 2"
    assert normal_rows[4]["p0_active"] == 0
    assert normal_rows[4]["p0_hp"] == 318
    assert normal_rows[4]["p1_hp"] == 326
    assert normal_rows[5]["p0_action"] == "move 3"
    assert normal_rows[5]["p0_hp"] == 318
    assert normal_rows[5]["p1_action"] == "switch 3"
    assert normal_rows[5]["p1_active"] == 2
    assert normal_rows[5]["p1_hp"] == 315


def test_battle313_hazard_ko_pivot_replacement_waits_until_post_upkeep_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(25417),
        pool_get_team(34543),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        867657247,
        35,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[29]["p1_hp"] == 400

    assert normal_rows[30]["p0_action"] == "move 3"
    assert normal_rows[30]["p0_hp"] == 369
    assert normal_rows[30]["p1_action"] == "move 1+forced:switch 2+die_switch:switch 2"
    assert normal_rows[30]["p1_active"] == 1
    assert normal_rows[30]["p1_hp"] == 350

    assert normal_rows[31]["p1_action"] == "move 1+forced:switch 3"
    assert normal_rows[31]["p1_active"] == 2
    assert normal_rows[31]["p1_hp"] == 514


def test_battle273_defog_pressure_partial_trap_and_stockpile_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49955),
        pool_get_team(1198),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1429547085,
        110,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[55]["p0_hp"] == 233
    assert normal_rows[55]["p1_action"] == "move 3+forced:switch 5"
    assert normal_rows[55]["p1_hp"] == 464

    assert normal_rows[60]["p0_action"] == "switch 2"
    assert normal_rows[60]["p0_hp"] == 271
    assert normal_rows[64]["p0_hp"] == 270
    assert normal_rows[65]["p0_hp"] == 282
    assert normal_rows[65]["p1_hp"] == 54

    assert normal_rows[98]["p0_action"] == "move 4"
    assert normal_rows[98]["p1_action"] == "move 1"
    assert normal_rows[98]["p1_hp"] == 449
    assert normal_rows[99]["p1_hp"] == 430


def test_battle351_faster_user_faint_in_damaging_hit_invalidates_slower_targeted_move_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(47921),
        pool_get_team(3830),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2137743677,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[4]["p0_action"] == "move 1"
    assert normal_rows[4]["p0_hp"] == 228
    assert normal_rows[4]["p1_action"] == "move 4+forced:switch 2"
    assert normal_rows[4]["p1_active"] == 1

    assert normal_rows[5]["p0_action"] == "move 1+forced:switch 2"
    assert normal_rows[5]["p0_active"] == 1
    assert normal_rows[5]["p0_hp"] == 345
    assert normal_rows[5]["p1_action"] == "move 2+forced:switch 3"
    assert normal_rows[5]["p1_active"] == 2
    assert normal_rows[5]["p1_hp"] == 289


def test_battle365_shields_down_switch_in_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4002),
        pool_get_team(41772),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1622059194,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert "forced:switch" in normal_rows[38]["p0_action"]
    assert normal_rows[38]["p0_active"] == 4
    assert normal_rows[38]["p0_hp"] == 261

    assert normal_rows[39]["p0_active"] == 4
    assert normal_rows[39]["p1_active"] == 4
    assert normal_rows[39]["p1_hp"] == 206

    assert normal_rows[40]["p1_hp"] == 72


def test_battle417_first_mover_eject_pack_retargets_slower_move_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(20578),
        pool_get_team(3701),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1704419440,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[4]["p0_action"] == "move 3+forced:switch 3"
    assert normal_rows[4]["p0_active"] == 2
    assert normal_rows[4]["p0_hp"] == 218
    assert normal_rows[4]["p1_hp"] == 142


def test_battle235_strike_turn_phantom_force_residual_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(28247),
        pool_get_team(392),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1229285168,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert "forced:switch" in normal_rows[13]["p1_action"]
    assert normal_rows[13]["p0_active"] == 3
    assert normal_rows[13]["p1_active"] == 3
    assert normal_rows[13]["p1_hp"] == 235

    assert normal_rows[14]["p0_active"] == 3
    assert normal_rows[14]["p0_hp"] == 117
    assert normal_rows[14]["p1_hp"] == 235

    assert normal_rows[15]["p0_active"] == 3
    assert normal_rows[15]["p0_hp"] == 117
    assert normal_rows[15]["p0_status"] == 2
    assert normal_rows[15]["p1_hp"] == 99


def test_battle256_mid_turn_pivot_pp_bookkeeping_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(6996),
        pool_get_team(18817),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1399983420,
        35,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[23]["p1_action"] == "move 2"
    assert normal_rows[24]["p1_action"] == "move 1"
    assert normal_rows[25]["p1_action"] == "move 4"
    assert normal_rows[25]["p0_hp"] == 354
    assert normal_rows[26]["p1_action"] == "move 1"
    assert normal_rows[27]["p1_action"] == "move 4"


def test_battle269_charge_turn_on_try_move_selfboost_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(5529),
        pool_get_team(19546),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2141115624,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[6]["p0_action"] == "move 1"
    assert normal_rows[6]["p0_hp"] == 42
    assert normal_rows[6]["p1_action"] == "switch 5"
    assert normal_rows[6]["p1_hp"] == 514

    assert normal_rows[7]["p0_action"] == "move 1+forced:switch 6"
    assert normal_rows[7]["p0_hp"] == 345
    assert normal_rows[7]["p1_action"] == "move 2"
    assert normal_rows[7]["p1_hp"] == 433

    assert normal_rows[8]["p0_action"] == "move 4"
    assert normal_rows[8]["p0_hp"] == 345
    assert normal_rows[8]["p1_action"] == "switch 5"
    assert normal_rows[8]["p1_hp"] == 225


def test_battle269_knock_off_unburden_residual_speed_refresh_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(5529),
        pool_get_team(19546),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2141115624,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[23]["p0_action"] == "move 2"
    assert normal_rows[23]["p0_hp"] == 299
    assert normal_rows[23]["p1_action"] == "switch 3"
    assert normal_rows[23]["p1_hp"] == 8

    assert normal_rows[24]["p0_action"] == "move 3"
    assert normal_rows[24]["p0_hp"] == 136
    assert normal_rows[24]["p1_action"] == "move 3+forced:switch 3"
    assert normal_rows[24]["p1_hp"] == 225


def test_battle270_disable_plus_choice_lock_disablemove_shuffle_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(2208),
        pool_get_team(43960),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1469943246,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[11]["p0_action"] == "move 2+forced:switch 2"
    assert normal_rows[11]["p0_hp"] == 257
    assert normal_rows[11]["p1_action"] == "move 1"
    assert normal_rows[11]["p1_hp"] == 264

    assert normal_rows[12]["p0_action"] == "move 2"
    assert normal_rows[12]["p0_hp"] == 229
    assert normal_rows[12]["p1_action"] == "switch 6"
    assert normal_rows[12]["p1_hp"] == 111

    assert normal_rows[13]["p0_action"] == "switch 3"
    assert normal_rows[13]["p0_hp"] == 254
    assert normal_rows[13]["p1_action"] == "move 2"
    assert normal_rows[13]["p1_hp"] == 111


def test_battle276_avalanche_base_power_callback_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(43172),
        pool_get_team(1449),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        528101773,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[1]["p0_action"] == "move 3"
    assert normal_rows[1]["p0_hp"] == 451
    assert normal_rows[1]["p1_action"] == "move 4+forced:switch 2"
    assert normal_rows[1]["p1_hp"] == 315

    assert normal_rows[2]["p0_action"] == "move 1"
    assert normal_rows[2]["p0_hp"] == 265
    assert normal_rows[2]["p1_action"] == "move 4"
    assert normal_rows[2]["p1_hp"] == 270

    assert normal_rows[3]["p0_action"] == "move 1"
    assert normal_rows[3]["p0_hp"] == 83
    assert normal_rows[3]["p1_action"] == "move 4"
    assert normal_rows[3]["p1_hp"] == 223

    assert normal_rows[21]["p0_action"] == "move 1+forced:switch 3"
    assert normal_rows[21]["p0_hp"] == 399
    assert normal_rows[21]["p1_action"] == "move 1"
    assert normal_rows[21]["p1_hp"] == 89

    assert normal_rows[22]["p0_action"] == "move 2"
    assert normal_rows[22]["p0_hp"] == 381
    assert normal_rows[22]["p1_action"] == "move 1"
    assert normal_rows[22]["p1_hp"] == 75


def test_battle276_fixed_damage_immunity_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(43172),
        pool_get_team(1449),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        528101773,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[34]["p0_action"] == "move 1"
    assert normal_rows[34]["p0_hp"] == 591
    assert normal_rows[34]["p1_action"] == "move 1"
    assert normal_rows[34]["p1_hp"] == 223

    assert normal_rows[35]["p0_action"] == "move 1"
    assert normal_rows[35]["p0_hp"] == 472
    assert normal_rows[35]["p1_action"] == "move 1"
    assert normal_rows[35]["p1_hp"] == 223

    assert normal_rows[36]["p0_action"] == "move 1"
    assert normal_rows[36]["p0_hp"] == 330
    assert normal_rows[36]["p1_action"] == "move 1"
    assert normal_rows[36]["p1_hp"] == 223

    assert normal_rows[37]["p0_action"] == "move 1"
    assert normal_rows[37]["p0_hp"] == 203
    assert normal_rows[37]["p1_action"] == "move 1"
    assert normal_rows[37]["p1_hp"] == 223

    assert normal_rows[38]["p0_action"] == "move 1"
    assert normal_rows[38]["p0_hp"] == 61
    assert normal_rows[38]["p1_action"] == "move 1"
    assert normal_rows[38]["p1_hp"] == 223


def test_battle276_knock_off_restores_contact_item_for_slower_recoil_projection():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(43172),
        pool_get_team(1449),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        528101773,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 21, Gliscor's faster Knock Off first triggers Rocky Helmet,
    # then removes Rocky Helmet, then Corviknight's slower Brave Bird recoils
    # off the actual 84 HP removed from Gliscor. The slower move's live
    # pre-hit HP projection must therefore restore the defender's pre-Knock
    # Off item while simulating the faster move's contact-damage chain.
    assert normal_rows[20]["p0_action"] == "move 2"
    assert normal_rows[20]["p0_hp"] == 132
    assert normal_rows[20]["p1_action"] == "move 4"
    assert normal_rows[20]["p1_hp"] == 180

    assert normal_rows[21]["p0_action"] == "move 1+forced:switch 3"
    assert normal_rows[21]["p0_hp"] == 399
    assert normal_rows[21]["p1_action"] == "move 1"
    assert normal_rows[21]["p1_hp"] == 89


def test_battle279_switch_out_ability_suppression_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(9910),
        pool_get_team(728),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1955580702,
        90,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[14]["p0_action"] == "move 4+pivot:switch 2"
    assert normal_rows[14]["p0_hp"] == 334
    assert normal_rows[14]["p1_action"] == "move 4"
    assert normal_rows[14]["p1_hp"] == 312
    assert normal_rows[14]["p1_max_hp"] == 414

    assert normal_rows[15]["p0_action"] == "move 1"
    assert normal_rows[15]["p0_hp"] == 334
    assert normal_rows[15]["p1_action"] == "switch 4"
    assert normal_rows[15]["p1_hp"] == 315

    assert normal_rows[47]["p0_action"] == "move 2"
    assert normal_rows[47]["p0_hp"] == 303
    assert normal_rows[47]["p1_action"] == "move 2"
    assert normal_rows[47]["p1_hp"] == 46

    assert normal_rows[48]["p0_action"] == "move 2"
    assert normal_rows[48]["p0_hp"] == 230
    assert normal_rows[48]["p1_action"] == "move 2+forced:switch 4"
    assert normal_rows[48]["p1_hp"] == 261
    assert normal_rows[48]["p1_max_hp"] == 414

    assert normal_rows[49]["p0_action"] == "move 4"
    assert normal_rows[49]["p0_hp"] == 141
    assert normal_rows[49]["p1_action"] == "move 4"
    assert normal_rows[49]["p1_hp"] == 193


def test_battle336_late_uturn_switch_resume_tie_frames_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36669),
        pool_get_team(26013),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        577167324,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[3]["p0_action"] == "move 3"
    assert normal_rows[3]["p0_hp"] == 96
    assert normal_rows[3]["p1_action"] == "move 1+forced:switch 2"
    assert normal_rows[3]["p1_hp"] == 321

    assert normal_rows[4]["p0_action"] == "move 3"
    assert normal_rows[4]["p0_hp"] == 39
    assert normal_rows[4]["p1_action"] == "switch 5"
    assert normal_rows[4]["p1_hp"] == 181
    assert normal_rows[4]["p1_max_hp"] == 394

    assert normal_rows[5]["p0_action"] == "move 3+forced:switch 4"
    assert normal_rows[5]["p0_hp"] == 391
    assert normal_rows[5]["p1_action"] == "move 2+forced:switch 5"
    assert normal_rows[5]["p1_hp"] == 321


def test_battle407_lockedmove_notarget_execute_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(27655),
        pool_get_team(48076),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1404674228,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[8]["p0_action"] == "move 3"
    assert normal_rows[8]["p0_hp"] == 258
    assert normal_rows[8]["p1_action"] == "move 3+forced:switch 3"
    assert normal_rows[8]["p1_hp"] == 128

    assert normal_rows[9]["p0_action"] == "move 1"
    assert normal_rows[9]["p0_hp"] == 258
    assert normal_rows[9]["p1_action"] == "switch 4"
    assert normal_rows[9]["p1_hp"] == 211
    assert normal_rows[9]["p1_max_hp"] == 394

    assert normal_rows[10]["p0_action"] == "switch 2"
    assert normal_rows[10]["p0_hp"] == 301
    assert normal_rows[10]["p1_action"] == "move 2"
    assert normal_rows[10]["p1_hp"] == 211

    assert normal_rows[11]["p0_action"] == "move 2"
    assert normal_rows[11]["p0_hp"] == 301
    assert normal_rows[11]["p1_action"] == "switch 4"
    assert normal_rows[11]["p1_hp"] == 29
    assert normal_rows[11]["p1_max_hp"] == 315

    assert normal_rows[12]["p0_action"] == "switch 5"
    assert normal_rows[12]["p0_hp"] == 100
    assert normal_rows[12]["p1_action"] == "move 1"
    assert normal_rows[12]["p1_hp"] == 29


def test_battle282_magic_bounce_phazing_target_side_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(7816),
        pool_get_team(2582),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1760895470,
        120,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[67]["p0_action"] == "switch 6+forced:switch 6"
    assert normal_rows[67]["p0_hp"] == 482
    assert normal_rows[67]["p1_action"] == "move 3"
    assert normal_rows[67]["p1_active"] == 1
    assert normal_rows[67]["p1_hp"] == 279
    assert normal_rows[67]["p1_max_hp"] == 318

    assert normal_rows[68]["p0_action"] == "move 1"
    assert normal_rows[68]["p0_hp"] == 514
    assert normal_rows[68]["p1_action"] == "move 3"
    assert normal_rows[68]["p1_active"] == 1
    assert normal_rows[68]["p1_hp"] == 279
    assert normal_rows[68]["p1_max_hp"] == 318

    assert normal_rows[69]["p0_action"] == "move 1"
    assert normal_rows[69]["p0_hp"] == 514
    assert normal_rows[69]["p1_action"] == "move 3"
    assert normal_rows[69]["p1_active"] == 1
    assert normal_rows[69]["p1_hp"] == 279
    assert normal_rows[69]["p1_max_hp"] == 318


def test_battle283_tied_ko_skip_update_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(30296),
        pool_get_team(49002),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1721169082,
        90,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[45]["p0_action"] == "move 1+forced:switch 6"
    assert normal_rows[45]["p0_active"] == 5
    assert normal_rows[45]["p0_hp"] == 299
    assert normal_rows[45]["p1_action"] == "move 1"
    assert normal_rows[45]["p1_active"] == 4
    assert normal_rows[45]["p1_hp"] == 371

    assert normal_rows[50]["p0_action"] == "switch 2"
    assert normal_rows[50]["p0_active"] == 4
    assert normal_rows[50]["p0_hp"] == 208
    assert normal_rows[50]["p1_action"] == "move 3"
    assert normal_rows[50]["p1_active"] == 5
    assert normal_rows[50]["p1_hp"] == 323

    assert normal_rows[51]["p0_action"] == "move 4"
    assert normal_rows[51]["p0_hp"] == 208
    assert normal_rows[51]["p1_action"] == "move 3"
    assert normal_rows[51]["p1_hp"] == 215
    assert normal_rows[51]["p1_max_hp"] == 323

    assert normal_rows[52]["p0_action"] == "move 4"
    assert normal_rows[52]["p0_hp"] == 208
    assert normal_rows[52]["p1_action"] == "move 3"
    assert normal_rows[52]["p1_hp"] == 22
    assert normal_rows[52]["p1_max_hp"] == 323

    assert normal_rows[53]["p0_action"] == "move 4"
    assert normal_rows[53]["p0_hp"] == 208
    assert normal_rows[53]["p1_action"] == "move 3+forced:switch 5"
    assert normal_rows[53]["p1_active"] == 3
    assert normal_rows[53]["p1_hp"] == 394
    assert normal_rows[53]["p1_max_hp"] == 394


def test_battle486_dragontail_single_target_phaze_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(13192),
        pool_get_team(25804),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1696845673,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[20]["p0_action"] == "switch 4"
    assert normal_rows[20]["p0_hp"] == 217
    assert normal_rows[20]["p1_action"] == "move 3"
    assert normal_rows[20]["p1_hp"] == 46
    assert normal_rows[20]["p1_max_hp"] == 319

    assert normal_rows[21]["p0_action"] == "move 1"
    assert normal_rows[21]["p0_hp"] == 164
    assert normal_rows[21]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[21]["p1_hp"] == 379
    assert normal_rows[21]["p1_max_hp"] == 404

    assert normal_rows[22]["p1_action"] == "move 4"
    assert normal_rows[22]["p1_hp"] == 379
    assert normal_rows[23]["p0_action"] == "move 1"
    assert normal_rows[23]["p0_hp"] == 164
    assert normal_rows[23]["p1_action"] == "move 4"
    assert normal_rows[23]["p1_hp"] == 227
    assert normal_rows[23]["p1_max_hp"] == 301

    assert normal_rows[24]["p1_action"] == "switch 3"
    assert normal_rows[24]["p1_hp"] == 190
    assert normal_rows[25]["p1_action"] == "switch 3"
    assert normal_rows[25]["p1_hp"] == 153
    assert normal_rows[26]["p1_action"] == "switch 3"
    assert normal_rows[26]["p1_hp"] == 116
    assert normal_rows[26]["p1_max_hp"] == 301

    assert normal_rows[29]["p0_hp"] == 33
    assert normal_rows[29]["p1_action"] == "move 1"
    assert normal_rows[29]["p1_hp"] == 0


def test_battle491_gulp_missile_fires_on_ko_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(10351),
        pool_get_team(48857),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        648693754,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    for turn in range(45, 50):
        assert normal_rows[turn]["p0_action"] == "move 1"
        assert normal_rows[turn]["p0_active"] == 3
        assert normal_rows[turn]["p0_hp"] == 67
        assert normal_rows[turn]["p1_action"] == "move 4"
        assert normal_rows[turn]["p1_active"] == 5
        assert normal_rows[turn]["p1_hp"] == 454
        assert normal_rows[turn]["p1_max_hp"] == 454

    assert normal_rows[50]["p0_action"] == "move 1"
    assert normal_rows[50]["p0_active"] == 3
    assert normal_rows[50]["p0_hp"] == 0
    assert normal_rows[50]["p0_max_hp"] == 281
    assert normal_rows[50]["p1_action"] == "move 3"
    assert normal_rows[50]["p1_active"] == 5
    assert normal_rows[50]["p1_hp"] == 323
    assert normal_rows[50]["p1_max_hp"] == 454


def test_battle699_multihit_update_frames_on_ko_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(43031),
        pool_get_team(37562),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        529742076,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[8]["p0_action"] == "move 4+forced:switch 2"
    assert normal_rows[8]["p0_active"] == 1
    assert normal_rows[8]["p0_hp"] == 394
    assert normal_rows[8]["p0_max_hp"] == 394
    assert normal_rows[8]["p1_action"] == "move 4"
    assert normal_rows[8]["p1_active"] == 1
    assert normal_rows[8]["p1_hp"] == 163
    assert normal_rows[8]["p1_max_hp"] == 323

    assert normal_rows[9]["p0_action"] == "move 2"
    assert normal_rows[9]["p0_hp"] == 394
    assert normal_rows[9]["p1_action"] == "switch 4"
    assert normal_rows[9]["p1_active"] == 3
    assert normal_rows[9]["p1_hp"] == 195
    assert normal_rows[9]["p1_max_hp"] == 301

    assert normal_rows[10]["p0_action"] == "switch 5"
    assert normal_rows[10]["p0_active"] == 4
    assert normal_rows[10]["p0_hp"] == 224
    assert normal_rows[10]["p0_max_hp"] == 357
    assert normal_rows[10]["p1_action"] == "move 3"
    assert normal_rows[10]["p1_active"] == 3
    assert normal_rows[10]["p1_hp"] == 195
    assert normal_rows[10]["p1_max_hp"] == 301


def test_battle408_double_shock_live_type_removal_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(20842),
        pool_get_team(48490),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        230324149,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}
    assert len(normal_rows) == 17

    assert normal_rows[11]["p0_action"] == "move 1"
    assert normal_rows[11]["p0_active"] == 1
    assert normal_rows[11]["p0_hp"] == 281
    assert normal_rows[11]["p1_action"] == "switch 4"
    assert normal_rows[11]["p1_active"] == 3
    assert normal_rows[11]["p1_hp"] == 36
    assert normal_rows[11]["p1_max_hp"] == 301

    assert normal_rows[12]["p0_action"] == "move 1"
    assert normal_rows[12]["p0_active"] == 1
    assert normal_rows[12]["p0_hp"] == 74
    assert normal_rows[12]["p0_max_hp"] == 281
    assert normal_rows[12]["p1_action"] == "move 3"
    assert normal_rows[12]["p1_active"] == 3
    assert normal_rows[12]["p1_hp"] == 140
    assert normal_rows[12]["p1_max_hp"] == 301


def test_battle464_water_pulse_confusion_secondary_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49199),
        pool_get_team(37115),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        463056159,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[14]["p0_action"] == "move 4+forced:switch 6"
    assert normal_rows[14]["p0_active"] == 1
    assert normal_rows[14]["p0_hp"] == 248
    assert normal_rows[14]["p0_max_hp"] == 283
    assert normal_rows[14]["p1_action"] == "move 2"
    assert normal_rows[14]["p1_active"] == 4
    assert normal_rows[14]["p1_hp"] == 212
    assert normal_rows[14]["p1_max_hp"] == 374

    assert normal_rows[15]["p0_action"] == "move 1"
    assert normal_rows[15]["p0_active"] == 1
    assert normal_rows[15]["p0_hp"] == 109
    assert normal_rows[15]["p0_max_hp"] == 283
    assert normal_rows[15]["p1_action"] == "move 4"
    assert normal_rows[15]["p1_active"] == 4
    assert normal_rows[15]["p1_hp"] == 13
    assert normal_rows[15]["p1_max_hp"] == 374


def test_battle287_red_card_suppresses_attacker_aftermove_hp_effects_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(46389),
        pool_get_team(6011),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1658090161,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[15]["p0_action"] == "move 2"
    assert normal_rows[15]["p0_active"] == 1
    assert normal_rows[15]["p0_hp"] == 134
    assert normal_rows[15]["p0_max_hp"] == 315
    assert normal_rows[15]["p1_action"] == "move 1"
    assert normal_rows[15]["p1_active"] == 4
    assert normal_rows[15]["p1_hp"] == 136
    assert normal_rows[15]["p1_max_hp"] == 514

    assert normal_rows[16]["p0_action"] == "move 1"
    assert normal_rows[16]["p0_active"] == 1
    assert normal_rows[16]["p0_hp"] == 134
    assert normal_rows[16]["p0_max_hp"] == 315
    assert normal_rows[16]["p1_action"] == "move 1+forced:switch 5"
    assert normal_rows[16]["p1_active"] == 1
    assert normal_rows[16]["p1_hp"] == 371
    assert normal_rows[16]["p1_max_hp"] == 371

    assert normal_rows[17]["p0_action"] == "switch 2"
    assert normal_rows[17]["p0_active"] == 4
    assert normal_rows[17]["p0_hp"] == 221
    assert normal_rows[17]["p0_max_hp"] == 299
    assert normal_rows[17]["p1_action"] == "move 2"
    assert normal_rows[17]["p1_active"] == 1
    assert normal_rows[17]["p1_hp"] == 371
    assert normal_rows[17]["p1_max_hp"] == 371


def test_battle300_weakness_policy_preapplies_before_slower_move_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(218),
        pool_get_team(2117),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        470590418,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[1]["p0_action"] == "move 3+pivot:switch 3"
    assert normal_rows[1]["p0_active"] == 2
    assert normal_rows[1]["p0_hp"] == 259
    assert normal_rows[1]["p0_max_hp"] == 371
    assert normal_rows[1]["p1_action"] == "move 1"
    assert normal_rows[1]["p1_active"] == 0
    assert normal_rows[1]["p1_hp"] == 161
    assert normal_rows[1]["p1_max_hp"] == 404

    assert normal_rows[2]["p0_action"] == "move 2"
    assert normal_rows[2]["p0_active"] == 2
    assert normal_rows[2]["p0_hp"] == 259
    assert normal_rows[2]["p0_max_hp"] == 371
    assert normal_rows[2]["p1_action"] == "switch 3"
    assert normal_rows[2]["p1_active"] == 2
    assert normal_rows[2]["p1_hp"] == 323
    assert normal_rows[2]["p1_max_hp"] == 323


def test_battle302_quick_draw_fractional_priority_prng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(20578),
        pool_get_team(8656),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        797052560,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[7]["p0_action"] == "move 4"
    assert normal_rows[7]["p0_active"] == 2
    assert normal_rows[7]["p0_hp"] == 22
    assert normal_rows[7]["p0_max_hp"] == 318
    assert normal_rows[7]["p1_action"] == "switch 5"
    assert normal_rows[7]["p1_active"] == 4
    assert normal_rows[7]["p1_hp"] == 277
    assert normal_rows[7]["p1_max_hp"] == 394

    assert normal_rows[8]["p0_action"] == "switch 6"
    assert normal_rows[8]["p0_active"] == 5
    assert normal_rows[8]["p0_hp"] == 322
    assert normal_rows[8]["p0_max_hp"] == 386
    assert normal_rows[8]["p1_action"] == "move 2"
    assert normal_rows[8]["p1_active"] == 4
    assert normal_rows[8]["p1_hp"] == 277
    assert normal_rows[8]["p1_max_hp"] == 394

    assert normal_rows[9]["p0_action"] == "move 1"
    assert normal_rows[9]["p0_active"] == 5
    assert normal_rows[9]["p0_hp"] == 251
    assert normal_rows[9]["p0_max_hp"] == 386
    assert normal_rows[9]["p1_action"] == "move 2"
    assert normal_rows[9]["p1_active"] == 4
    assert normal_rows[9]["p1_hp"] == 277
    assert normal_rows[9]["p1_max_hp"] == 394


def test_battle309_endure_and_substitute_secondary_preroll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(15067),
        pool_get_team(20918),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1241282519,
        120,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[50]["p0_action"] == "move 3"
    assert normal_rows[50]["p0_hp"] == 340
    assert normal_rows[50]["p1_action"] == "move 2"
    assert normal_rows[50]["p1_hp"] == 364

    assert normal_rows[51]["p0_action"] == "move 3"
    assert normal_rows[51]["p0_hp"] == 340
    assert normal_rows[51]["p1_action"] == "move 2"
    assert normal_rows[51]["p1_hp"] == 389

    assert normal_rows[52]["p0_action"] == "move 3"
    assert normal_rows[52]["p0_hp"] == 340
    assert normal_rows[52]["p1_action"] == "move 2"
    assert normal_rows[52]["p1_hp"] == 404

    assert normal_rows[71]["p0_action"] == "move 1"
    assert normal_rows[71]["p0_active"] == 2
    assert normal_rows[71]["p0_hp"] == 380
    assert normal_rows[71]["p1_action"] == "move 3"
    assert normal_rows[71]["p1_active"] == 1
    assert normal_rows[71]["p1_hp"] == 404
    assert normal_rows[71]["p1_status"] == 0

    assert normal_rows[72]["p0_action"] == "move 1"
    assert normal_rows[72]["p0_hp"] == 380
    assert normal_rows[72]["p1_action"] == "move 3"
    assert normal_rows[72]["p1_hp"] == 396
    assert normal_rows[72]["p1_status"] == 0

    assert normal_rows[73]["p0_action"] == "move 1"
    assert normal_rows[73]["p0_hp"] == 380
    assert normal_rows[73]["p1_action"] == "move 3"
    assert normal_rows[73]["p1_hp"] == 353
    assert normal_rows[73]["p1_status"] == 6


def test_battle926_same_turn_freeze_thaw_before_flinch_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(19435),
        pool_get_team(2255),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        667779876,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[11]["p0_action"] == "move 1"
    assert normal_rows[11]["p0_active"] == 2
    assert normal_rows[11]["p0_hp"] == 61
    assert normal_rows[11]["p0_status"] == 4
    assert normal_rows[11]["p1_action"] == "move 3"
    assert normal_rows[11]["p1_hp"] == 104

    assert normal_rows[12]["p0_action"] == "move 1+forced:switch 6"
    assert normal_rows[12]["p0_active"] == 5
    assert normal_rows[12]["p0_hp"] == 325
    assert normal_rows[12]["p0_status"] == 0
    assert normal_rows[12]["p1_action"] == "move 3"
    assert normal_rows[12]["p1_hp"] == 124

    assert normal_rows[13]["p0_action"] == "move 2"
    assert normal_rows[13]["p0_hp"] == 255
    assert normal_rows[13]["p1_action"] == "move 1"
    assert normal_rows[13]["p1_hp"] == 111

    assert normal_rows[14]["p0_action"] == "move 2"
    assert normal_rows[14]["p0_hp"] == 176
    assert normal_rows[14]["p1_action"] == "move 1"
    assert normal_rows[14]["p1_hp"] == 96


def test_battle311_status_immunity_blocks_accuracy_preroll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(32461),
        pool_get_team(22133),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2019911114,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    for turn in range(30, 35):
        assert normal_rows[turn]["p0_action"] == "move 1"
        assert normal_rows[turn]["p0_hp"] == 101
        assert normal_rows[turn]["p1_action"] == "move 1"
        assert normal_rows[turn]["p1_hp"] == 300
        assert normal_rows[turn]["p1_status"] == 0

    assert normal_rows[35]["p0_action"] == "move 3"
    assert normal_rows[35]["p0_hp"] == 101
    assert normal_rows[35]["p1_action"] == "move 1"
    assert normal_rows[35]["p1_hp"] == 27
    assert normal_rows[35]["p1_status"] == 0

    assert normal_rows[36]["p0_action"] == "move 3"
    assert normal_rows[36]["p0_hp"] == 101
    assert normal_rows[36]["p1_action"] == "move 1+forced:switch 3"
    assert normal_rows[36]["p1_active"] == 2
    assert normal_rows[36]["p1_hp"] == 261


def test_battle316_instaswitch_update_and_repeated_wish_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(47392),
        pool_get_team(13903),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        200594802,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[5]["p1_action"] == "move 1"
    assert normal_rows[5]["p1_hp"] == 394
    assert normal_rows[6]["p1_action"] == "move 1"
    assert normal_rows[6]["p1_hp"] == 394
    assert normal_rows[7]["p1_action"] == "move 1"
    assert normal_rows[7]["p1_hp"] == 337
    assert normal_rows[8]["p1_action"] == "move 1"
    assert normal_rows[8]["p1_hp"] == 361

    assert normal_rows[21]["p1_action"] == "move 2"
    assert normal_rows[21]["p1_hp"] == 281
    assert normal_rows[22]["p1_action"] == "move 2"
    assert normal_rows[22]["p1_hp"] == 327
    assert normal_rows[23]["p1_action"] == "move 2"
    assert normal_rows[23]["p1_hp"] == 134


def test_battle504_skill_link_skips_multihit_preroll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(37197),
        pool_get_team(16536),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        530144334,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[47]["p0_action"] == "move 1"
    assert normal_rows[47]["p1_hp"] == 254
    assert normal_rows[48]["p0_action"] == "move 1"
    assert normal_rows[48]["p1_hp"] == 254
    assert normal_rows[49]["p0_action"] == "move 1"
    assert normal_rows[49]["p1_hp"] == 254

    assert normal_rows[50]["p0_action"] == "move 3+forced:switch 3"
    assert normal_rows[50]["p0_active"] == 1
    assert normal_rows[50]["p0_hp"] == 289
    assert normal_rows[50]["p1_action"] == "move 4"
    assert normal_rows[50]["p1_hp"] == 35

    assert normal_rows[51]["p0_action"] == "move 2"
    assert normal_rows[51]["p0_hp"] == 289
    assert normal_rows[51]["p1_action"] == "move 4+forced:switch 4"
    assert normal_rows[51]["p1_active"] == 3
    assert normal_rows[51]["p1_hp"] == 301

    assert normal_rows[52]["p0_action"] == "move 4"
    assert normal_rows[52]["p0_hp"] == 75
    assert normal_rows[52]["p1_action"] == "move 4"
    assert normal_rows[52]["p1_hp"] == 40


def test_battle319_fainted_pivot_user_does_not_switch_in_replacement_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(31731),
        pool_get_team(48573),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        884357166,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[1]["p0_action"] == "move 4"
    assert normal_rows[1]["p0_hp"] == 278
    assert normal_rows[1]["p1_action"] == "move 1"
    assert normal_rows[1]["p1_hp"] == 14

    assert normal_rows[2]["p0_action"] == "move 4"
    assert normal_rows[2]["p0_hp"] == 151
    assert normal_rows[2]["p1_action"] == "move 4+forced:switch 2"
    assert normal_rows[2]["p1_active"] == 1
    assert normal_rows[2]["p1_hp"] == 381

    assert normal_rows[3]["p0_action"] == "move 4+forced:switch 5"
    assert normal_rows[3]["p0_active"] == 4
    assert normal_rows[3]["p0_hp"] == 412
    assert normal_rows[3]["p1_action"] == "move 3"
    assert normal_rows[3]["p1_hp"] == 343

    assert normal_rows[4]["p0_action"] == "move 3"
    assert normal_rows[4]["p0_hp"] == 412
    assert normal_rows[4]["p1_action"] == "switch 4"
    assert normal_rows[4]["p1_hp"] == 255


def test_battle731_prankster_parting_shot_dark_immunity_blocks_inline_pivot_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(10942),
        pool_get_team(32871),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1752466951,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[10]["p0_action"] == "move 1"
    assert normal_rows[10]["p0_hp"] == 253
    assert normal_rows[10]["p1_action"] == "move 3+forced:switch 3"
    assert normal_rows[10]["p1_active"] == 1
    assert normal_rows[10]["p1_hp"] == 394

    assert normal_rows[11]["p0_action"] == "move 2"
    assert normal_rows[11]["p0_hp"] == 225
    assert normal_rows[11]["p1_action"] == "move 1"
    assert normal_rows[11]["p1_active"] == 1
    assert normal_rows[11]["p1_hp"] == 35
    assert normal_rows[11]["p1_max_hp"] == 394


def test_battle330_covert_cloak_blocks_confusion_secondary_preroll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(17135),
        pool_get_team(27457),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        422996468,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[49]["p0_action"] == "move 1"
    assert normal_rows[49]["p0_hp"] == 346
    assert normal_rows[49]["p1_action"] == "switch 6"
    assert normal_rows[49]["p1_hp"] == 100

    assert normal_rows[50]["p0_action"] == "move 3"
    assert normal_rows[50]["p0_hp"] == 141
    assert normal_rows[50]["p1_action"] == "move 4+forced:switch 6"
    assert normal_rows[50]["p1_active"] == 3
    assert normal_rows[50]["p1_hp"] == 206

    assert normal_rows[51]["p0_action"] == "move 1"
    assert normal_rows[51]["p0_hp"] == 130
    assert normal_rows[51]["p1_action"] == "move 2"
    assert normal_rows[51]["p1_hp"] == 27


def test_battle334_toxic_debris_does_not_double_lay_toxic_spikes_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(27915),
        pool_get_team(44320),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        275453251,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[11]["p0_action"] == "switch 4"
    assert normal_rows[11]["p0_hp"] == 301
    assert normal_rows[11]["p1_action"] == "move 1"
    assert normal_rows[11]["p1_hp"] == 296

    assert normal_rows[12]["p0_action"] == "move 1"
    assert normal_rows[12]["p0_active"] == 5
    assert normal_rows[12]["p0_hp"] == 165
    assert normal_rows[12]["p0_status"] == 5
    assert normal_rows[12]["p1_action"] == "switch 2"
    assert normal_rows[12]["p1_active"] == 0
    assert normal_rows[12]["p1_hp"] == 164


def test_battle338_residual_uses_cached_speed_for_fainted_switchin_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(25964),
        pool_get_team(41869),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        824475008,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[41]["p0_action"] == "move 1"
    assert normal_rows[41]["p0_hp"] == 292
    assert normal_rows[41]["p1_action"] == "switch 6+die_switch:switch 5"
    assert normal_rows[41]["p1_active"] == 2
    assert normal_rows[41]["p1_hp"] == 33

    assert normal_rows[42]["p0_action"] == "move 1"
    assert normal_rows[42]["p0_active"] == 3
    assert normal_rows[42]["p0_hp"] == 95
    assert normal_rows[42]["p1_action"] == "move 4+forced:switch 6"
    assert normal_rows[42]["p1_active"] == 4
    assert normal_rows[42]["p1_hp"] == 282

    assert normal_rows[43]["p0_hp"] == 2
    assert normal_rows[43]["p1_hp"] == 0


def test_battle342_disablemove_shuffle_skips_expiring_heal_block_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(3882),
        pool_get_team(14030),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1213961993,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[10]["p0_action"] == "move 1"
    assert normal_rows[10]["p0_hp"] == 301
    assert normal_rows[10]["p1_action"] == "move 1"
    assert normal_rows[10]["p1_hp"] == 68

    assert normal_rows[11]["p0_action"] == "move 1"
    assert normal_rows[11]["p0_hp"] == 301
    assert normal_rows[11]["p1_action"] == "move 1+forced:switch 2"
    assert normal_rows[11]["p1_active"] == 1
    assert normal_rows[11]["p1_hp"] == 361

    assert normal_rows[12]["p0_action"] == "move 3"
    assert normal_rows[12]["p0_hp"] == 301
    assert normal_rows[12]["p1_action"] == "switch 5"
    assert normal_rows[12]["p1_active"] == 4
    assert normal_rows[12]["p1_hp"] == 182


def test_battle343_speed_boost_and_residual_use_showdown_cached_action_speeds_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(16706),
        pool_get_team(24378),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        371502742,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[3]["p0_action"] == "move 4"
    assert normal_rows[3]["p0_hp"] == 301
    assert normal_rows[3]["p1_action"] == "switch 6"
    assert normal_rows[3]["p1_active"] == 5
    assert normal_rows[3]["p1_hp"] == 205

    assert normal_rows[4]["p0_action"] == "switch 2"
    assert normal_rows[4]["p0_active"] == 0
    assert normal_rows[4]["p0_hp"] == 64
    assert normal_rows[4]["p1_action"] == "move 1"
    assert normal_rows[4]["p1_hp"] == 205

    assert normal_rows[12]["p0_action"] == "move 2+forced:switch 3"
    assert normal_rows[12]["p0_active"] == 2
    assert normal_rows[12]["p0_hp"] == 534
    assert normal_rows[12]["p1_action"] == "move 2"
    assert normal_rows[12]["p1_hp"] == 205

    assert normal_rows[13]["p0_action"] == "move 2"
    assert normal_rows[13]["p0_hp"] == 341
    assert normal_rows[13]["p1_action"] == "move 4"
    assert normal_rows[13]["p1_hp"] == 114

    assert normal_rows[14]["p0_action"] == "move 2"
    assert normal_rows[14]["p0_hp"] == 163
    assert normal_rows[14]["p1_action"] == "move 4"
    assert normal_rows[14]["p1_hp"] == 2


def test_battle344_all_moves_disabled_fall_back_to_struggle_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36538),
        pool_get_team(15694),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        142895323,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[50]["p0_action"] == "move 1"
    assert normal_rows[50]["p0_active"] == 4
    assert normal_rows[50]["p0_hp"] == 159
    assert normal_rows[50]["p1_action"] == "move 1"
    assert normal_rows[50]["p1_active"] == 0
    assert normal_rows[50]["p1_hp"] == 120


def test_battle347_payback_does_not_boost_vs_fresh_switch_in_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42827),
        pool_get_team(20760),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        799425641,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[25]["p0_action"] == "switch 6"
    assert normal_rows[25]["p0_active"] == 4
    assert normal_rows[25]["p0_hp"] == 262
    assert normal_rows[25]["p1_action"] == "move 1"
    assert normal_rows[25]["p1_hp"] == 366


def test_battle354_deferred_hit_loop_update_preserves_cursed_body_roll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(7202),
        pool_get_team(35270),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1617221371,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[12]["p0_action"] == "switch 6"
    assert normal_rows[12]["p0_active"] == 5
    assert normal_rows[12]["p0_hp"] == 1
    assert normal_rows[12]["p1_action"] == "move 2"
    assert normal_rows[12]["p1_hp"] == 301

    assert normal_rows[13]["p0_action"] == "move 1+forced:switch 6"
    assert normal_rows[13]["p0_active"] == 4
    assert normal_rows[13]["p0_hp"] == 281
    assert normal_rows[13]["p1_action"] == "move 2"
    assert normal_rows[13]["p1_hp"] == 301

    assert normal_rows[14]["p0_action"] == "switch 3+forced:switch 3"
    assert normal_rows[14]["p0_active"] == 4
    assert normal_rows[14]["p0_hp"] == 281
    assert normal_rows[14]["p1_action"] == "move 3"
    assert normal_rows[14]["p1_hp"] == 301

    assert normal_rows[15]["p0_action"] == "move 3"
    assert normal_rows[15]["p0_hp"] == 0
    assert normal_rows[15]["p1_action"] == "move 3"
    assert normal_rows[15]["p1_hp"] == 144


def test_battle361_lando_uturn_flame_body_contact_preroll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(27798),
        pool_get_team(10024),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1801888526,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[29]["p0_action"] == "move 3+forced:switch 5"
    assert normal_rows[29]["p0_active"] == 0
    assert normal_rows[29]["p0_hp"] == 243
    assert normal_rows[29]["p1_action"] == "move 3"
    assert normal_rows[29]["p1_hp"] == 45

    assert normal_rows[30]["p0_action"] == "move 1"
    assert normal_rows[30]["p0_active"] == 0
    assert normal_rows[30]["p0_hp"] == 105
    assert normal_rows[30]["p1_action"] == "move 3+forced:switch 4"
    assert normal_rows[30]["p1_active"] == 3
    assert normal_rows[30]["p1_hp"] == 318

    assert normal_rows[31]["p0_action"] == "move 2"
    assert normal_rows[31]["p0_hp"] == 0
    assert normal_rows[31]["p1_action"] == "move 3"
    assert normal_rows[31]["p1_hp"] == 212


def test_battle362_residual_ability_frames_preserve_turn6_focus_blast_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(40081),
        pool_get_team(49516),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1397899961,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[5]["p0_action"] == "move 4"
    assert normal_rows[5]["p0_active"] == 3
    assert normal_rows[5]["p0_hp"] == 109
    assert normal_rows[5]["p0_status"] == 5
    assert normal_rows[5]["p1_action"] == "move 3"
    assert normal_rows[5]["p1_active"] == 1
    assert normal_rows[5]["p1_hp"] == 281

    assert normal_rows[6]["p0_action"] == "move 4"
    assert normal_rows[6]["p0_active"] == 3
    assert normal_rows[6]["p0_hp"] == 63
    assert normal_rows[6]["p0_status"] == 5
    assert normal_rows[6]["p1_action"] == "move 3+forced:switch 3"
    assert normal_rows[6]["p1_active"] == 2
    assert normal_rows[6]["p1_hp"] == 241

    assert normal_rows[7]["p0_action"] == "move 3+forced:switch 2"
    assert normal_rows[7]["p0_active"] == 1
    assert normal_rows[7]["p0_hp"] == 320
    assert normal_rows[7]["p1_action"] == "move 1"
    assert normal_rows[7]["p1_active"] == 2
    assert normal_rows[7]["p1_hp"] == 241


def test_battle367_psychic_noise_blocks_poison_heal_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(23466),
        pool_get_team(45946),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        323567747,
        15,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[5]["p0_action"] == "move 3+forced:switch 2"
    assert normal_rows[5]["p0_active"] == 1
    assert normal_rows[5]["p0_hp"] == 317
    assert normal_rows[5]["p0_status"] == 6
    assert normal_rows[5]["p1_action"] == "move 2"
    assert normal_rows[5]["p1_active"] == 0
    assert normal_rows[5]["p1_hp"] == 352
    assert normal_rows[5]["p1_status"] == 6

    assert normal_rows[6]["p0_action"] == "move 1"
    assert normal_rows[6]["p0_active"] == 1
    assert normal_rows[6]["p0_hp"] == 199
    assert normal_rows[6]["p0_status"] == 6
    assert normal_rows[6]["p1_action"] == "move 2"
    assert normal_rows[6]["p1_active"] == 0
    assert normal_rows[6]["p1_hp"] == 165
    assert normal_rows[6]["p1_max_hp"] == 352
    assert normal_rows[6]["p1_status"] == 6

    assert normal_rows[7]["p0_action"] == "move 1"
    assert normal_rows[7]["p0_active"] == 1
    assert normal_rows[7]["p0_hp"] == 71
    assert normal_rows[7]["p0_status"] == 6
    assert normal_rows[7]["p1_action"] == "move 2+forced:switch 2"
    assert normal_rows[7]["p1_active"] == 1
    assert normal_rows[7]["p1_hp"] == 281
    assert normal_rows[7]["p1_status"] == 0


def test_battle371_faster_ko_skips_slower_status_accuracy_frame_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(14602),
        pool_get_team(9719),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1052691206,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[17]["p0_action"] == "move 1+forced:switch 2"
    assert normal_rows[17]["p0_active"] == 1
    assert normal_rows[17]["p0_hp"] == 281
    assert normal_rows[17]["p1_action"] == "move 1"
    assert normal_rows[17]["p1_active"] == 2
    assert normal_rows[17]["p1_hp"] == 97

    assert normal_rows[18]["p0_action"] == "move 1"
    assert normal_rows[18]["p0_active"] == 1
    assert normal_rows[18]["p0_hp"] == 281
    assert normal_rows[18]["p1_action"] == "move 1+forced:switch 3"
    assert normal_rows[18]["p1_active"] == 1
    assert normal_rows[18]["p1_hp"] == 450

    assert normal_rows[19]["p0_action"] == "move 2"
    assert normal_rows[19]["p0_active"] == 1
    assert normal_rows[19]["p0_hp"] == 142
    assert normal_rows[19]["p1_action"] == "move 3"
    assert normal_rows[19]["p1_active"] == 1
    assert normal_rows[19]["p1_hp"] == 420

    assert normal_rows[20]["p0_action"] == "move 2+forced:switch 5"
    assert normal_rows[20]["p0_active"] == 3
    assert normal_rows[20]["p0_hp"] == 384
    assert normal_rows[20]["p1_action"] == "move 3"
    assert normal_rows[20]["p1_active"] == 1
    assert normal_rows[20]["p1_hp"] == 388


def test_battle456_booster_energy_is_nonremovable_on_paradox_holder_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(48157),
        pool_get_team(31727),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1891330374,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[5]["p0_action"] == "move 3+pivot:switch 6"
    assert normal_rows[5]["p0_active"] == 5
    assert normal_rows[5]["p0_hp"] == 293
    assert normal_rows[5]["p1_action"] == "switch 3"
    assert normal_rows[5]["p1_active"] == 2
    assert normal_rows[5]["p1_hp"] == 208

    assert normal_rows[6]["p0_action"] == "move 4"
    assert normal_rows[6]["p0_active"] == 5
    assert normal_rows[6]["p0_hp"] == 293
    assert normal_rows[6]["p1_action"] == "switch 3"
    assert normal_rows[6]["p1_active"] == 1
    assert normal_rows[6]["p1_hp"] == 270

    assert normal_rows[7]["p0_action"] == "move 3+forced:switch 6"
    assert normal_rows[7]["p0_active"] == 2
    assert normal_rows[7]["p0_hp"] == 384
    assert normal_rows[7]["p1_action"] == "move 2"
    assert normal_rows[7]["p1_active"] == 1
    assert normal_rows[7]["p1_hp"] == 270


def test_battle372_plate_family_type_boost_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49780),
        pool_get_team(29322),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1553152756,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[1]["p0_action"] == "switch 4"
    assert normal_rows[1]["p0_active"] == 3
    assert normal_rows[1]["p0_hp"] == 97
    assert normal_rows[1]["p1_action"] == "move 1"
    assert normal_rows[1]["p1_active"] == 0
    assert normal_rows[1]["p1_hp"] == 323

    assert normal_rows[2]["p0_action"] == "move 2+forced:switch 4"
    assert normal_rows[2]["p0_active"] == 0
    assert normal_rows[2]["p0_hp"] == 394
    assert normal_rows[2]["p1_action"] == "move 1"
    assert normal_rows[2]["p1_active"] == 0
    assert normal_rows[2]["p1_hp"] == 190

    assert normal_rows[3]["p0_action"] == "switch 2"
    assert normal_rows[3]["p0_active"] == 1
    assert normal_rows[3]["p0_hp"] == 301
    assert normal_rows[3]["p1_action"] == "move 1"
    assert normal_rows[3]["p1_active"] == 0
    assert normal_rows[3]["p1_hp"] == 190


def test_battle460_phazing_triggers_switch_out_abilities_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49025),
        pool_get_team(14227),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        744548739,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[47]["p0_action"] == "move 1"
    assert normal_rows[47]["p0_hp"] == 40
    assert normal_rows[47]["p1_action"] == "move 2+forced:switch 6"
    assert normal_rows[47]["p1_active"] == 3
    assert normal_rows[47]["p1_hp"] == 272

    assert normal_rows[48]["p0_action"] == "move 1+forced:switch 2"
    assert normal_rows[48]["p0_active"] == 1
    assert normal_rows[48]["p0_hp"] == 301
    assert normal_rows[48]["p1_action"] == "move 2+forced:switch 6"
    assert normal_rows[48]["p1_active"] == 5
    assert normal_rows[48]["p1_hp"] == 364

    assert normal_rows[49]["p0_action"] == "move 1"
    assert normal_rows[49]["p0_hp"] == 301
    assert normal_rows[49]["p1_action"] == "move 4"
    assert normal_rows[49]["p1_active"] == 5
    assert normal_rows[49]["p1_hp"] == 169


def test_battle460_pivot_then_hidden_drag_preserves_showdown_switch_order():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49025),
        pool_get_team(14227),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        744548739,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 29 is the key ordering case: Tornadus pivots to Keldeo, then the
    # slower Whirlwind secretly drags Hydrapple in. Future `switch N` labels
    # must keep the visible pivot rewrite before the hidden drag rewrite.
    assert normal_rows[29]["p1_action"] == "move 2+forced:switch 5"
    assert normal_rows[29]["p1_active"] == 5
    assert normal_rows[29]["p1_hp"] == 364
    assert normal_rows[29]["p1_max_hp"] == 415

    # With the updated order preserved, the next voluntary `switch 5` still
    # points at Keldeo, which then faints to Stealth Rock before Tornadus
    # enters as the same-turn replacement.
    assert normal_rows[31]["p1_action"] == "switch 6+die_switch:switch 5"
    assert normal_rows[31]["p1_active"] == 3
    assert normal_rows[31]["p1_hp"] == 272
    assert normal_rows[31]["p1_max_hp"] == 362

    assert normal_rows[32]["p1_action"] == "switch 4"
    assert normal_rows[32]["p1_active"] == 4
    assert normal_rows[32]["p1_hp"] == 201
    assert normal_rows[32]["p1_max_hp"] == 386


def test_battle374_lockedmove_immunity_clears_forced_repeat_regression():
    from pokepy.core.constants import (
        M_ACTIVE0,
        M_ACTIVE1,
        M_LOCKED_MOVE_0,
        M_LOCKED_TURNS_0,
        OFF_META,
        OFF_MOVES,
        OFF_SIDE0,
        OFF_SIDE1,
        POKEMON_SIZE,
    )
    from pokepy.engine.battle_gen9 import step_battle_gen9
    from pokepy.env import init_battle_state
    from pokepy.utils.gen5_prng import Gen5PRNG

    gd = load_game_data()
    me = load_move_effect_data()

    battle_seed = 1805130656
    state = init_battle_state(
        pool_get_team(13443), pool_get_team(12183), gd, seed=battle_seed
    )
    prng = Gen5PRNG((battle_seed & 0xFFFF, (battle_seed >> 16) & 0xFFFF, 0, 0))

    battle_arr = state.battle_state
    for slot in range(6):
        spe0 = int(battle_arr[OFF_SIDE0 + slot * POKEMON_SIZE + 11])
        spe1 = int(battle_arr[OFF_SIDE1 + slot * POKEMON_SIZE + 11])
        if spe0 == spe1:
            prng.random(0, 2)
    if int(battle_arr[OFF_SIDE0 + 11]) == int(battle_arr[OFF_SIDE1 + 11]):
        prng.random(0, 2)
        prng.random(0, 2)
        prng.random(0, 2)

    # Scripted Battle 374 opening:
    # T1: Dragon Dance vs switch Dragonite
    # T2: Outrage KOs Dragonite; side 1 auto-switches Slither Wing
    # T3: repeated Outrage into incoming Weezing-Galar immunity
    scripted_turns = [
        (3, 7),  # move 4 / switch 4
        (1, 2),  # move 2 / move 3
        (1, 9),  # move 2 / switch 6
    ]
    for a0, a1 in scripted_turns:
        _, _, done = step_battle_gen9(state, a0, a1, gd, me, MODERN_TYPE_CHART, prng)
        assert not done

    # Showdown clears lockedmove after the immune repeat turn, so the next turn
    # is a normal manual choice rather than a forced Outrage continuation.
    assert int(state.battle_state[OFF_MOVES + M_LOCKED_MOVE_0]) == -1
    assert int(state.battle_state[OFF_MOVES + M_LOCKED_TURNS_0]) == 0

    mask = get_battle_action_mask(state, 0, gd)
    assert mask.tolist()[:4] == [True, True, True, True]

    # If the stale lock persists, step_battle_gen9 overrides this Earthquake
    # choice back into Outrage and the HP row diverges.
    _, _, done = step_battle_gen9(state, 2, 1, gd, me, MODERN_TYPE_CHART, prng)
    assert not done

    active0 = int(state.battle_state[OFF_META + M_ACTIVE0])
    active1 = int(state.battle_state[OFF_META + M_ACTIVE1])
    p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
    p1_off = OFF_SIDE1 + active1 * POKEMON_SIZE

    assert int(state.battle_state[p0_off + 1]) == 149
    assert int(state.battle_state[p1_off + 1]) == 85


def test_battle375_mid_turn_pivot_replacement_hp_caps_slower_recoil_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42179),
        pool_get_team(42706),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1369211308,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[27]["p0_hp"] == 92
    assert normal_rows[27]["p1_hp"] == 113

    # The slower Brave Bird hits the Flip Turn replacement, so recoil must be
    # capped by Great Tusk's live post-hazard HP rather than a projection of
    # Alomomola's postmove chain.
    assert normal_rows[28]["p0_hp"] == 394
    assert normal_rows[28]["p0_max_hp"] == 394
    assert normal_rows[28]["p1_hp"] == 62
    assert normal_rows[28]["p1_max_hp"] == 371

    assert normal_rows[29]["p0_hp"] == 301
    assert normal_rows[29]["p0_max_hp"] == 394
    assert normal_rows[29]["p1_hp"] == 62


def test_battle381_cursed_body_preroll_before_slower_move_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42223),
        pool_get_team(16555),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1036316571,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[40]["p0_hp"] == 301
    assert normal_rows[40]["p1_hp"] == 317

    # Dragapult's Cursed Body roll is consumed right after Sucker Punch lands,
    # before Dragapult starts Draco Meteor, so the slower move reads the same
    # accuracy / crit / damage frames as Showdown.
    assert normal_rows[41]["p0_hp"] == 57
    assert normal_rows[41]["p0_max_hp"] == 301
    assert normal_rows[41]["p1_hp"] == 9
    assert normal_rows[41]["p1_max_hp"] == 317


def test_battle381_midturn_forced_switch_sd_pos_sync_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    team0 = pool_get_team(42223)
    team1 = pool_get_team(16555)
    battle_seed = 1036316571
    seed_tuple = (battle_seed & 0xFFFF, (battle_seed >> 16) & 0xFFFF, 0, 0)

    rows, p0_actions, p1_actions = run_pokepy_battle(
        team0,
        team1,
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        battle_seed,
        95,
    )

    show_rows, raw_data, meta = run_showdown(
        seed_tuple,
        team_to_showdown_packed(team0, mappings),
        team_to_showdown_packed(team1, mappings),
        p0_actions,
        p1_actions,
        95,
        timeout_s=120,
    )

    assert meta["returncode"] == 0
    assert not meta["timeout"]
    assert meta["error"] is None
    assert raw_data is not None
    assert show_rows

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}
    by_turn = {row["turn"]: row for row in show_rows}

    assert normal_rows[84]["p0_action"] == "move 2+forced:switch 5+forced:switch 3"
    assert normal_rows[84]["p0_hp"] == 83
    assert by_turn[84]["p0_hp"] == 83
    assert by_turn[84]["p1_hp"] == 325

    # After Roar drags Umbreon into Stealth Rock, the inline forced-switch
    # labels must be serialized against Showdown's post-drag slot order.
    # If sd_pos stays stale here, the queued fallback switch token is reused on
    # turn 85 and Showdown re-switches through Gengar, dropping Dragonite to 3.
    assert normal_rows[85]["p0_hp"] == 83
    assert by_turn[85]["p0_hp"] == 83
    assert by_turn[85]["p1_hp"] == 394


def test_battle428_court_change_orders_before_slower_hazard_setter_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(19090),
        pool_get_team(2448),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1949523082,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[34]["p0_hp"] == 101
    assert normal_rows[34]["p1_hp"] == 88
    assert normal_rows[35]["p0_hp"] == 126
    assert normal_rows[35]["p1_hp"] == 88

    # Smeargle's faster Court Change must swap hazards before Garganacl's
    # slower Stealth Rock evaluates its target side. If Stealth Rock resolves
    # against the pre-swap field, Kingambit later re-enters into stray rocks
    # and drops from 190 to 165 HP.
    assert normal_rows[36]["p0_hp"] == 151
    assert normal_rows[36]["p0_max_hp"] == 403
    assert normal_rows[36]["p1_hp"] == 190
    assert normal_rows[36]["p1_max_hp"] == 404


def test_battle66_red_card_drag_and_faster_drain_sash_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(16947),
        pool_get_team(48094),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        786544956,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Mimikyu's Red Card must drag Ceruledge out immediately after Bitter
    # Blade so the slower Shadow Claw hits the replacement Gliscor instead of
    # failing or reusing stale Ceruledge state.
    assert normal_rows[20]["p0_hp"] == 275
    assert normal_rows[20]["p0_max_hp"] == 314
    assert normal_rows[20]["p1_hp"] == 286
    assert normal_rows[20]["p1_max_hp"] == 353

    # When Ceruledge later moves first with Bitter Blade, the slower Kowtow
    # Cleave must hit Ceruledge's drained-to-full live HP. That lets Focus
    # Sash save it at 1 instead of forcing an incorrect switch to Ninetales.
    assert normal_rows[34]["p0_hp"] == 65
    assert normal_rows[34]["p0_max_hp"] == 404
    assert normal_rows[34]["p1_hp"] == 1
    assert normal_rows[34]["p1_max_hp"] == 291
    assert normal_rows[35]["p1_hp"] == 34
    assert normal_rows[35]["p1_max_hp"] == 291


def test_battle47_faster_drain_replay_uses_live_slower_hit_hp_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(10729),
        pool_get_team(13827),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        877308582,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Comfey's faster Draining Kiss must heal to full, then lose only Life
    # Orb HP before the slower Hurricane miss. Replaying the slower move from
    # the turn-start HP instead of Comfey's live post-drain HP wrongly drops
    # it back to 249 instead of the Showdown-correct 275.
    assert normal_rows[11]["p1_hp"] == 249
    assert normal_rows[12]["p0_hp"] == 168
    assert normal_rows[12]["p1_hp"] == 275
    assert normal_rows[12]["p1_max_hp"] == 305
    assert normal_rows[12]["p1_status"] == 2
    assert normal_rows[13]["p1_hp"] == 275


def test_battle28_red_card_does_not_drag_on_switch_turn_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(31488),
        pool_get_team(35013),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        208792607,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Side 1's opening voluntary switch must stay on Walking Wake after
    # Glimmora sets Stealth Rock. Red Card only triggers when its holder is
    # damaged by a move, so the switch turn must not drag Walking Wake back
    # out into Great Tusk.
    assert normal_rows[1]["p0_hp"] == 307
    assert normal_rows[1]["p1_action"] == "switch 4"
    assert normal_rows[1]["p1_hp"] == 339
    assert normal_rows[1]["p1_max_hp"] == 339
    assert normal_rows[2]["p1_hp"] == 339


def test_battle385_disablemove_shuffle_respects_showdown_suborders_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(29356),
        pool_get_team(6008),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1825950153,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Volcanion's Assault Vest and Psychic Noise's Heal Block both participate
    # in Showdown's request-time DisableMove pass, but they use different
    # subOrders (item vs condition) and therefore must not consume a hidden
    # shuffle frame against each other. If pokepy flattens them into one tie
    # group, the next Earth Power damage roll shifts and Primarina survives at
    # 51 instead of the Showdown-correct 40.
    assert normal_rows[17]["p0_hp"] == 144
    assert normal_rows[17]["p1_hp"] == 141
    assert normal_rows[18]["p0_hp"] == 41
    assert normal_rows[18]["p0_max_hp"] == 301
    assert normal_rows[18]["p1_hp"] == 40
    assert normal_rows[18]["p1_max_hp"] == 321


def test_battle641_same_turn_spikes_materialize_before_eject_pack_switch_in_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(21728),
        pool_get_team(18068),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        282941349,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Meowscarada's turn-1 Spikes must already exist when Raging Bolt's
    # Draco Meteor triggers Eject Pack. Showdown applies the fresh Spikes chip
    # during Kingambit's same-turn switch-in, leaving it at 354/404 instead of
    # the stale 404/404 state pokepy had when hazards were only materialized
    # in the late ordered field-effects bucket.
    assert normal_rows[1]["p0_hp"] == 321
    assert normal_rows[1]["p0_max_hp"] == 321
    assert normal_rows[1]["p1_hp"] == 354
    assert normal_rows[1]["p1_max_hp"] == 404


def test_battle390_item_forced_switch_skips_ko_only_move2_update_frame_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(31448),
        pool_get_team(31541),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        840747732,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Alomomola's same-turn Eject Button switch after Vaporeon's Shadow Ball
    # must reuse the holder's tied switch-action speed for the hidden resume
    # frames, but it must NOT also spend the extra KO-only move-2 update frame
    # from the faint-replacement path. If pokepy burns that stale
    # Alomomola/Vaporeon tie frame, turn 30's First Impression and turn 33's
    # Lava Plume damage rolls drift and Kingambit incorrectly survives.
    assert normal_rows[29]["p0_action"] == "move 3+forced:switch 4"
    assert normal_rows[30]["p0_action"] == "move 4+forced:switch 2"
    assert normal_rows[30]["p0_active"] == 0
    assert normal_rows[30]["p1_hp"] == 191
    assert normal_rows[31]["p1_action"] == "move 3+forced:switch 5"
    assert normal_rows[33]["p1_action"] == "switch 2+die_switch:switch 2"
    assert normal_rows[33]["p1_active"] == 1
    assert normal_rows[33]["p1_hp"] == 334
    assert normal_rows[33]["p1_max_hp"] == 334


def test_battle232_eject_button_canceled_move_skips_between_move_update_frames_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36092),
        pool_get_team(29503),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        107688083,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # When faster Alomomola's Flip Turn triggers the target's Eject Button,
    # the slower queued Scald never reaches runMove. Showdown therefore skips
    # the tied between-moves Update frames that normally happen before the
    # slower move. If pokepy still burns those three frames, the later Hex
    # damage roll shifts and Great Tusk lands at 196 instead of 210.
    assert normal_rows[9]["p0_action"] == "move 2"
    assert normal_rows[9]["p1_action"] == "move 1+forced:switch 3"
    assert normal_rows[10]["p0_active"] == 3
    assert normal_rows[10]["p0_hp"] == 363
    assert normal_rows[11]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[12]["p0_active"] == 2
    assert normal_rows[12]["p0_hp"] == 210
    assert normal_rows[12]["p0_max_hp"] == 371
    assert normal_rows[12]["p1_hp"] == 315


def test_battle398_rapid_spin_does_not_clear_hazards_after_rocky_helmet_ko_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(26291),
        pool_get_team(31688),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1532190156,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown gates Rapid Spin / Mortal Spin's hazard clear on the spinner
    # still being alive in `onAfterHit`. Great Tusk faints to Landorus's
    # Rocky Helmet before that hook, so Stealth Rock stays up and both the
    # first fallback Ribombee and the eventual Ceruledge replacement take the
    # switch-in chip immediately.
    assert normal_rows[13]["p1_action"] == "move 2+forced:switch 5+die_switch:switch 2"
    assert normal_rows[13]["p1_active"] == 1
    assert normal_rows[13]["p1_hp"] == 53
    assert normal_rows[13]["p1_max_hp"] == 354
    assert normal_rows[14]["p1_action"] == "switch 4"
    assert normal_rows[14]["p1_active"] == 3
    assert normal_rows[14]["p1_hp"] == 296
    assert normal_rows[14]["p1_max_hp"] == 315


def test_battle418_trick_room_expires_before_glimmora_can_reverse_turn_order_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(32491),
        pool_get_team(4240),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        476586935,
        80,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Scream Tail's third Trick Room on turn 13 should age out at order 27 on
    # turn 18. If the custom residual path forgets to decrement it, Glimmora
    # incorrectly keeps moving first later on, KOs Tentacruel with Earth
    # Power, and the whole late-game switch chain drifts. The correct state is
    # that Tentacruel stays active through Glimmora's switch-in and KOs it with
    # Hydro Pump before it can move on turn 24.
    assert normal_rows[23]["p0_action"] == "move 4"
    assert normal_rows[23]["p0_active"] == 5
    assert normal_rows[23]["p0_hp"] == 163
    assert normal_rows[23]["p1_action"] == "switch 4"
    assert normal_rows[23]["p1_active"] == 3
    assert normal_rows[23]["p1_hp"] == 61
    assert normal_rows[24]["p0_action"] == "move 4"
    assert normal_rows[24]["p0_active"] == 5
    assert normal_rows[24]["p0_hp"] == 163
    assert normal_rows[24]["p1_action"] == "move 3+forced:switch 4"
    assert normal_rows[24]["p1_active"] == 0
    assert normal_rows[24]["p1_hp"] == 261
    assert normal_rows[24]["p1_max_hp"] == 261


def test_battle420_crit_rebuilds_attacker_stat_after_intimidate_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(13304),
        pool_get_team(46235),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        954499887,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Corviknight's turn-2 Brave Bird crit should ignore Landorus-Therian's
    # Intimidate drop by rebuilding the attack stat at boost 0 before the
    # ModifyAtk chain. If pokepy rescales the already-modified atk stat
    # instead, the crit under-damages by 2 and Brave Bird recoil under-chips
    # by 1 on the same turn, drifting the rest of the game immediately.
    assert normal_rows[1]["p0_hp"] == 312
    assert normal_rows[1]["p1_hp"] == 376
    assert normal_rows[2]["p0_action"] == "move 1"
    assert normal_rows[2]["p0_hp"] == 171
    assert normal_rows[2]["p0_max_hp"] == 382
    assert normal_rows[2]["p1_action"] == "move 4"
    assert normal_rows[2]["p1_hp"] == 329
    assert normal_rows[2]["p1_max_hp"] == 399
    assert normal_rows[3]["p0_hp"] == 105
    assert normal_rows[3]["p1_hp"] == 307


def test_battle425_toxic_damage_respects_neutralizing_gas_suppression_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(35662),
        pool_get_team(32851),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1661552514,
        100,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Reuniclus switches into Toxic Spikes while opposing Weezing-Galar keeps
    # Neutralizing Gas active. The gas suppresses Magic Guard for turn 57, so
    # Reuniclus should take its toxic tick after Leftovers, then stop taking
    # poison damage again on turn 58 once Weezing switches out and Magic Guard
    # comes back online.
    assert normal_rows[56]["p0_action"] == "switch 5"
    assert normal_rows[56]["p0_hp"] == 327
    assert normal_rows[56]["p1_action"] == "move 2"
    assert normal_rows[56]["p1_hp"] == 66
    assert normal_rows[57]["p0_action"] == "move 4"
    assert normal_rows[57]["p0_hp"] == 334
    assert normal_rows[57]["p1_action"] == "switch 6"
    assert normal_rows[57]["p1_hp"] == 352
    assert normal_rows[57]["p1_max_hp"] == 424
    assert normal_rows[57]["p1_status"] == 6
    assert normal_rows[58]["p0_action"] == "switch 5"
    assert normal_rows[58]["p1_action"] == "move 1"
    assert normal_rows[58]["p1_hp"] == 378
    assert normal_rows[58]["p1_status"] == 6


def test_battle426_triple_axel_stops_after_ko_without_burning_later_hit_frames_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(406),
        pool_get_team(43272),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1588847798,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown's hitStepMoveHitLoop stops Triple Axel as soon as the second
    # hit KOs Landorus-Therian. pokepy used to pre-roll the third hit's crit
    # + damage and a stale later-hit accuracy check anyway, which kept the
    # visible board state intact on turn 6 but shifted Dragapult's later
    # Draco Meteor miss and Clodsire's Earthquake roll. The correct strict
    # state is that turn 6 ends with Dragapult at 278, then turn 8 leaves it
    # at 169 after the missed Draco Meteor into Clodsire.
    assert normal_rows[6]["p0_action"] == "move 4+forced:switch 5"
    assert normal_rows[6]["p0_active"] == 4
    assert normal_rows[6]["p0_hp"] == 278
    assert normal_rows[6]["p0_max_hp"] == 317
    assert normal_rows[6]["p1_action"] == "move 2"
    assert normal_rows[6]["p1_hp"] == 291
    assert normal_rows[7]["p0_action"] == "move 4"
    assert normal_rows[7]["p1_action"] == "move 2+forced:switch 4"
    assert normal_rows[7]["p1_hp"] == 184
    assert normal_rows[8]["p0_action"] == "move 4"
    assert normal_rows[8]["p0_hp"] == 169
    assert normal_rows[8]["p0_max_hp"] == 317
    assert normal_rows[8]["p1_action"] == "move 2"
    assert normal_rows[8]["p1_hp"] == 213


def test_battle530_miracle_berry_cures_lockedmove_fatigue_confusion():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36794),
        pool_get_team(17323),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        823205125,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Dragonite's first Outrage chain ends around this state and applies
    # fatigue confusion. Showdown's Miracle Berry immediately eats and removes
    # that confusion, preventing a hidden BeforeMove self-hit roll from
    # shifting later damage RNG; the first visible check is turn 39 Earthquake
    # damage.
    assert normal_rows[25]["p0_action"] == "move 1"
    assert normal_rows[25]["p0_hp"] == 131
    assert normal_rows[25]["p0_max_hp"] == 514
    assert normal_rows[25]["p1_action"] == "move 2"
    assert normal_rows[25]["p1_hp"] == 351
    assert normal_rows[25]["p1_max_hp"] == 351
    assert normal_rows[39]["p0_action"] == "move 1"
    assert normal_rows[39]["p0_hp"] == 283
    assert normal_rows[39]["p0_max_hp"] == 394
    assert normal_rows[39]["p1_action"] == "move 4"
    assert normal_rows[39]["p1_hp"] == 0
    assert normal_rows[39]["p1_max_hp"] == 351


def test_battle959_eject_button_regenerator_hp_persists_on_bench():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(46808),
        pool_get_team(43556),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2098505408,
        8,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 1 Bite damages Alomomola and Eject Button switches it out. Showdown
    # later runs the switch-out path, so Regenerator heals the stored bench HP
    # before Alomomola is forced back in on turn 6.
    assert normal_rows[1]["p0_action"] == "move 3"
    assert normal_rows[1]["p1_action"] == "move 2+forced:switch 2"
    assert normal_rows[1]["p1_active"] == 1
    assert normal_rows[1]["p1_hp"] == 318
    assert normal_rows[1]["p1_max_hp"] == 318
    assert normal_rows[6]["p0_action"] == "move 1"
    assert normal_rows[6]["p0_hp"] == 281
    assert normal_rows[6]["p0_max_hp"] == 384
    assert normal_rows[6]["p1_action"] == "move 1+forced:switch 2"
    assert normal_rows[6]["p1_active"] == 0
    assert normal_rows[6]["p1_hp"] == 532
    assert normal_rows[6]["p1_max_hp"] == 532


def test_battle433_effect_spore_sleep_uses_standard_duration_roll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(1480),
        pool_get_team(37982),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        861904496,
        16,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown routes Effect Spore sleep through the normal setStatus('slp')
    # path, so turn-2 sleep uses the standard random(2, 5) duration and its
    # hidden onStart frame lands before the slower Leech Seed accuracy check.
    # If pokepy hardcodes the stored duration or lets Leech Seed steal that
    # extra frame first, Meowscarada wakes too early and the residual loop
    # drifts immediately.
    assert normal_rows[3]["p0_hp"] == 224
    assert normal_rows[3]["p1_status"] == 3
    assert normal_rows[4]["p0_hp"] == 260
    assert normal_rows[4]["p1_hp"] == 101
    assert normal_rows[4]["p1_status"] == 3
    assert normal_rows[5]["p0_hp"] == 296
    assert normal_rows[5]["p1_hp"] == 65
    assert normal_rows[5]["p1_status"] == 3
    assert normal_rows[6]["p0_hp"] == 251
    assert normal_rows[6]["p1_hp"] == 29
    assert normal_rows[6]["p1_status"] == 0


def test_battle445_hidden_startup_and_eject_button_frames_stay_in_sync():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49937),
        pool_get_team(31912),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        636432671,
        5,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown spends hidden PRNG frames in battle startup before turn 1:
    # team-preview queue.sort, tied lead runSwitch insertion, tied startup
    # runSwitch speedSort, Drought's WeatherChange speedSort, and the startup
    # post-runSwitch Update. If pokepy only replays the preview frame, the
    # first visible Scald and Ice Beam damage rolls drift immediately.
    assert normal_rows[1]["p0_action"] == "move 3"
    assert normal_rows[1]["p0_hp"] == 341
    assert normal_rows[1]["p1_action"] == "switch 3"
    assert normal_rows[1]["p1_hp"] == 274
    assert normal_rows[1]["p1_max_hp"] == 301
    assert normal_rows[2]["p0_action"] == "move 4"
    assert normal_rows[2]["p0_hp"] == 341
    assert normal_rows[2]["p1_action"] == "move 1"
    assert normal_rows[2]["p1_hp"] == 102
    # Turn 4's Scald burns Hatterene and its Eject Button forces a same-turn
    # auto-switch to Ninetales. Showdown spends four additional hidden
    # tied-speed switch / runSwitch frames there; without them, turn 5's Scald
    # damage roll drifts even though the visible startup frames already match.
    assert normal_rows[5]["p0_action"] == "move 3"
    assert normal_rows[5]["p0_hp"] == 341
    assert normal_rows[5]["p1_action"] == "switch 4"
    assert normal_rows[5]["p1_hp"] == 181
    assert normal_rows[5]["p1_max_hp"] == 318
    assert normal_rows[5]["p1_status"] == 1
    assert normal_rows[2]["p1_max_hp"] == 301


def test_battle439_toxic_chain_respects_purifying_salt_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(29516),
        pool_get_team(22448),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        852422304,
        22,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Okidogi's Toxic Chain still spends its 30% chance roll on Garganacl, but
    # Purifying Salt blocks the bad poison inside Showdown's shared
    # trySetStatus gate. If pokepy writes tox directly from the ability path,
    # turn 19 drifts immediately and the later Kingambit pivots desync.
    assert normal_rows[19]["p0_action"] == "move 2"
    assert normal_rows[19]["p1_action"] == "switch 6"
    assert normal_rows[19]["p1_hp"] == 106
    assert normal_rows[19]["p1_max_hp"] == 404
    assert normal_rows[19]["p1_status"] == 0
    assert normal_rows[20]["p1_hp"] == 253
    assert normal_rows[20]["p1_status"] == 0


def test_battle448_static_respects_comatose_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(9334),
        pool_get_team(49879),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1837545187,
        4,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Komala's Knock Off makes contact with Zapdos, so Static still gets to
    # roll after the hit. Showdown then interrupts the paralysis because
    # Comatose leaves Komala permanently-asleep and immune to new status.
    assert normal_rows[2]["p0_action"] == "move 1"
    assert normal_rows[2]["p0_hp"] == 216
    assert normal_rows[2]["p0_max_hp"] == 334
    assert normal_rows[2]["p0_status"] == 0
    assert normal_rows[2]["p1_hp"] == 240
    assert normal_rows[3]["p0_status"] == 0
    assert normal_rows[3]["p1_hp"] == 183


def test_battle448_ice_beam_freeze_respects_ice_type_immunity_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(9334),
        pool_get_team(49879),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1837545187,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Mesprit's Ice Beam still spends its freeze secondary on Mamoswine at turn
    # 21, but Showdown rejects the status with "natural status immunity"
    # because Ice-types cannot be frozen. Pokepy used to land the freeze and
    # only clear it later when Mamoswine fainted on the next turn.
    assert normal_rows[20]["p0_action"] == "move 4"
    assert normal_rows[20]["p1_action"] == "move 4"
    assert normal_rows[20]["p1_hp"] == 114
    assert normal_rows[20]["p1_status"] == 0
    assert normal_rows[21]["p1_hp"] == 32
    assert normal_rows[21]["p1_status"] == 0


def test_battle448_failed_hyper_beam_does_not_force_recharge_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(9334),
        pool_get_team(49879),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1837545187,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Porygon-Z's Hyper Beam into Drifblim fails on turn 25 because Normal is
    # immune into Ghost. Showdown does not add mustrecharge after that failed
    # hit, so turn 26 is a normal Tera Blast turn instead of a fake recharge
    # skip. The shared recharge fix also covers plain misses and Protect.
    assert normal_rows[25]["p0_action"] == "move 2"
    assert normal_rows[25]["p1_action"] == "move 3"
    assert normal_rows[25]["p0_hp"] == 504
    assert normal_rows[25]["p0_max_hp"] == 504
    assert normal_rows[25]["p1_hp"] == 197
    assert normal_rows[25]["p1_max_hp"] == 311
    assert normal_rows[26]["p0_action"] == "move 2"
    assert normal_rows[26]["p1_action"] == "move 4"
    assert normal_rows[26]["p0_hp"] == 158
    assert normal_rows[26]["p1_hp"] == 147


def test_battle466_effect_spore_powder_immunity_skips_roll_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(1250),
        pool_get_team(14969),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1835269303,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Ogerpon-Wellspring's Knock Off into Vileplume still makes contact, but
    # Showdown skips Effect Spore's random(100) because Grass-types are
    # naturally immune to powder. Spending that bogus frame drifts the later
    # Magma Storm damage roll and immediately corrupts turn 19.
    assert normal_rows[16]["p0_action"] == "switch 2"
    assert normal_rows[16]["p1_action"] == "move 4"
    assert normal_rows[19]["p0_hp"] == 386
    assert normal_rows[19]["p0_max_hp"] == 386
    assert normal_rows[19]["p1_hp"] == 187
    assert normal_rows[19]["p1_max_hp"] == 379
    assert normal_rows[20]["p0_hp"] == 355
    assert normal_rows[20]["p1_hp"] == 140


def test_battle469_game_ending_weather_ko_skips_later_leftovers_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(43715),
        pool_get_team(76),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        608268157,
        45,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Slowking survives Tyranitar's Earthquake at 24 HP on turn 41, hits back
    # with Ice Beam, then faints to sandstorm before the later Leftovers slot.
    # Showdown ends the battle immediately there, so Tyranitar must stay at
    # 228 instead of collecting a bogus 25 HP heal.
    assert normal_rows[40]["p0_action"] == "move 4+forced:switch 4"
    assert normal_rows[40]["p1_action"] == "move 2"
    assert normal_rows[41]["p0_action"] == "move 4"
    assert normal_rows[41]["p1_action"] == "move 3"
    assert normal_rows[41]["p0_hp"] == 228
    assert normal_rows[41]["p0_max_hp"] == 404
    assert normal_rows[41]["p1_hp"] == 0
    assert normal_rows[41]["p1_max_hp"] == 393


def test_battle14_slower_brave_bird_uses_post_helmet_target_hp_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(35026),
        pool_get_team(17726),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        145857092,
        42,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Dragonite moves first on turns 10 and 11, takes Rocky Helmet chip, and
    # only then gets hit by Corviknight's slower Brave Bird. Showdown bases the
    # slower recoil on Dragonite's live post-Helmet HP (211, then 46), not on
    # the pre-turn snapshot.
    assert normal_rows[10]["p0_action"] == "move 2"
    assert normal_rows[10]["p1_action"] == "move 1"
    assert normal_rows[10]["p0_hp"] == 220
    assert normal_rows[10]["p1_hp"] == 100
    assert normal_rows[11]["p0_action"] == "move 2"
    assert normal_rows[11]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[11]["p0_hp"] == 138
    assert normal_rows[11]["p1_hp"] == 434


def test_battle481_slower_draining_kiss_uses_post_drain_target_hp_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(21648),
        pool_get_team(6983),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        76971722,
        27,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Wo-Chien's Giga Drain heals it from 30 to 73 before Hatterene's slower
    # Draining Kiss lands. Showdown bases the slower drain heal on that live
    # 73 HP snapshot, so Hatterene heals from 233 to 288 instead of using the
    # pre-drain 30 HP cap.
    assert normal_rows[21]["p0_hp"] == 30
    assert normal_rows[21]["p1_hp"] == 318
    assert normal_rows[22]["p0_action"] == "move 4"
    assert normal_rows[22]["p1_action"] == "move 1"
    assert normal_rows[22]["p0_hp"] == 0
    assert normal_rows[22]["p0_max_hp"] == 374
    assert normal_rows[22]["p1_hp"] == 288
    assert normal_rows[22]["p1_max_hp"] == 318


def test_battle441_slower_move_after_hard_switch_hits_new_active_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(44327),
        pool_get_team(43317),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        402854364,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # After Alomomola hard-switches back in on turn 32, Blastoise's slower Surf
    # must still hit the new active. Pokepy used to compute the 103 damage
    # correctly, then cancel the move because the switch-side branch left the
    # slower attacker's placeholder `new_hp1` at zero before `p1_move_ran`.
    assert normal_rows[32]["p0_action"] == "switch 5"
    assert normal_rows[32]["p1_action"] == "move 1"
    assert normal_rows[32]["p0_hp"] == 431
    assert normal_rows[32]["p0_max_hp"] == 534
    assert normal_rows[32]["p1_hp"] == 126
    assert normal_rows[33]["p0_hp"] == 307


def test_battle488_electromorphosis_charge_boost_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42599),
        pool_get_team(5580),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1364862092,
        72,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Bellibolt gets Charge from Electromorphosis on turn 39, then its next
    # Volt Switch must consume that volatile and hit Dusclops for 115 damage.
    assert normal_rows[39]["p0_action"] == "switch 6"
    assert normal_rows[39]["p0_hp"] == 237
    assert normal_rows[40]["p0_action"] == "move 3+pivot:switch 6"
    assert normal_rows[40]["p1_action"] == "switch 3"
    assert normal_rows[40]["p1_hp"] == 168
    assert normal_rows[40]["p1_status"] == 3


def test_battle488_switched_in_sleep_does_not_tick_before_move_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42599),
        pool_get_team(5580),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1364862092,
        72,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Dusclops re-enters asleep on turn 58. Showdown does not tick sleep on
    # that switch turn, so it stays at 168 HP / slp through turn 59's `cant`.
    assert normal_rows[58]["p1_action"] == "switch 4"
    assert normal_rows[58]["p1_hp"] == 168
    assert normal_rows[58]["p1_status"] == 3
    assert normal_rows[59]["p1_action"] == "move 1"
    assert normal_rows[59]["p1_hp"] == 168
    assert normal_rows[59]["p1_status"] == 3


def test_battle88_paradox_stat_flag_does_not_act_as_charge_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49484),
        pool_get_team(29053),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        960007121,
        29,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Raging Bolt's Protosynthesis best-stat marker shares the same packed flag
    # word as Charge. A bit collision used to make turn 14 Thunderbolt behave
    # as if Charge were active, which KOed Serperior and pulled the forced
    # switch sequence a turn early.
    assert normal_rows[14]["p0_action"] == "move 2"
    assert normal_rows[14]["p0_hp"] == 85
    assert normal_rows[14]["p0_max_hp"] == 291
    assert normal_rows[14]["p1_action"] == "move 4"
    assert normal_rows[14]["p1_hp"] == 55
    assert normal_rows[14]["p1_max_hp"] == 391


def test_battle748_tied_speed_inline_pivot_resume_uses_switch_request_frames_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(35543),
        pool_get_team(41626),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1236164676,
        28,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 23, Zapdos's Volt Switch brings in Alomomola before the opposing
    # Alomomola uses Flip Turn. Showdown treats that immediate resume as a
    # switch-request continuation, so the tied-speed switch-in burns the 3
    # hidden Update/runSwitch/Update frames before Flip Turn's crit + damage.
    assert normal_rows[23]["p0_action"] == "move 2+pivot:switch 3"
    assert normal_rows[23]["p1_action"] == "move 1"
    assert normal_rows[23]["p0_hp"] == 371
    assert normal_rows[23]["p0_max_hp"] == 534
    assert normal_rows[23]["p1_hp"] == 120
    assert normal_rows[23]["p1_max_hp"] == 534


def test_battle512_confused_second_mover_does_not_preroll_blocked_curse_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(7677),
        pool_get_team(5531),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1516687854,
        90,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 49, confused Dondozo self-hits instead of using Curse. Showdown
    # does not spend Curse's non-Ghost selfDrops PRNG after that blocked move,
    # so turn 50's Malignant Chain roll stays high enough for toxic to finish
    # Dondozo and immediately bring in Alomomola through the same-turn
    # residual replacement path.
    assert normal_rows[49]["p1_hp"] == 136
    assert normal_rows[49]["p1_max_hp"] == 504
    assert normal_rows[49]["p1_status"] == 6
    assert normal_rows[50]["p0_action"] == "move 2"
    assert normal_rows[50]["p1_action"] == "move 1+forced:switch 6"
    assert normal_rows[50]["p1_hp"] == 401
    assert normal_rows[50]["p1_max_hp"] == 534
    assert normal_rows[50]["p1_status"] == 0


def test_battle516_kee_berry_on_damaging_hit_reduces_future_pyro_balls_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(24699),
        pool_get_team(237),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        602727458,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 3's first Pyro Ball makes Latias eat Kee Berry and gain +1 Def, so
    # later resisted Pyro Balls follow the reduced-damage line instead of the
    # pre-berry one.
    assert normal_rows[12]["p0_hp"] == 240
    assert normal_rows[13]["p0_hp"] == 185
    assert normal_rows[14]["p0_hp"] == 131
    assert normal_rows[15]["p0_hp"] == 78
    assert normal_rows[16]["p0_hp"] == 25
    assert normal_rows[16]["p1_hp"] == 255
    assert normal_rows[17]["p0_hp"] == 301
    assert normal_rows[17]["p0_max_hp"] == 301
    assert normal_rows[17]["p1_hp"] == 227


def test_battle537_fractional_custap_priority_is_not_blocked_by_psychic_terrain_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(7185),
        pool_get_team(33511),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1488411095,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Custap Berry makes Araquanid's Surf win the +0 tie-break on turn 7, but
    # it stays in the +0 bracket. Psychic Terrain must not block it as if it
    # were a true positive-priority move.
    assert normal_rows[7]["p0_hp"] == 43
    assert normal_rows[7]["p0_max_hp"] == 340
    assert normal_rows[7]["p1_hp"] == 253
    assert normal_rows[7]["p1_max_hp"] == 714
    assert normal_rows[8]["p0_hp"] == 19
    assert normal_rows[8]["p1_hp"] == 102
    assert normal_rows[9]["p0_hp"] == 271
    assert normal_rows[9]["p0_max_hp"] == 289
    assert normal_rows[9]["p1_hp"] == 102


def test_battle530_lockedmove_expiry_confusion_rolls_even_on_immune_hit_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36794),
        pool_get_team(17323),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        823205125,
        45,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Dragonite's final locked Outrage turn hits Clefable's Fairy immunity, but
    # Showdown still ends lockedmove and rolls fatigue confusion before
    # Clefable's Ice Beam. Missing that random(2, 6) frame drifts the damage
    # roll and under-damages Dragonite by 2 HP.
    assert normal_rows[36]["p0_hp"] == 394
    assert normal_rows[36]["p0_max_hp"] == 394
    assert normal_rows[36]["p1_hp"] == 351
    assert normal_rows[36]["p1_max_hp"] == 351
    assert normal_rows[37]["p0_hp"] == 394
    assert normal_rows[37]["p0_max_hp"] == 394
    assert normal_rows[37]["p1_hp"] == 211
    assert normal_rows[37]["p1_max_hp"] == 351
    assert normal_rows[38]["p0_hp"] == 394
    assert normal_rows[38]["p1_hp"] == 211


def test_battle514_fainted_slower_lockedmove_user_does_not_spend_post_move_frame_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4442),
        pool_get_team(28460),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        495455159,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Garchomp is still locked into Outrage on turn 43, but Meowscarada KOs it
    # before that forced move can run. Showdown does not spend any post-move
    # lockedmove confusion frame on the fainted slower attacker, so turn 44's
    # Play Rough into Heatran must use the unshifted accuracy / crit / damage
    # rolls.
    assert normal_rows[43]["p0_hp"] == 257
    assert normal_rows[43]["p0_max_hp"] == 293
    assert normal_rows[43]["p1_hp"] == 423
    assert normal_rows[43]["p1_max_hp"] == 423
    assert normal_rows[44]["p0_hp"] == 257
    assert normal_rows[44]["p1_hp"] == 339
    assert normal_rows[44]["p1_max_hp"] == 386
    assert normal_rows[45]["p0_hp"] == 512
    assert normal_rows[45]["p1_hp"] == 339


def test_battle525_prism_armor_reduces_crit_super_effective_damage_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(15919),
        pool_get_team(27408),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1716297970,
        100,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown still applies Prism Armor's 0.75x modifier on a critical
    # super-effective Shadow Ball, so Necrozma survives at 122/335 before the
    # next turn's switch.
    assert normal_rows[57]["p0_hp"] == 341
    assert normal_rows[57]["p1_hp"] == 335
    assert normal_rows[58]["p0_hp"] == 341
    assert normal_rows[58]["p1_hp"] == 122
    assert normal_rows[58]["p1_max_hp"] == 335
    assert normal_rows[59]["p0_hp"] == 341
    assert normal_rows[59]["p1_hp"] == 307


def test_battle529_future_sight_keeps_live_ruin_aura_when_source_is_off_field_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(3905),
        pool_get_team(43093),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        64069680,
        100,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Slowking's delayed Future Sight resolves while Ting-Lu is the current
    # ally on the field, so Vessel of Ruin still lowers the special damage into
    # the newly switched Ogerpon-Wellspring.
    assert normal_rows[18]["p0_hp"] == 469
    assert normal_rows[18]["p1_hp"] == 112
    assert normal_rows[19]["p0_hp"] == 501
    assert normal_rows[19]["p1_hp"] == 175
    assert normal_rows[19]["p1_max_hp"] == 301
    assert normal_rows[20]["p0_hp"] == 394
    assert normal_rows[20]["p1_hp"] == 175


def test_battle551_cached_runswitch_speed_and_fainted_field_move_replay_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(29785),
        pool_get_team(7813),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        496671177,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Sandy Shocks' Volt Switch KOs the opposing Iron Jugulis on turn 6, then
    # the replacement Iron Jugulis ties the dead slot on cached switch-in
    # speed before Booster Energy / Quark Drive update. Missing that hidden
    # random(0, 2) frame used to shift turn 7's Hurricane into a miss.
    assert normal_rows[6]["p0_action"] == "move 4+pivot:switch 3"
    assert normal_rows[6]["p0_hp"] == 329
    assert normal_rows[6]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[6]["p1_hp"] == 305
    assert normal_rows[7]["p0_action"] == "move 3"
    assert normal_rows[7]["p0_hp"] == 329
    assert normal_rows[7]["p1_action"] == "move 3+forced:switch 2"
    assert normal_rows[7]["p1_active"] == 1
    assert normal_rows[7]["p1_hp"] == 279
    assert normal_rows[7]["p1_max_hp"] == 318

    # On turn 10, Hatterene KOs the slower Pincurchin before Toxic Spikes can
    # run. Deferred field-effect replay must not lay a second layer off the
    # canceled move, so Iron Leaves enters only normally poisoned.
    assert normal_rows[10]["p0_action"] == "move 4+forced:switch 6"
    assert normal_rows[10]["p0_active"] == 5
    assert normal_rows[10]["p0_hp"] == 322
    assert normal_rows[10]["p0_max_hp"] == 322
    assert normal_rows[10]["p0_status"] == 5
    assert normal_rows[10]["p1_action"] == "move 2"
    assert normal_rows[10]["p1_hp"] == 318


def test_battle552_air_balloon_blocks_psychic_terrain_offensive_boost_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42369),
        pool_get_team(43694),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1801804848,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Indeedee starts Psychic Terrain and is grounded, so the opening Psychics
    # get the offensive terrain boost.
    assert normal_rows[1]["p0_hp"] == 249
    assert normal_rows[2]["p0_hp"] == 117

    # Hatterene switches in with Air Balloon, so its later Psychics do not get
    # the Psychic Terrain boost even while the terrain is still active.
    assert normal_rows[4]["p0_action"] == "move 2"
    assert normal_rows[4]["p0_hp"] == 166
    assert normal_rows[4]["p1_action"] == "move 1"
    assert normal_rows[5]["p0_action"] == "move 2"
    assert normal_rows[5]["p0_hp"] == 60
    assert normal_rows[6]["p0_action"] == "move 2+forced:switch 6"
    assert normal_rows[6]["p0_hp"] == 307


def test_battle555_red_card_overrides_inline_pivot_before_switch_in_state_mutates():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(19808),
        pool_get_team(37583),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1455179509,
        5,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Meloetta's U-turn hits Ting-Lu, but Red Card overrides the self-switch
    # and drags in Garganacl directly before Ting-Lu's Stealth Rock. That drag
    # is automatic, so the replay queue must not serialize a fake player
    # `pivot:switch` token on turn 1.
    assert normal_rows[1]["p0_action"] == "move 4"
    assert normal_rows[1]["p0_active"] == 5
    assert normal_rows[1]["p0_hp"] == 404

    # Alomomola should enter turn 2 at full HP, take only Stealth Rock to 468,
    # then Earthquake to 353. The old provisional inline pivot used to leave it
    # at 519 before hazards, and the later fake replay token misaligned the
    # Showdown request stream.
    assert normal_rows[2]["p0_action"] == "switch 5"
    assert normal_rows[2]["p0_active"] == 4
    assert normal_rows[2]["p0_hp"] == 353
    assert normal_rows[2]["p0_max_hp"] == 534


def test_battle565_red_card_drag_cancels_late_pivot_and_switch_turn_stays_aligned():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(34129),
        pool_get_team(23439),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        320778558,
        24,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Pelipper's U-turn lands into Deoxys-Speed's Red Card on turn 11. The
    # automatic drag to Great Tusk must cancel the late self-switch replay so
    # the dragged-in Great Tusk, not a stale pivot target, starts turn 12.
    assert normal_rows[11]["p0_active"] == 5
    assert normal_rows[11]["p0_hp"] == 360
    assert normal_rows[11]["p0_max_hp"] == 371

    # Later, the opponent's manual Ninetales -> Great Tusk switch should leave
    # the incoming Great Tusk at Showdown's exact end-of-turn HP after Rapid
    # Spin + Leftovers, with the following Earthquake turn still aligned.
    assert normal_rows[14]["p1_action"] == "switch 3"
    assert normal_rows[14]["p1_active"] == 2
    assert normal_rows[14]["p1_hp"] == 403
    assert normal_rows[14]["p1_max_hp"] == 434
    assert normal_rows[15]["p1_hp"] == 252


def test_battle575_red_card_drag_keeps_showdown_switch_order_for_same_turn_replacement():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(39148),
        pool_get_team(47487),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1223923555,
        4,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Hatterene's Red Card drags Great Tusk in on turn 1, Great Tusk faints,
    # and the same-turn replacement must follow the updated side order:
    # Walking Wake enters, not the original Ninetales slot.
    assert normal_rows[1]["p1_action"] == "move 4+forced:switch 2"
    assert normal_rows[1]["p1_active"] == 1
    assert normal_rows[1]["p1_hp"] == 340
    assert normal_rows[1]["p1_max_hp"] == 340

    # The updated order must also carry into the next voluntary opponent
    # switch token, so the original Ninetales slot is now encoded as
    # `switch 4` instead of being treated like the next untouched bench slot.
    assert normal_rows[2]["p1_action"] == "switch 4"
    assert normal_rows[2]["p1_active"] == 0
    assert normal_rows[2]["p1_hp"] == 277
    assert normal_rows[2]["p1_max_hp"] == 349


def test_battle622_poison_puppeteer_confusion_waits_until_after_red_card_drag():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(5549),
        pool_get_team(38000),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        64770960,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Pecharunt's Malignant Chain badly poisons Araquanid on turn 6, Poison
    # Puppeteer confuses it, then Araquanid's Red Card drags Gliscor in before
    # Araquanid reaches its own confusion onBeforeMove check.
    assert normal_rows[6]["p0_action"] == "move 3"
    assert normal_rows[6]["p0_active"] == 1
    assert normal_rows[6]["p0_hp"] == 354
    assert normal_rows[6]["p0_max_hp"] == 354
    assert normal_rows[6]["p0_status"] == 6
    assert normal_rows[6]["p1_action"] == "move 2"
    assert normal_rows[6]["p1_active"] == 0
    assert normal_rows[6]["p1_hp"] == 206
    assert normal_rows[6]["p1_max_hp"] == 339
    assert normal_rows[6]["p1_status"] == 6

    # The repaired turn-6 PRNG ordering must also keep the later action queue
    # aligned; the next turn now matches Showdown's voluntary Pecharunt switch.
    assert normal_rows[7]["p0_action"] == "switch 2"
    assert normal_rows[7]["p0_active"] == 3
    assert normal_rows[7]["p0_hp"] == 380
    assert normal_rows[7]["p1_hp"] == 126


def test_battle632_tied_ko_before_slower_move_keeps_followup_prng_aligned():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(28715),
        pool_get_team(43728),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1604332210,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 15 is a faster Flip Turn into Weezing-Galar followed by Primarina's
    # Moonblast. Even though the self-switch resolves inline, Showdown still
    # spends the late tied-speed update chain after the slower move, so turn 16
    # must start aligned.
    assert normal_rows[15]["p0_action"] == "move 1+pivot:switch 2"
    assert normal_rows[15]["p0_active"] == 1
    assert normal_rows[15]["p0_hp"] == 249
    assert normal_rows[15]["p0_max_hp"] == 334
    assert normal_rows[15]["p1_action"] == "move 3"
    assert normal_rows[15]["p1_active"] == 2
    assert normal_rows[15]["p1_hp"] == 362
    assert normal_rows[15]["p1_max_hp"] == 364

    # The recovered late-update frames keep the next turn's damage rolls on the
    # Showdown line: Strange Steam leaves Primarina at 275 before Leftovers, and
    # Psychic Noise drops Weezing-Galar to 49.
    assert normal_rows[16]["p0_action"] == "move 3"
    assert normal_rows[16]["p0_active"] == 1
    assert normal_rows[16]["p0_hp"] == 49
    assert normal_rows[16]["p0_max_hp"] == 334
    assert normal_rows[16]["p1_action"] == "move 1"
    assert normal_rows[16]["p1_active"] == 2
    assert normal_rows[16]["p1_hp"] == 297
    assert normal_rows[16]["p1_max_hp"] == 364

    # On turn 17, faster Primarina KOs the tied slower Weezing before it can
    # act. Showdown then spends one extra tied speedSort frame when the last
    # active Neutralizing Gas user ends before Alomomola switches in.
    assert normal_rows[17]["p0_action"] == "move 3+forced:switch 2"
    assert normal_rows[17]["p0_active"] == 0
    assert normal_rows[17]["p0_hp"] == 360
    assert normal_rows[17]["p0_max_hp"] == 534
    assert normal_rows[17]["p1_action"] == "move 1"
    assert normal_rows[17]["p1_active"] == 2
    assert normal_rows[17]["p1_hp"] == 319
    assert normal_rows[17]["p1_max_hp"] == 364

    # That extra Neutralizing Gas onEnd frame keeps turn 18 aligned:
    # Alomomola survives Moonblast at 104 and Primarina ends at 320.
    assert normal_rows[18]["p0_action"] == "move 2"
    assert normal_rows[18]["p0_active"] == 0
    assert normal_rows[18]["p0_hp"] == 104
    assert normal_rows[18]["p0_max_hp"] == 534
    assert normal_rows[18]["p1_action"] == "move 3"
    assert normal_rows[18]["p1_active"] == 2
    assert normal_rows[18]["p1_hp"] == 320
    assert normal_rows[18]["p1_max_hp"] == 364

    # The following turn stays aligned too: Alomomola can still use Wish
    # before the later forced switch to Lokix.
    assert normal_rows[19]["p0_action"] == "move 1+forced:switch 3"
    assert normal_rows[19]["p0_active"] == 0
    assert normal_rows[19]["p0_hp"] == 282
    assert normal_rows[19]["p1_hp"] == 318


def test_battle634_ko_residual_refresh_uses_fainted_active_speed_not_stale_action_speed():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(35997),
        pool_get_team(12420),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        715301852,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 17: Serperior crit-KOs the opposing Serperior. Showdown keeps the
    # fainted slot's stale cached residual speed through upkeep before the
    # forced Gholdengo switch, so the hidden residual speedSort frame must
    # stay aligned here.
    assert normal_rows[17]["p0_action"] == "move 1"
    assert normal_rows[17]["p0_active"] == 2
    assert normal_rows[17]["p0_hp"] == 291
    assert normal_rows[17]["p1_action"] == "move 2+forced:switch 6"
    assert normal_rows[17]["p1_active"] == 1
    assert normal_rows[17]["p1_hp"] == 279

    # Turn 18 only matches Showdown if the KO-path residual helper used that
    # stale fainted-slot cache instead of the live boosted speed from before
    # the KO: Gholdengo survives Leaf Storm at 30 and can Nasty Plot.
    assert normal_rows[18]["p0_action"] == "move 1"
    assert normal_rows[18]["p0_active"] == 2
    assert normal_rows[18]["p0_hp"] == 291
    assert normal_rows[18]["p1_action"] == "move 1"
    assert normal_rows[18]["p1_active"] == 1
    assert normal_rows[18]["p1_hp"] == 30


def test_battle643_rapid_spin_still_clears_before_life_orb_ko_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(47612),
        pool_get_team(4470),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1035153174,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Great Tusk's final Rapid Spin on turn 40 still clears the current Spikes
    # layer even though Life Orb KOs it later in the same move. Ting-Lu then
    # rebuilds the stack over the next turns, and Iron Moth's later Eject
    # Button switch-in only takes the correctly preserved two-layer chip.
    assert normal_rows[40]["p0_action"] == "move 1+forced:switch 3"
    assert normal_rows[40]["p0_active"] == 2
    assert normal_rows[40]["p0_hp"] == 362
    assert normal_rows[41]["p0_action"] == "move 2"
    assert normal_rows[41]["p0_hp"] == 362
    assert normal_rows[42]["p0_action"] == "move 2"
    assert normal_rows[42]["p0_hp"] == 362
    assert normal_rows[42]["p1_hp"] == 342
    assert normal_rows[43]["p0_action"] == "move 2+forced:switch 6"
    assert normal_rows[43]["p0_active"] == 5
    assert normal_rows[43]["p0_hp"] == 61
    assert normal_rows[43]["p0_max_hp"] == 301
    assert normal_rows[43]["p1_action"] == "move 2"
    assert normal_rows[43]["p1_hp"] == 313


def test_battle577_struggle_recoil_ignores_rock_head():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(45198),
        pool_get_team(34974),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1858802874,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Both sides exhaust their usable moves and begin Struggling on turn 33.
    # Showdown still applies recoil to the Rock Head user; only ordinary move
    # recoil is suppressed by Rock Head.
    assert normal_rows[33]["p0_action"] == "move 2"
    assert normal_rows[33]["p0_hp"] == 153
    assert normal_rows[33]["p1_action"] == "move 4"
    assert normal_rows[33]["p1_hp"] == 201
    assert normal_rows[33]["p1_max_hp"] == 331

    assert normal_rows[34]["p0_action"] == "move 2+forced:switch 2"
    assert normal_rows[34]["p0_active"] == 1
    assert normal_rows[34]["p0_hp"] == 389
    assert normal_rows[34]["p1_action"] == "move 4"
    assert normal_rows[34]["p1_hp"] == 69
    assert normal_rows[34]["p1_max_hp"] == 331


def test_battle577_mist_ball_secondary_consumes_showdown_prng_frame():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(45198),
        pool_get_team(34974),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1858802874,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Mist Ball's 50% SpA-drop secondary fails on both turns here, but the
    # failed roll still spends Showdown PRNG. Missing that frame drifts the
    # following Hydro Pump rolls immediately.
    assert normal_rows[37]["p0_action"] == "move 2"
    assert normal_rows[37]["p0_active"] == 3
    assert normal_rows[37]["p0_hp"] == 218
    assert normal_rows[37]["p0_max_hp"] == 364
    assert normal_rows[37]["p1_action"] == "move 1"
    assert normal_rows[37]["p1_active"] == 1
    assert normal_rows[37]["p1_hp"] == 203
    assert normal_rows[37]["p1_max_hp"] == 304

    assert normal_rows[38]["p0_action"] == "move 2"
    assert normal_rows[38]["p0_hp"] == 165
    assert normal_rows[38]["p1_action"] == "move 1"
    assert normal_rows[38]["p1_hp"] == 116
    assert normal_rows[38]["p1_max_hp"] == 304


def test_battle577_trick_choice_lock_does_not_reapply_after_item_swap():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(45198),
        pool_get_team(34974),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1858802874,
        90,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Latias repeatedly Tricks a Choice Scarf back and forth with Kingambit on
    # turns 70-72. Showdown clears the old choice lock and does not immediately
    # recreate it from the post-Trick held item, so turn 73 can use Mist Ball
    # instead of being forced back into 0-PP Trick -> Struggle.
    assert normal_rows[70]["p0_action"] == "move 1"
    assert normal_rows[70]["p0_hp"] == 229
    assert normal_rows[71]["p0_hp"] == 251
    assert normal_rows[72]["p0_hp"] == 251

    assert normal_rows[73]["p0_action"] == "move 2"
    assert normal_rows[73]["p0_hp"] == 251
    assert normal_rows[73]["p1_action"] == "move 4"
    assert normal_rows[73]["p1_hp"] == 369

    assert normal_rows[74]["p0_hp"] == 251
    assert normal_rows[74]["p1_hp"] == 369


def test_battle20_one_sided_switch_resume_uses_raw_switchin_speed_frames():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(46126),
        pool_get_team(37238),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        787359109,
        61,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On the opponent's hard switch into Iron Treads, the one-sided switch
    # resume still needs the tied raw switch-in speed comparator frames so the
    # same-turn Earth Power roll stays aligned.
    assert normal_rows[38]["p1_action"] == "switch 2+die_switch:switch 2"
    assert normal_rows[39]["p1_action"] == "switch 6"
    assert normal_rows[39]["p1_active"] == 5
    assert normal_rows[39]["p1_hp"] == 127
    assert normal_rows[39]["p1_max_hp"] == 321


def test_battle259_one_sided_switch_resume_keeps_passive_speed_mods_but_skips_paradox_entry_boosts():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(5923),
        pool_get_team(13548),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1826507168,
        29,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Choice Scarf on the incoming Samurott-Hisui should still count for the
    # one-sided switch resume comparator, but the special-case Battle 565
    # paradox handling must not clobber that passive speed modifier.
    assert normal_rows[2]["p0_hp"] == 232
    assert normal_rows[3]["p0_action"] == "move 2"
    assert normal_rows[3]["p0_hp"] == 279
    assert normal_rows[3]["p0_max_hp"] == 291
    assert normal_rows[3]["p1_action"] == "switch 5"
    assert normal_rows[3]["p1_active"] == 4
    assert normal_rows[3]["p1_hp"] == 227
    assert normal_rows[3]["p1_max_hp"] == 321


def test_battle574_screen_break_moves_ignore_aurora_veil_for_their_own_damage():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(32638),
        pool_get_team(36556),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1146632765,
        18,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Metagross's Psychic Fangs should break Aurora Veil before its own damage
    # on turn 12, KO Polteageist through Stealth Rock chip, and force in Iron
    # Moth before the turn ends.
    assert normal_rows[12]["p0_active"] == 1
    assert normal_rows[12]["p0_hp"] == 226
    assert normal_rows[12]["p0_max_hp"] == 301
    assert normal_rows[12]["p1_active"] == 3
    assert normal_rows[12]["p1_hp"] == 150
    assert normal_rows[12]["p1_max_hp"] == 364


def test_battle822_hazard_fainted_switchin_counts_for_supreme_overlord():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(15058),
        pool_get_team(33976),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        57945699,
        24,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Dragapult dies to Stealth Rock on the forced switch at the end of turn
    # 12, so Kingambit's turn-13 Brick Break should already use Supreme
    # Overlord's fallen4 boost before Leftovers heals Ting-Lu back up.
    assert normal_rows[12]["p1_active"] == 4
    assert normal_rows[12]["p1_hp"] == 320
    assert normal_rows[12]["p1_max_hp"] == 341
    assert normal_rows[13]["p0_hp"] == 192
    assert normal_rows[13]["p0_max_hp"] == 514
    assert normal_rows[13]["p1_hp"] == 125
    assert normal_rows[13]["p1_max_hp"] == 341


def test_battle905_fickle_beam_roll_happens_before_damage_roll():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(23183),
        pool_get_team(21680),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1482035952,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 44, Hydrapple's Fickle Beam should use the non-doubled power
    # roll, leaving Zamazenta alive at 64 before Leftovers and 84 after.
    assert normal_rows[43]["p0_hp"] == 206
    assert normal_rows[43]["p1_hp"] == 354
    assert normal_rows[44]["p0_hp"] == 84
    assert normal_rows[44]["p0_max_hp"] == 325
    assert normal_rows[44]["p0_status"] == 2
    assert normal_rows[44]["p1_hp"] == 291
    assert normal_rows[44]["p1_max_hp"] == 416


def test_battle858_tachyon_cutter_rechecks_multiscale_each_hit():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(38647),
        pool_get_team(16463),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        581815108,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Iron Crown's Tachyon Cutter should only get Multiscale's 0.5x reduction
    # on the first hit; the second hit lands after Dragonite is no longer at
    # full HP.
    assert normal_rows[16]["p0_hp"] == 351
    assert normal_rows[17]["p0_action"] == "move 1"
    assert normal_rows[17]["p0_active"] == 4
    assert normal_rows[17]["p0_hp"] == 233
    assert normal_rows[17]["p0_max_hp"] == 351
    assert normal_rows[17]["p1_action"] == "move 4"
    assert normal_rows[17]["p1_active"] == 3
    assert normal_rows[17]["p1_hp"] == 360
    assert normal_rows[17]["p1_max_hp"] == 383
    assert normal_rows[18]["p0_hp"] == 97


def test_battle142_later_multihit_crit_uses_per_hit_crit_state():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(43215),
        pool_get_team(40206),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        53904684,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Weavile's Icicle Spear on turn 25 crits on a later hit, so the crit
    # multiplier and crit-only screen bypass must be keyed off the current hit
    # rather than whatever happened on hit 1.
    assert normal_rows[25]["p0_action"] == "switch 4"
    assert normal_rows[25]["p0_active"] == 1
    assert normal_rows[25]["p0_hp"] == 255
    assert normal_rows[25]["p0_max_hp"] == 400
    assert normal_rows[25]["p1_action"] == "move 2"
    assert normal_rows[25]["p1_active"] == 2
    assert normal_rows[25]["p1_hp"] == 157
    assert normal_rows[25]["p1_max_hp"] == 281


def test_battle580_lockedmove_pp_and_typeless_struggle_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(21551),
        pool_get_team(37180),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        978838227,
        42,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown skips the normal PP deduction while Braviary is in the hard
    # locked Thrash chain, so it stays on Thrash through turn 21 and only
    # reaches Struggle on turn 22. That Struggle is typeless ('???'), so it
    # deals neutral non-STAB damage to Gholdengo before Braviary faints.
    assert normal_rows[21]["p0_action"] == "move 4"
    assert normal_rows[21]["p0_hp"] == 315
    assert normal_rows[21]["p0_max_hp"] == 315
    assert normal_rows[21]["p1_action"] == "move 1"
    assert normal_rows[21]["p1_hp"] == 31
    assert normal_rows[21]["p1_max_hp"] == 341

    assert normal_rows[22]["p0_action"] == "move 4"
    assert normal_rows[22]["p0_hp"] == 253
    assert normal_rows[22]["p0_max_hp"] == 315
    assert normal_rows[22]["p1_action"] == "move 1+forced:switch 3"
    assert normal_rows[22]["p1_hp"] == 321
    assert normal_rows[22]["p1_max_hp"] == 321


def test_battle296_slower_recoil_caps_at_post_confusion_target_hp_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(787),
        pool_get_team(13537),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        214706214,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 40, Garchomp first hurts itself in confusion, then Corviknight's
    # slower Brave Bird KOs from 109 HP. Showdown bases recoil on that actual
    # 109 HP removed, not on the raw pre-cap damage roll from the turn-start
    # target HP snapshot.
    assert normal_rows[39]["p0_action"] == "move 3"
    assert normal_rows[39]["p0_hp"] == 206
    assert normal_rows[39]["p0_max_hp"] == 399
    assert normal_rows[39]["p1_action"] == "move 2"
    assert normal_rows[39]["p1_hp"] == 157
    assert normal_rows[39]["p1_max_hp"] == 357

    assert normal_rows[40]["p0_action"] == "move 3"
    assert normal_rows[40]["p0_hp"] == 145
    assert normal_rows[40]["p0_max_hp"] == 399
    assert normal_rows[40]["p1_action"] == "move 2+forced:switch 4"
    assert normal_rows[40]["p1_hp"] == 357
    assert normal_rows[40]["p1_max_hp"] == 380


def test_battle582_confusion_bit_does_not_block_grassy_terrain_switch_in_heal_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(49401),
        pool_get_team(18857),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1728693357,
        6,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 3, Kyurem switches into Zapdos's Hurricane, becomes confused,
    # and still gets the same-turn Grassy Terrain heal. The side-volatile
    # confusion helper bit must not masquerade as semi-invulnerability.
    assert normal_rows[3]["p0_action"] == "move 2"
    assert normal_rows[3]["p0_hp"] == 384
    assert normal_rows[3]["p0_max_hp"] == 384
    assert normal_rows[3]["p1_action"] == "switch 4"
    assert normal_rows[3]["p1_active"] == 3
    assert normal_rows[3]["p1_hp"] == 255
    assert normal_rows[3]["p1_max_hp"] == 391

    assert normal_rows[4]["p1_hp"] == 279
    assert normal_rows[4]["p1_max_hp"] == 391


def test_battle0_paradox_switch_in_does_not_false_flag_semi_invul_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4462),
        pool_get_team(38697),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1405681632,
        5,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 3 is the regression guard for the abandoned "Pokemon flag 0x10"
    # attempt. A freshly switched-in Quark Drive Iron Valiant must not look
    # semi-invulnerable and dodge Great Tusk's Headlong Rush.
    assert normal_rows[3]["p0_action"] == "move 1"
    assert normal_rows[3]["p0_hp"] == 371
    assert normal_rows[3]["p1_action"] == "switch 5+die_switch:switch 5"
    assert normal_rows[3]["p1_active"] == 0
    assert normal_rows[3]["p1_hp"] == 301
    assert normal_rows[3]["p1_max_hp"] == 301

    assert normal_rows[4]["p1_action"] == "switch 3"
    assert normal_rows[4]["p1_active"] == 2
    assert normal_rows[4]["p1_hp"] == 323
    assert normal_rows[4]["p1_max_hp"] == 323


def test_battle586_healing_wish_consumes_on_manual_switch_before_hazards_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(28569),
        pool_get_team(29049),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        826084853,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Hatterene has already armed Healing Wish earlier in the battle. On turn
    # 15, Walking Wake switches in at 201/339, Healing Wish should restore it
    # to full before Gholdengo's Make It Rain and sandstorm resolve, and the
    # pending sentinel must survive the generic Destiny Bond cleanup path until
    # that switch happens.
    assert normal_rows[14]["p0_action"] == "move 2"
    assert normal_rows[14]["p0_active"] == 5
    assert normal_rows[14]["p0_hp"] == 287
    assert normal_rows[14]["p0_max_hp"] == 306

    assert normal_rows[15]["p0_action"] == "switch 2"
    assert normal_rows[15]["p0_active"] == 2
    assert normal_rows[15]["p0_hp"] == 219
    assert normal_rows[15]["p0_max_hp"] == 339
    assert normal_rows[15]["p1_action"] == "move 2"
    assert normal_rows[15]["p1_active"] == 3
    assert normal_rows[15]["p1_hp"] == 312
    assert normal_rows[15]["p1_max_hp"] == 378


def test_battle783_sticky_web_switch_resume_keeps_pre_hazard_tie_frames_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(25208),
        pool_get_team(26939),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1611183940,
        8,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Side 0 hard-switches Great Tusk into Sticky Web on turn 5 while the foe's
    # active is an equally fast Great Tusk. Showdown still spends the three
    # runSwitch tie frames using the entrant's pre-hazard neutral speed, then
    # Headlong Rush leaves the switch-in at 151/371. Recomputing the entrant's
    # speed after Sticky Web would skip those frames and drift the damage roll.
    assert normal_rows[4]["p0_action"] == "move 4"
    assert normal_rows[4]["p1_action"] == "switch 4"
    assert normal_rows[4]["p1_active"] == 3
    assert normal_rows[4]["p1_hp"] == 371
    assert normal_rows[4]["p1_max_hp"] == 371

    assert normal_rows[5]["p0_action"] == "switch 2"
    assert normal_rows[5]["p0_active"] == 1
    assert normal_rows[5]["p0_hp"] == 151
    assert normal_rows[5]["p0_max_hp"] == 371
    assert normal_rows[5]["p1_action"] == "move 3"
    assert normal_rows[5]["p1_active"] == 3
    assert normal_rows[5]["p1_hp"] == 371
    assert normal_rows[5]["p1_max_hp"] == 371


def test_battle650_late_pivot_residual_snapshot_uses_pre_weather_speed_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(45771),
        pool_get_team(30406),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        154093129,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 11 is a switch-vs-pivot sequence under sun: side 1 switches Great
    # Tusk into Stealth Rock, Primarina uses Flip Turn, then side 0 pivots into
    # its own Great Tusk. Showdown snapshots the residual cached speed pair
    # before weather expiry ends Protosynthesis, so turn 12 keeps the correct
    # Rapid Spin / Headlong Rush damage rolls.
    assert normal_rows[11]["p0_action"] == "move 3+pivot:switch 2"
    assert normal_rows[11]["p0_active"] == 1
    assert normal_rows[11]["p0_hp"] == 60
    assert normal_rows[11]["p0_max_hp"] == 434
    assert normal_rows[11]["p1_action"] == "switch 5"
    assert normal_rows[11]["p1_active"] == 4
    assert normal_rows[11]["p1_hp"] == 347
    assert normal_rows[11]["p1_max_hp"] == 371

    assert normal_rows[12]["p0_action"] == "move 4"
    assert normal_rows[12]["p0_active"] == 1
    assert normal_rows[12]["p0_hp"] == 10
    assert normal_rows[12]["p0_max_hp"] == 434
    assert normal_rows[12]["p1_action"] == "move 1"
    assert normal_rows[12]["p1_active"] == 4
    assert normal_rows[12]["p1_hp"] == 235
    assert normal_rows[12]["p1_max_hp"] == 371


def test_battle910_last_queued_protect_fails_before_stall_rng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(48052),
        pool_get_team(26112),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2000356841,
        112,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Gliscor spends turns 94-109 clicking Protect into repeated switches.
    # Showdown's onPrepareHit first checks `queue.willAct()`, so the
    # last-queued Protect fails before StallMove RNG or protect/stall
    # volatiles can consume hidden frames. The turn stays byte-aligned
    # through the later Struggle turns only if those failed Protects spend
    # zero hidden PRNG frames.
    assert normal_rows[109]["p0_action"] == "switch 3"
    assert normal_rows[109]["p0_active"] == 5
    assert normal_rows[109]["p0_hp"] == 339
    assert normal_rows[109]["p1_action"] == "move 4"
    assert normal_rows[109]["p1_active"] == 5
    assert normal_rows[109]["p1_hp"] == 354
    assert normal_rows[109]["p1_max_hp"] == 354

    assert normal_rows[110]["p0_action"] == "switch 3"
    assert normal_rows[110]["p0_active"] == 1
    assert normal_rows[110]["p0_hp"] == 299
    assert normal_rows[110]["p0_max_hp"] == 394
    assert normal_rows[110]["p1_action"] == "move 1"
    assert normal_rows[110]["p1_active"] == 5
    assert normal_rows[110]["p1_hp"] == 309
    assert normal_rows[110]["p1_max_hp"] == 354

    assert normal_rows[111]["p0_action"] == "switch 3"
    assert normal_rows[111]["p0_active"] == 5
    assert normal_rows[111]["p0_hp"] == 304
    assert normal_rows[111]["p0_max_hp"] == 415
    assert normal_rows[111]["p1_action"] == "move 1"
    assert normal_rows[111]["p1_active"] == 5
    assert normal_rows[111]["p1_hp"] == 264
    assert normal_rows[111]["p1_max_hp"] == 354


def test_battle587_purifying_salt_halves_shadow_ball_on_spa_chain_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(20456),
        pool_get_team(509),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1292301448,
        7,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown's Purifying Salt halves Ghost damage through the defender's
    # onSourceModifyAtk / onSourceModifySpA hook, not at final damage.
    # Against Garganacl, Pecharunt's Shadow Ball should therefore follow the
    # same rounded damage sequence Showdown gets from the attacker-stat chain:
    # 30 damage on turn 4, 31 on turn 5, then 28 on turn 6.
    assert normal_rows[4]["p0_action"] == "move 4"
    assert normal_rows[4]["p0_active"] == 0
    assert normal_rows[4]["p0_hp"] == 379
    assert normal_rows[4]["p1_action"] == "move 3"
    assert normal_rows[4]["p1_active"] == 1
    assert normal_rows[4]["p1_hp"] == 399
    assert normal_rows[4]["p1_max_hp"] == 404

    assert normal_rows[5]["p0_action"] == "move 4"
    assert normal_rows[5]["p0_active"] == 0
    assert normal_rows[5]["p0_hp"] == 298
    assert normal_rows[5]["p1_action"] == "move 1"
    assert normal_rows[5]["p1_active"] == 1
    assert normal_rows[5]["p1_hp"] == 393
    assert normal_rows[5]["p1_max_hp"] == 404

    assert normal_rows[6]["p0_action"] == "move 4"
    assert normal_rows[6]["p0_active"] == 0
    assert normal_rows[6]["p0_hp"] == 224
    assert normal_rows[6]["p1_action"] == "move 1"
    assert normal_rows[6]["p1_active"] == 1
    assert normal_rows[6]["p1_hp"] == 390
    assert normal_rows[6]["p1_max_hp"] == 404


def test_battle588_revelation_dance_uses_live_primary_type_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(28789),
        pool_get_team(46769),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1082932105,
        23,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Oricorio-Pom-Pom's Revelation Dance should become Electric from its live
    # primary type, matching Showdown's pokemon.getTypes()[0] onModifyType.
    # If pokepy leaves the move as Normal, Iron Crown is under-damaged by 90
    # on turn 12 and that hidden bench-state error reappears when it switches
    # back into Corviknight's Body Press on turn 22.
    assert normal_rows[11]["p0_action"] == "move 3"
    assert normal_rows[11]["p0_active"] == 5
    assert normal_rows[11]["p0_hp"] == 354
    assert normal_rows[11]["p1_action"] == "switch 4"
    assert normal_rows[11]["p1_active"] == 3
    assert normal_rows[11]["p1_hp"] == 321
    assert normal_rows[11]["p1_max_hp"] == 321

    assert normal_rows[12]["p0_action"] == "move 1"
    assert normal_rows[12]["p0_active"] == 5
    assert normal_rows[12]["p0_hp"] == 283
    assert normal_rows[12]["p1_action"] == "move 2+forced:switch 4"
    assert normal_rows[12]["p1_active"] == 1
    assert normal_rows[12]["p1_hp"] == 267
    assert normal_rows[12]["p1_max_hp"] == 371

    assert normal_rows[22]["p0_action"] == "move 4"
    assert normal_rows[22]["p0_active"] == 2
    assert normal_rows[22]["p0_hp"] == 384
    assert normal_rows[22]["p1_action"] == "switch 4"
    assert normal_rows[22]["p1_active"] == 3
    assert normal_rows[22]["p1_hp"] == 118
    assert normal_rows[22]["p1_max_hp"] == 321


def test_battle719_eject_button_does_not_consume_or_clear_without_bench_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(31610),
        pool_get_team(40649),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        717498482,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 37, Alomomola's Eject Button holder has no valid bench
    # replacement. Showdown therefore keeps the item path from clearing the
    # holder's switch-out state, so Moonblast's SpA drop remains in place and
    # the follow-up Scald leaves Ribombee at 21 before burn, not at 0.
    assert normal_rows[36]["p0_action"] == "move 2+forced:switch 2"
    assert normal_rows[36]["p0_active"] == 0
    assert normal_rows[36]["p0_hp"] == 66
    assert normal_rows[36]["p0_max_hp"] == 262
    assert normal_rows[36]["p1_action"] == "move 3"
    assert normal_rows[36]["p1_active"] == 4
    assert normal_rows[36]["p1_hp"] == 534
    assert normal_rows[36]["p1_max_hp"] == 534

    assert normal_rows[37]["p0_action"] == "move 2"
    assert normal_rows[37]["p0_active"] == 0
    assert normal_rows[37]["p0_hp"] == 5
    assert normal_rows[37]["p0_max_hp"] == 262
    assert normal_rows[37]["p0_status"] == 1
    assert normal_rows[37]["p1_action"] == "move 3"
    assert normal_rows[37]["p1_active"] == 4
    assert normal_rows[37]["p1_hp"] == 285
    assert normal_rows[37]["p1_max_hp"] == 534

    assert normal_rows[38]["p0_action"] == "move 2+forced:switch 4"
    assert normal_rows[38]["p0_active"] == 2
    assert normal_rows[38]["p0_hp"] == 301
    assert normal_rows[38]["p0_max_hp"] == 301
    assert normal_rows[38]["p1_action"] == "move 1"
    assert normal_rows[38]["p1_active"] == 4
    assert normal_rows[38]["p1_hp"] == 20
    assert normal_rows[38]["p1_max_hp"] == 534


def test_battle590_switching_side_does_not_spend_incoming_priority_rng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(16901),
        pool_get_team(18860),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        778861908,
        6,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 3, side 1 is switching Slowbro-Galar in. Showdown does not run
    # the incoming switch target's move-priority hooks for a switch action, so
    # Quick Draw must not spend a hidden random(10) before Ogerpon-Wellspring's
    # Ivy Cudgel crit/damage rolls.
    assert normal_rows[2]["p0_action"] == "switch 6"
    assert normal_rows[2]["p0_active"] == 5
    assert normal_rows[2]["p0_hp"] == 301
    assert normal_rows[2]["p1_action"] == "move 4"
    assert normal_rows[2]["p1_active"] == 2
    assert normal_rows[2]["p1_hp"] == 403

    assert normal_rows[3]["p0_action"] == "move 1"
    assert normal_rows[3]["p0_active"] == 5
    assert normal_rows[3]["p0_hp"] == 301
    assert normal_rows[3]["p1_action"] == "switch 5"
    assert normal_rows[3]["p1_active"] == 4
    assert normal_rows[3]["p1_hp"] == 214
    assert normal_rows[3]["p1_max_hp"] == 393

    assert normal_rows[4]["p0_action"] == "switch 3"
    assert normal_rows[4]["p0_active"] == 2
    assert normal_rows[4]["p0_hp"] == 317
    assert normal_rows[4]["p1_action"] == "move 2"
    assert normal_rows[4]["p1_active"] == 4
    assert normal_rows[4]["p1_hp"] == 238
    assert normal_rows[4]["p1_max_hp"] == 393


def test_battle603_rain_perfect_storm_moves_skip_accuracy_rng_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(47966),
        pool_get_team(12237),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1401274313,
        15,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown's Bleakwind / Wildbolt / Sandsear storm family uses the
    # defender's effectiveWeather() and sets accuracy=true in rain or
    # primordial sea. If pokepy still spends random(100) here, turn 12 drifts
    # before both Bleakwind Storm's damage roll and Primarina's Moonblast.
    assert normal_rows[11]["p0_action"] == "move 2"
    assert normal_rows[11]["p0_active"] == 5
    assert normal_rows[11]["p0_hp"] == 312
    assert normal_rows[11]["p0_max_hp"] == 364
    assert normal_rows[11]["p1_action"] == "move 2+forced:switch 3"
    assert normal_rows[11]["p1_active"] == 2
    assert normal_rows[11]["p1_hp"] == 225
    assert normal_rows[11]["p1_max_hp"] == 299

    assert normal_rows[12]["p0_action"] == "move 2"
    assert normal_rows[12]["p0_active"] == 5
    assert normal_rows[12]["p0_hp"] == 221
    assert normal_rows[12]["p0_max_hp"] == 364
    assert normal_rows[12]["p1_action"] == "move 2"
    assert normal_rows[12]["p1_active"] == 2
    assert normal_rows[12]["p1_hp"] == 48
    assert normal_rows[12]["p1_max_hp"] == 299

    assert normal_rows[13]["p0_action"] == "move 2"
    assert normal_rows[13]["p0_active"] == 5
    assert normal_rows[13]["p0_hp"] == 165
    assert normal_rows[13]["p0_max_hp"] == 364
    assert normal_rows[13]["p1_action"] == "move 2+forced:switch 4"
    assert normal_rows[13]["p1_active"] == 3
    assert normal_rows[13]["p1_hp"] == 271
    assert normal_rows[13]["p1_max_hp"] == 321


def test_battle619_multihit_contact_status_resolves_per_hit_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(29211),
        pool_get_team(26201),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1578678424,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown resolves multihit contact abilities inside the hit loop. On
    # turn 10, Moltres's Flame Body burns Maushold after the first Population
    # Bomb hit, so the remaining hits are burn-halved instead of KOing Moltres.
    assert normal_rows[9]["p0_action"] == "move 1"
    assert normal_rows[9]["p0_active"] == 4
    assert normal_rows[9]["p0_hp"] == 289
    assert normal_rows[9]["p0_max_hp"] == 289
    assert normal_rows[9]["p1_action"] == "move 1+forced:switch 4"
    assert normal_rows[9]["p1_active"] == 3
    assert normal_rows[9]["p1_hp"] == 384
    assert normal_rows[9]["p1_max_hp"] == 384

    assert normal_rows[10]["p0_action"] == "move 1"
    assert normal_rows[10]["p0_active"] == 4
    assert normal_rows[10]["p0_hp"] == 79
    assert normal_rows[10]["p0_max_hp"] == 289
    assert normal_rows[10]["p0_status"] == 1
    assert normal_rows[10]["p1_action"] == "move 2"
    assert normal_rows[10]["p1_active"] == 3
    assert normal_rows[10]["p1_hp"] == 101
    assert normal_rows[10]["p1_max_hp"] == 384


def test_battle825_fainted_active_skips_disablemove_shuffle_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(21136),
        pool_get_team(26072),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        378254117,
        15,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Walking Wake is fainted on turn 7 while still carrying both Choice lock
    # and Cursed Body Disable. Showdown does not rebuild DisableMove handlers
    # for a forced-switch request, so no hidden random(0, 2) shuffle frame is
    # spent before Iron Valiant's turn-8 Shadow Ball or turn-9 Close Combat.
    assert normal_rows[7]["p0_action"] == "move 1+forced:switch 4"
    assert normal_rows[7]["p0_active"] == 3
    assert normal_rows[7]["p0_hp"] == 241
    assert normal_rows[7]["p0_max_hp"] == 289
    assert normal_rows[7]["p1_action"] == "move 3"
    assert normal_rows[7]["p1_active"] == 3
    assert normal_rows[7]["p1_hp"] == 68
    assert normal_rows[7]["p1_max_hp"] == 261

    assert normal_rows[8]["p0_action"] == "move 4"
    assert normal_rows[8]["p0_active"] == 3
    assert normal_rows[8]["p0_hp"] == 241
    assert normal_rows[8]["p0_max_hp"] == 289
    assert normal_rows[8]["p1_action"] == "move 1+forced:switch 5"
    assert normal_rows[8]["p1_active"] == 4
    assert normal_rows[8]["p1_hp"] == 349
    assert normal_rows[8]["p1_max_hp"] == 349

    assert normal_rows[9]["p0_action"] == "move 3"
    assert normal_rows[9]["p0_active"] == 3
    assert normal_rows[9]["p0_hp"] == 241
    assert normal_rows[9]["p0_max_hp"] == 289
    assert normal_rows[9]["p1_action"] == "move 1"
    assert normal_rows[9]["p1_active"] == 4
    assert normal_rows[9]["p1_hp"] == 73
    assert normal_rows[9]["p1_max_hp"] == 349


def test_battle662_snowscape_def_boost_applies_to_psyshock_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(41366),
        pool_get_team(10597),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        452345345,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown's Snowscape boost is an onModifyDef hook, so it still applies
    # when Psyshock uses the target's Def despite the move category not being
    # Physical. Ninetales-Alola must survive Hatterene's turn-15 Psyshock and
    # stay active instead of triggering a forced switch.
    assert normal_rows[12]["p0_action"] == "move 2+forced:switch 5"
    assert normal_rows[12]["p0_active"] == 0
    assert normal_rows[12]["p0_hp"] == 142
    assert normal_rows[12]["p0_max_hp"] == 287
    assert normal_rows[12]["p1_action"] == "move 3"
    assert normal_rows[12]["p1_active"] == 1
    assert normal_rows[12]["p1_hp"] == 40
    assert normal_rows[12]["p1_max_hp"] == 321

    assert normal_rows[15]["p0_action"] == "move 3"
    assert normal_rows[15]["p0_active"] == 0
    assert normal_rows[15]["p0_hp"] == 40
    assert normal_rows[15]["p0_max_hp"] == 287
    assert normal_rows[15]["p1_action"] == "move 3"
    assert normal_rows[15]["p1_active"] == 3
    assert normal_rows[15]["p1_hp"] == 110
    assert normal_rows[15]["p1_max_hp"] == 318


def test_battle666_fainted_target_cannot_reflect_moonblast_secondary_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(14032),
        pool_get_team(14044),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1885770941,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Corviknight's Mirror Armor should not reflect Moonblast's SpA-drop
    # secondary after Corviknight faints to the primary hit. If pokepy applies
    # that impossible reflected drop anyway, Iron Valiant stays at -1 SpA and
    # under-damages Furret on turn 15.
    assert normal_rows[13]["p0_action"] == "switch 2+forced:switch 3"
    assert normal_rows[13]["p0_active"] == 3
    assert normal_rows[13]["p0_hp"] == 513
    assert normal_rows[13]["p0_max_hp"] == 513
    assert normal_rows[13]["p1_action"] == "move 1"
    assert normal_rows[13]["p1_active"] == 0
    assert normal_rows[13]["p1_hp"] == 253
    assert normal_rows[13]["p1_max_hp"] == 289

    assert normal_rows[15]["p0_action"] == "switch 6"
    assert normal_rows[15]["p0_active"] == 5
    assert normal_rows[15]["p0_hp"] == 38
    assert normal_rows[15]["p0_max_hp"] == 312
    assert normal_rows[15]["p1_action"] == "move 1"
    assert normal_rows[15]["p1_active"] == 0
    assert normal_rows[15]["p1_hp"] == 253
    assert normal_rows[15]["p1_max_hp"] == 289


def test_battle679_first_move_self_ko_skips_extra_tied_update_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(2012),
        pool_get_team(49398),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        965578845,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 5, the faster Corviknight faints to Rocky Helmet + recoil after
    # Brave Bird. Showdown skips the later generic post-action Update because
    # `instaswitch` is pending, so turn 6's U-turn damage roll must still land
    # at 39 before sand chip. If pokepy burns one extra tied Update frame on
    # turn 5, Slowking-Galar falls to 327 instead of Showdown's 331.
    assert normal_rows[5]["p0_action"] == "move 1"
    assert normal_rows[5]["p0_active"] == 3
    assert normal_rows[5]["p0_hp"] == 106
    assert normal_rows[5]["p0_max_hp"] == 399
    assert normal_rows[5]["p1_action"] == "move 2+forced:switch 3"
    assert normal_rows[5]["p1_active"] == 2
    assert normal_rows[5]["p1_hp"] == 394
    assert normal_rows[5]["p1_max_hp"] == 394

    assert normal_rows[6]["p0_action"] == "move 2+pivot:switch 2"
    assert normal_rows[6]["p0_active"] == 1
    assert normal_rows[6]["p0_hp"] == 341
    assert normal_rows[6]["p0_max_hp"] == 341
    assert normal_rows[6]["p1_action"] == "move 4"
    assert normal_rows[6]["p1_active"] == 2
    assert normal_rows[6]["p1_hp"] == 331
    assert normal_rows[6]["p1_max_hp"] == 394


def test_battle680_slower_same_turn_confusion_checks_after_between_move_updates():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(14483),
        pool_get_team(44624),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1602887780,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 13, Pelipper's Hurricane confuses the slower Alomomola. Showdown
    # spends the tied Update frames before Alomomola reaches confusion's
    # onBeforeMove, so the same-turn self-hit still lands and leaves it at 183.
    # If pokepy rolls that confusion check too early, it keeps Alomomola at 200
    # and drifts the follow-up PRNG.
    assert normal_rows[12]["p0_action"] == "move 4"
    assert normal_rows[12]["p0_active"] == 2
    assert normal_rows[12]["p0_hp"] == 169
    assert normal_rows[12]["p0_max_hp"] == 323
    assert normal_rows[12]["p1_action"] == "switch 6"
    assert normal_rows[12]["p1_active"] == 4
    assert normal_rows[12]["p1_hp"] == 410
    assert normal_rows[12]["p1_max_hp"] == 534

    assert normal_rows[13]["p0_action"] == "move 2"
    assert normal_rows[13]["p0_active"] == 2
    assert normal_rows[13]["p0_hp"] == 169
    assert normal_rows[13]["p0_max_hp"] == 323
    assert normal_rows[13]["p1_action"] == "move 1"
    assert normal_rows[13]["p1_active"] == 4
    assert normal_rows[13]["p1_hp"] == 183
    assert normal_rows[13]["p1_max_hp"] == 534


def test_battle685_fatal_hit_on_damaging_hit_abilities_still_apply():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(18316),
        pool_get_team(36736),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1585939679,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 20, Iron Valiant KOs Goodra-Hisui with Close Combat, but Goodra's
    # Gooey still applies the Speed drop before faint resolution. That leaves
    # Rillaboom faster on turn 21, so Wood Hammer resolves first and survives
    # recoil at 255 HP. If pokepy suppresses fatal-hit onDamagingHit effects,
    # Iron Valiant wrongly keeps full Speed and KOs Rillaboom instead.
    assert normal_rows[20]["p0_action"] == "move 3"
    assert normal_rows[20]["p0_active"] == 5
    assert normal_rows[20]["p0_hp"] == 261
    assert normal_rows[20]["p0_max_hp"] == 289
    assert normal_rows[20]["p1_action"] == "move 1+forced:switch 5"
    assert normal_rows[20]["p1_active"] == 4
    assert normal_rows[20]["p1_hp"] == 341
    assert normal_rows[20]["p1_max_hp"] == 341

    assert normal_rows[21]["p0_action"] == "move 3"
    assert normal_rows[21]["p0_active"] == 5
    assert normal_rows[21]["p0_hp"] == 0
    assert normal_rows[21]["p0_max_hp"] == 289
    assert normal_rows[21]["p1_action"] == "move 4"
    assert normal_rows[21]["p1_active"] == 4
    assert normal_rows[21]["p1_hp"] == 255
    assert normal_rows[21]["p1_max_hp"] == 341


def test_battle688_neutralizing_gas_suppresses_late_status_ability_checks():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(29478),
        pool_get_team(25446),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1557149444,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 16, Weezing-Galar's Defog targets Gholdengo while Neutralizing
    # Gas is active, so Good as Gold is suppressed and Defog still clears the
    # lingering hazards. On turn 17, Iron Moth should switch in and take only
    # Shadow Ball damage (171 HP), not Shadow Ball plus 25% Stealth Rock chip.
    assert normal_rows[16]["p0_action"] == "move 1"
    assert normal_rows[16]["p0_active"] == 4
    assert normal_rows[16]["p0_hp"] == 313
    assert normal_rows[16]["p0_max_hp"] == 334
    assert normal_rows[16]["p1_action"] == "switch 3"
    assert normal_rows[16]["p1_active"] == 5
    assert normal_rows[16]["p1_hp"] == 299
    assert normal_rows[16]["p1_max_hp"] == 315

    assert normal_rows[17]["p0_action"] == "switch 3"
    assert normal_rows[17]["p0_active"] == 3
    assert normal_rows[17]["p0_hp"] == 171
    assert normal_rows[17]["p0_max_hp"] == 301
    assert normal_rows[17]["p1_action"] == "move 1"
    assert normal_rows[17]["p1_active"] == 5
    assert normal_rows[17]["p1_hp"] == 299
    assert normal_rows[17]["p1_max_hp"] == 315


def test_battle102_fatal_hit_toxic_debris_only_adds_one_layer():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(15108),
        pool_get_team(27701),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        677943770,
        22,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 3, Ting-Lu's fatal Earthquake triggers Glimmora's Toxic Debris
    # exactly once, leaving a single Toxic Spikes layer on P2's side. When
    # Ogerpon-Wellspring switches in on turn 18, it should be normally poisoned
    # (psn), not badly poisoned (tox), and the turn-19 residual should match
    # one-layer poison damage.
    assert normal_rows[18]["p0_action"] == "move 2"
    assert normal_rows[18]["p0_active"] == 4
    assert normal_rows[18]["p0_hp"] == 258
    assert normal_rows[18]["p1_action"] == "move 4+forced:switch 2"
    assert normal_rows[18]["p1_active"] == 1
    assert normal_rows[18]["p1_hp"] == 264
    assert normal_rows[18]["p1_max_hp"] == 301
    assert normal_rows[18]["p1_status"] == 5

    assert normal_rows[19]["p0_action"] == "move 2"
    assert normal_rows[19]["p0_hp"] == 54
    assert normal_rows[19]["p1_action"] == "move 1"
    assert normal_rows[19]["p1_hp"] == 150
    assert normal_rows[19]["p1_max_hp"] == 301
    assert normal_rows[19]["p1_status"] == 5


def test_battle696_tied_speed_blocked_move_and_weather_expiry_updates_stay_aligned():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(15251),
        pool_get_team(988),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1905574367,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Battle 696 has two silent PRNG traps on tied Kingambit turns:
    # 1. A frozen first mover still spends the generic post-action Update
    #    before the slower move runs.
    # 2. Snowscape expiring at residual start still fires WeatherChange's
    #    tied-speed speedSort even though the weather residual body is skipped.
    # Missing either frame leaves the row state matching for a while, then
    # flips the later Kowtow Cleave speed tie on turn 41.
    assert normal_rows[10]["p1_action"] == "move 3"
    assert normal_rows[10]["p1_status"] == 4

    assert normal_rows[11]["p0_action"] == "move 2"
    assert normal_rows[11]["p0_hp"] == 404
    assert normal_rows[11]["p1_action"] == "move 3"
    assert normal_rows[11]["p1_hp"] == 204
    assert normal_rows[11]["p1_status"] == 0

    assert normal_rows[41]["p0_action"] == "move 1"
    assert normal_rows[41]["p0_active"] == 2
    assert normal_rows[41]["p0_hp"] == 404
    assert normal_rows[41]["p1_action"] == "move 2+forced:switch 4"
    assert normal_rows[41]["p1_active"] == 0
    assert normal_rows[41]["p1_hp"] == 323

    assert normal_rows[42]["p0_action"] == "switch 3"
    assert normal_rows[42]["p0_active"] == 0
    assert normal_rows[42]["p0_hp"] == 212
    assert normal_rows[42]["p1_action"] == "move 4"
    assert normal_rows[42]["p1_hp"] == 323


def test_battle712_neutralizing_gas_end_replays_grassy_surge_before_upkeep():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(27223),
        pool_get_team(1161),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1158009078,
        90,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 79, Weezing-Galar faints to Rillaboom's High Horsepower. When
    # Neutralizing Gas ends, Showdown immediately replays the surviving
    # Rillaboom's Grassy Surge before upkeep, so Grassy Terrain is active in
    # time for the same-turn residual heal from 261 to 282 HP.
    assert normal_rows[79]["p0_action"] == "move 1+forced:switch 5"
    assert normal_rows[79]["p0_active"] == 1
    assert normal_rows[79]["p0_hp"] == 208
    assert normal_rows[79]["p1_action"] == "move 2"
    assert normal_rows[79]["p1_active"] == 2
    assert normal_rows[79]["p1_hp"] == 282
    assert normal_rows[79]["p1_max_hp"] == 341


def test_battle716_switch_in_hazard_damage_triggers_same_switch_update_items():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(42365),
        pool_get_team(42699),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        704639688,
        50,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 39, Ogerpon-Wellspring faints and Scizor replaces it at 187/343.
    # Stealth Rock drops the switch-in to 145/343, and Showdown's post-runSwitch
    # Update immediately eats Sitrus Berry back to 230/343 before turn 40 starts.
    assert normal_rows[39]["p0_action"] == "move 4"
    assert normal_rows[39]["p0_active"] == 2
    assert normal_rows[39]["p0_hp"] == 301
    assert normal_rows[39]["p1_action"] == "move 3+forced:switch 4"
    assert normal_rows[39]["p1_active"] == 3
    assert normal_rows[39]["p1_hp"] == 230
    assert normal_rows[39]["p1_max_hp"] == 343


def test_battle752_fainted_target_skips_partial_trap_duration_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(39823),
        pool_get_team(32634),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2678666,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[4]["p0_action"] == "move 2+forced:switch 2"
    assert normal_rows[4]["p0_active"] == 1
    assert normal_rows[4]["p0_hp"] == 301
    assert normal_rows[4]["p1_action"] == "move 1"
    assert normal_rows[4]["p1_hp"] == 187

    assert normal_rows[5]["p0_action"] == "move 4"
    assert normal_rows[5]["p0_active"] == 1
    assert normal_rows[5]["p0_hp"] == 77
    assert normal_rows[5]["p0_max_hp"] == 301
    assert normal_rows[5]["p1_action"] == "move 3"
    assert normal_rows[5]["p1_hp"] == 187


def test_battle5_first_mover_recoil_is_not_double_applied_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(39303),
        pool_get_team(25661),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        275121931,
        10,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    assert normal_rows[3]["p0_action"] == "move 4"
    assert normal_rows[3]["p0_hp"] == 130
    assert normal_rows[3]["p1_action"] == "move 2"
    assert normal_rows[3]["p1_hp"] == 125

    assert normal_rows[4]["p0_action"] == "move 4"
    assert normal_rows[4]["p0_hp"] == 4
    assert normal_rows[4]["p1_action"] == "move 2+forced:switch 3"
    assert normal_rows[4]["p1_active"] == 0
    assert normal_rows[4]["p1_hp"] == 277


def test_battle127_first_mover_drain_replay_does_not_double_apply_rocky_helmet():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(46840),
        pool_get_team(47386),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        517480369,
        30,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 25 is Ceruledge's Bitter Blade into Rocky Helmet Landorus-T.
    # Showdown applies exactly one Rocky Helmet tick to the draining attacker
    # before the forced Heatran switch, leaving Ceruledge at 243/291.
    assert normal_rows[25]["p0_action"] == "move 1"
    assert normal_rows[25]["p0_hp"] == 243
    assert normal_rows[25]["p0_max_hp"] == 291
    assert normal_rows[25]["p1_action"] == "move 2+forced:switch 5"
    assert normal_rows[25]["p1_active"] == 2
    assert normal_rows[25]["p1_hp"] == 208
    assert normal_rows[26]["p0_hp"] == 243


def test_battle129_faster_recover_updates_slower_target_hp_snapshot_regression():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(41555),
        pool_get_team(44595),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        329175563,
        55,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 52, Toxapex's faster Recover heals to full before a fully
    # paralyzed Alomomola does nothing, so only burn should drop it to 285.
    assert normal_rows[52]["p0_action"] == "move 3"
    assert normal_rows[52]["p0_hp"] == 285
    assert normal_rows[52]["p0_max_hp"] == 303
    assert normal_rows[52]["p1_action"] == "move 4"
    assert normal_rows[52]["p1_hp"] == 140

    # Turn 53 uses the same faster Recover baseline, but Alomomola's Scald
    # lands into the healed HP snapshot before Black Sludge and burn.
    assert normal_rows[53]["p0_action"] == "move 3"
    assert normal_rows[53]["p0_hp"] == 284
    assert normal_rows[53]["p1_action"] == "move 4"
    assert normal_rows[53]["p1_hp"] == 140


def test_battle121_slower_knock_off_does_not_double_apply_faster_life_orb():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(44672),
        pool_get_team(5799),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1114239844,
        6,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 4, the faster Life Orb Moonblast user should only pay Life Orb
    # once before the slower Knock Off removes the item, ending at 196/289.
    assert normal_rows[4]["p0_action"] == "move 4"
    assert normal_rows[4]["p0_hp"] == 152
    assert normal_rows[4]["p0_max_hp"] == 394
    assert normal_rows[4]["p1_action"] == "move 1"
    assert normal_rows[4]["p1_hp"] == 196
    assert normal_rows[4]["p1_max_hp"] == 289


def test_battle181_canceled_slower_move_preserves_faster_recoil_hp_snapshot():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(8732),
        pool_get_team(41549),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        114636827,
        10,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 8, Staraptor's faster Brave Bird must keep its recoil-applied
    # HP (203/311) even though Hatterene's slower move is canceled by Eject
    # Button and Ceruledge comes in immediately.
    assert normal_rows[8]["p0_action"] == "move 2+forced:switch 2"
    assert normal_rows[8]["p0_active"] == 1
    assert normal_rows[8]["p0_hp"] == 291
    assert normal_rows[8]["p1_action"] == "move 2"
    assert normal_rows[8]["p1_hp"] == 203
    assert normal_rows[8]["p1_max_hp"] == 311


def test_battle210_first_mover_drain_is_not_removed_by_slower_side_status_move():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(36847),
        pool_get_team(16374),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        931786189,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 21, Polteageist's faster Giga Drain heals to 157/261 before the
    # slower Ting-Lu uses Spikes. The slower status move must not subtract from
    # that already-applied drain baseline.
    assert normal_rows[21]["p0_action"] == "move 2"
    assert normal_rows[21]["p0_hp"] == 402
    assert normal_rows[21]["p0_max_hp"] == 514
    assert normal_rows[21]["p1_action"] == "move 3"
    assert normal_rows[21]["p1_hp"] == 157
    assert normal_rows[21]["p1_max_hp"] == 261


def test_battle768_missing_hidden_burn_secondary_must_not_shift_followup_prng():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(45786),
        pool_get_team(4804),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        699850947,
        65,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 50's Blaze Kick burn-secondary frame is hidden in the Showdown log
    # because it does not proc, but omitting that frame shifts the later
    # Malignant Chain / Hex rolls. By turn 51, the byte-identical outcome is
    # Glimmora at 226/307 after Pecharunt's Hex.
    assert normal_rows[51]["p0_action"] == "move 4"
    assert normal_rows[51]["p0_hp"] == 226
    assert normal_rows[51]["p0_max_hp"] == 307
    assert normal_rows[51]["p1_action"] == "move 3"
    assert normal_rows[51]["p1_hp"] == 89
    assert normal_rows[51]["p1_max_hp"] == 379


def test_battle798_red_card_drag_must_clear_same_turn_partial_trap_before_residual():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(23657),
        pool_get_team(32803),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2032764815,
        20,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Toxapex's Infestation hits Glimmora, but Glimmora's Red Card immediately
    # drags Toxapex out. Showdown silently ends the source-tied partial trap
    # before residual order 13, so Glimmora stays at 364/370 on turn 14.
    assert normal_rows[14]["p0_action"] == "move 4"
    assert normal_rows[14]["p0_hp"] == 281
    assert normal_rows[14]["p0_max_hp"] == 281
    assert normal_rows[14]["p1_action"] == "move 1"
    assert normal_rows[14]["p1_hp"] == 364
    assert normal_rows[14]["p1_max_hp"] == 370


def test_battle801_player_red_card_drag_must_consume_hidden_tie_frames_before_next_turn():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(27545),
        pool_get_team(19197),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1779290972,
        25,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 1's Hurricane triggers the opponent Araquanid's Red Card and drags
    # in the player's Araquanid. Showdown spends the hidden switch/runSwitch
    # tie frames on that inline drag, so turn 2's same-speed Araquanid mirror
    # has the opponent's Leech Life act first, leaving the player at 274/340
    # and the opponent at 45/340 after Lunge.
    assert normal_rows[2]["p0_action"] == "move 2"
    assert normal_rows[2]["p0_hp"] == 274
    assert normal_rows[2]["p0_max_hp"] == 340
    assert normal_rows[2]["p1_action"] == "move 2"
    assert normal_rows[2]["p1_hp"] == 45
    assert normal_rows[2]["p1_max_hp"] == 340


def test_battle860_double_future_sight_must_resolve_in_target_slot_order():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(19184),
        pool_get_team(23090),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        625382177,
        45,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Both Slowking-Galar queue Future Sight into the opposing slot. Showdown
    # resolves slot conditions in target-slot order, so the side-0 target takes
    # the first delayed hit and the side-1 target takes the second one.
    assert normal_rows[37]["p0_action"] == "move 3"
    assert normal_rows[37]["p0_hp"] == 103
    assert normal_rows[37]["p0_max_hp"] == 394
    assert normal_rows[37]["p1_action"] == "move 1"
    assert normal_rows[37]["p1_hp"] == 288
    assert normal_rows[37]["p1_max_hp"] == 393


def test_battle873_failed_future_sight_turn_keeps_pre_residual_and_delayed_hit_updates():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(47275),
        pool_get_team(42359),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        842854548,
        40,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Showdown spends the slower failed Future Sight action's generic tied
    # Update before residuals, then the queued Future Sight hit spends its own
    # move-internal Updates. That keeps the delayed hit on the non-crit roll
    # and leaves the active Slowking-Galar at 262/394 on turn 8.
    assert normal_rows[8]["p0_action"] == "move 2"
    assert normal_rows[8]["p0_hp"] == 69
    assert normal_rows[8]["p0_max_hp"] == 394
    assert normal_rows[8]["p1_action"] == "move 3"
    assert normal_rows[8]["p1_hp"] == 262
    assert normal_rows[8]["p1_max_hp"] == 394
    assert normal_rows[9]["p0_action"] == "move 2+forced:switch 3"
    assert normal_rows[9]["p0_hp"] == 357
    assert normal_rows[9]["p0_max_hp"] == 357
    assert normal_rows[9]["p1_hp"] == 262


def test_battle876_same_turn_paralysis_blocks_slower_soft_boiled():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(30830),
        pool_get_team(1927),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        2009388159,
        12,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Zapdos's faster Discharge paralyzes Blissey, then Blissey's same-turn
    # BeforeMove paralysis roll blocks Soft-Boiled. The recovery move must not
    # heal after Showdown logs `cant|par`.
    assert normal_rows[7]["p0_action"] == "move 3"
    assert normal_rows[7]["p0_hp"] == 322
    assert normal_rows[7]["p0_max_hp"] == 384
    assert normal_rows[7]["p1_action"] == "move 1"
    assert normal_rows[7]["p1_hp"] == 593
    assert normal_rows[7]["p1_max_hp"] == 714
    assert normal_rows[7]["p1_status"] == 2


def test_battle878_target_eject_pack_cancels_forced_out_slow_move():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4163),
        pool_get_team(1326),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        720638507,
        8,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Ninetales-Alola's faster Moonblast lowers Hoopa's SpA, triggering Hoopa's
    # Eject Pack. Showdown switches Hoopa to Slowking-Galar immediately and
    # cancels Hoopa's queued slower Psychic, so Ninetales stays at 152/350.
    assert normal_rows[2]["p0_action"] == "move 3"
    assert normal_rows[2]["p0_hp"] == 152
    assert normal_rows[2]["p0_max_hp"] == 350
    assert normal_rows[2]["p1_action"] == "move 2+forced:switch 2"
    assert normal_rows[2]["p1_active"] == 1
    assert normal_rows[2]["p1_hp"] == 394
    assert normal_rows[2]["p1_max_hp"] == 394


def test_battle895_multihit_contact_hooks_use_actual_hits_after_early_ko():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(24088),
        pool_get_team(12222),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1212024052,
        22,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Cinccino's Triple Axel KOs Garchomp on hit 1. Showdown stops the
    # hitStepMoveHitLoop immediately, so Rough Skin only fires once and
    # Cinccino ends turn 17 at 255/291, not after three contact chunks.
    assert normal_rows[17]["p0_action"] == "move 3+forced:switch 5"
    assert normal_rows[17]["p0_active"] == 4
    assert normal_rows[17]["p0_hp"] == 400
    assert normal_rows[17]["p0_max_hp"] == 400
    assert normal_rows[17]["p1_action"] == "move 2"
    assert normal_rows[17]["p1_hp"] == 255
    assert normal_rows[17]["p1_max_hp"] == 291

    # Loaded Dice deletes Triple Axel's later-hit multiaccuracy checks in
    # Showdown. Those hidden PRNG frames first became visible in Lokix's turn
    # 21 First Impression roll into Cinderace.
    assert normal_rows[21]["p0_action"] == "switch 6"
    assert normal_rows[21]["p0_active"] == 5
    assert normal_rows[21]["p0_hp"] == 88
    assert normal_rows[21]["p0_max_hp"] == 301
    assert normal_rows[21]["p1_action"] == "move 2"
    assert normal_rows[21]["p1_hp"] == 220
    assert normal_rows[21]["p1_max_hp"] == 283


def test_battle902_lockedmove_fatigue_confusion_after_ko_update():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(35305),
        pool_get_team(15189),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1568744533,
        35,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Glastrier's locked Outrage KOs Slowking on turn 29. Showdown spends the
    # move-2 hit-loop Update before lockedmove.onEnd rolls fatigue confusion,
    # so the duration is 5 and Glastrier self-hits on turns 31-33 before ending
    # confusion on turn 34.
    assert normal_rows[29]["p0_action"] == "move 2+forced:switch 2"
    assert normal_rows[29]["p0_active"] == 1
    assert normal_rows[29]["p1_action"] == "move 4"
    assert normal_rows[29]["p1_hp"] == 243
    assert normal_rows[29]["p1_max_hp"] == 403
    assert normal_rows[31]["p1_hp"] == 206
    assert normal_rows[32]["p1_hp"] == 172
    assert normal_rows[33]["p1_hp"] == 139
    assert normal_rows[34]["p1_hp"] == 139


def test_battle580_lockedmove_fatigue_confusion_before_future_sight_after_ko():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(21551),
        pool_get_team(37180),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        978838227,
        6,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Braviary's Thrash KOs Slowking-Galar on turn 4. Showdown runs
    # lockedmove.onEnd and rolls fatigue confusion before the residual
    # Future Sight hit lands, so the delayed damage roll leaves Braviary at
    # 151/341 before Gholdengo replaces the fainted Slowking-Galar.
    assert normal_rows[3]["p0_action"] == "move 3"
    assert normal_rows[3]["p0_hp"] == 156
    assert normal_rows[3]["p1_action"] == "move 1"
    assert normal_rows[3]["p1_hp"] == 341
    assert normal_rows[4]["p0_action"] == "move 3+forced:switch 3"
    assert normal_rows[4]["p0_active"] == 2
    assert normal_rows[4]["p0_hp"] == 315
    assert normal_rows[4]["p0_max_hp"] == 315
    assert normal_rows[4]["p1_action"] == "move 1"
    assert normal_rows[4]["p1_hp"] == 151
    assert normal_rows[4]["p1_max_hp"] == 341


def test_battle915_future_sight_can_queue_into_fainted_target_slot():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(20249),
        pool_get_team(11230),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1737248993,
        60,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Turn 55 hard-switches Raging Bolt into Stealth Rock, where it faints
    # before Slowking-Galar's Future Sight runs. Showdown exempts futuremove
    # from the normal fainted-target no-target check, so the slot condition
    # still queues and later KOs Iron Moth on turn 57.
    assert normal_rows[55]["p0_action"] == "switch 5+forced:switch 5"
    assert normal_rows[55]["p0_active"] == 5
    assert normal_rows[55]["p0_hp"] == 134
    assert normal_rows[55]["p1_action"] == "move 1"
    assert normal_rows[56]["p0_action"] == "move 2"
    assert normal_rows[56]["p0_hp"] == 134
    assert normal_rows[56]["p1_hp"] == 264
    assert normal_rows[57]["p0_action"] == "move 2"
    assert normal_rows[57]["p0_hp"] == 0
    assert normal_rows[57]["p0_max_hp"] == 301
    assert normal_rows[57]["p1_action"] == "move 4"
    assert normal_rows[57]["p1_hp"] == 84
    assert normal_rows[57]["p1_max_hp"] == 394


def test_battle924_last_resort_requires_other_move_slots_since_switch_in():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(10172),
        pool_get_team(35289),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1170631017,
        6,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Furret has Tidy Up / Brick Break / Double-Edge / Last Resort. Showdown
    # resets moveSlot.used on switch-in and Last Resort fails until every
    # other known move slot has been used since that switch-in, so Cresselia
    # remains untouched while it queues Future Sight.
    assert normal_rows[1]["p0_action"] == "move 4"
    assert normal_rows[1]["p0_hp"] == 311
    assert normal_rows[1]["p1_action"] == "move 3"
    assert normal_rows[1]["p1_hp"] == 444
    assert normal_rows[2]["p0_action"] == "move 4"
    assert normal_rows[2]["p1_hp"] == 444
    assert normal_rows[3]["p0_action"] == "move 4"
    assert normal_rows[3]["p0_hp"] == 140
    assert normal_rows[3]["p1_hp"] == 444
    assert normal_rows[4]["p0_action"] == "move 4"
    assert normal_rows[4]["p1_hp"] == 444
    assert normal_rows[5]["p0_action"] == "move 4"
    assert normal_rows[5]["p1_hp"] == 444
    assert normal_rows[6]["p0_action"] == "move 4+forced:switch 6"
    assert normal_rows[6]["p1_hp"] == 444


def test_battle933_neutralizing_gas_suppresses_rock_head_recoil_immunity():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(19861),
        pool_get_team(39641),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        955664700,
        5,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Arcanine-Hisui has Rock Head and uses Flare Blitz into Weezing-Galar.
    # While Weezing's Neutralizing Gas is active on turn 3, Rock Head is
    # suppressed and Showdown applies recoil before Weezing's Sludge Wave.
    # On turn 4 Weezing faints from Flare Blitz before the recoil hook, so
    # Neutralizing Gas ends and Rock Head suppresses that second recoil.
    assert normal_rows[2]["p0_action"] == "switch 2"
    assert normal_rows[2]["p0_hp"] == 277
    assert normal_rows[2]["p1_hp"] == 334
    assert normal_rows[3]["p0_action"] == "move 1"
    assert normal_rows[3]["p0_hp"] == 165
    assert normal_rows[3]["p1_hp"] == 179
    assert normal_rows[4]["p0_action"] == "move 1"
    assert normal_rows[4]["p0_hp"] == 165
    assert normal_rows[4]["p1_action"] == "move 4+forced:switch 3"
    assert normal_rows[4]["p1_active"] == 2


def test_battle941_residual_faint_skips_post_residual_update_frame():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(48794),
        pool_get_team(30955),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        1808513351,
        13,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # The opposing Alomomola faints from burn residual on turn 10. Showdown
    # does not run the final post-residual Update speedSort with only one
    # active remaining, so the next turn's Scald secondary roll misses burn.
    assert normal_rows[10]["p0_action"] == "move 1"
    assert normal_rows[10]["p1_action"] == "move 4+forced:switch 4"
    assert normal_rows[10]["p1_active"] == 1
    assert normal_rows[11]["p0_action"] == "move 1"
    assert normal_rows[11]["p1_action"] == "switch 5"
    assert normal_rows[11]["p1_active"] == 4
    assert normal_rows[11]["p1_hp"] == 243
    assert normal_rows[11]["p1_status"] == 0
    assert normal_rows[12]["p0_hp"] == 103
    assert normal_rows[12]["p1_hp"] == 174
    assert normal_rows[12]["p1_status"] == 1


def test_battle951_notarget_status_move_skips_accuracy_frame():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(4487),
        pool_get_team(36192),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        166292726,
        66,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # On turn 63, Scream Tail's Encore has no target after Gothitelle faints
    # from Struggle recoil. Showdown fails it before hitStepAccuracy, so the
    # next turn's Gunk Shot uses the expected accuracy frame and hits.
    assert normal_rows[63]["p0_action"] == "move 1+forced:switch 2"
    assert normal_rows[63]["p1_action"] == "move 4"
    assert normal_rows[63]["p0_active"] == 1
    assert normal_rows[64]["p0_action"] == "move 4"
    assert normal_rows[64]["p1_action"] == "move 4"
    assert normal_rows[64]["p1_hp"] == 129
    assert normal_rows[64]["p1_status"] == 2
    assert normal_rows[65]["p1_action"] == "switch 2"
    assert normal_rows[65]["p1_active"] == 1
    assert normal_rows[65]["p1_hp"] == 298


def test_battle964_mold_breaker_good_as_gold_still_reaches_accuracy():
    gd = load_game_data()
    me = load_move_effect_data()
    mappings = load_id_mappings()

    rows, _, _ = run_pokepy_battle(
        pool_get_team(18217),
        pool_get_team(45622),
        gd,
        me,
        mappings,
        MODERN_TYPE_CHART,
        560102504,
        16,
    )

    normal_rows = {row["turn"]: row for row in rows if row.get("type") == "normal"}

    # Tinkaton's Mold Breaker suppresses Good as Gold on turn 5, so Encore
    # reaches accuracy before failing later against the just-switched target.
    # That PRNG frame prevents turn 6 Make It Rain from becoming a false crit.
    assert normal_rows[5]["p0_action"] == "move 2"
    assert normal_rows[5]["p1_action"] == "switch 3"
    assert normal_rows[5]["p1_active"] == 2
    assert normal_rows[6]["p0_action"] == "move 1"
    assert normal_rows[6]["p1_action"] == "move 3"
    assert normal_rows[6]["p0_hp"] == 169
    assert normal_rows[6]["p1_hp"] == 221
    assert normal_rows[7]["p0_hp"] == 25
    assert normal_rows[8]["p0_action"] == "move 1+forced:switch 6"
    assert normal_rows[8]["p0_active"] == 5
    assert normal_rows[14]["p0_action"] == "move 2"
    assert normal_rows[14]["p1_action"] == "move 4"
    assert normal_rows[14]["p1_hp"] == 391
