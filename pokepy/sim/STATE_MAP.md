# Packed state map for Showdown effectState fields (gens 1–4)

Source of truth for `pokepy/sim/views.py`. Offsets from `pokepy/core/constants.py`.

## Per-Pokémon slot (+0..+15 relative to slot base)

| Field | Offset | Accessor |
|-------|--------|----------|
| species | +0 | direct `battle[base+0]` |
| hp / maxhp | +1 / +2 | direct |
| level | +3 | direct |
| types | +4 | low/high byte |
| ability | +5 | direct |
| item | +6 | direct |
| stats atk..spe | +7..+11 | direct |
| status id | +12 low byte | `bitpack.get_status(battle[base+12])` |
| status turns (slp time, tox counter display) | +12 high byte | `bitpack.get_status_turns(battle[base+12])` |
| stat boosts | +13, +14 | `bitpack.extract_boost` / `apply_boost_to_packed` |
| flags (fainted, active, tera, truant, charge, …) | +15 | direct bitmask |

## Side field volatiles (relative to `OFF_FIELD`)

Side 0 uses `F_*_0`, side 1 uses `F_*_1`.

| Showdown condition / volatile | effectState fields | Storage |
|------------------------------|-------------------|---------|
| **slp** | `time`, `startTime` | `+12` high byte via `set_status(STATUS_SLEEP, turns)`; gen1 wake uses flag timing in handler |
| **par** | (none mutable) | status id `STATUS_PARALYSIS` at `+12` low byte |
| **frz** | (none mutable) | status id `STATUS_FREEZE` at `+12` low byte |
| **brn** | (none mutable) | status id `STATUS_BURN` at `+12` low byte |
| **psn** | (none mutable) | status id `STATUS_POISON` at `+12` low byte |
| **tox** | `stage` | status id `STATUS_TOXIC` + counter in `+12` high byte |
| **confusion** | `time` | `F_VOLATILE_x` bit `VOLATILE_CONFUSION`; turns in high bits of volatile word (see `bitpack` confusion helpers) |
| **flinch** | (single turn) | `F_VOLATILE_x` bit `VOLATILE_FLINCH` |
| **taunt** | `time` | `F_VOLATILE_x` `VOLATILE_TAUNT` + duration in volatile word |
| **encore** | `move`, `time` | `F_VOLATILE_x` `VOLATILE_ENCORE` + `F_LAST_MOVE_x` / duration bits |
| **disable** | `move`, `time` | `F_DISABLE_x` move id, `F_DISABLE_TURNS_x` |
| **partiallytrapped** | `time` | `OFF_MOVES` `M_PARTIAL_TRAP_TURNS_x` + `EXT_VOL_PARTIAL_TRAP` |
| **lockedmove** | `time` | `OFF_MOVES` `M_LOCKED_MOVE_x`, `M_LOCKED_TURNS_x` |
| **mustrecharge** | (flag) | `OFF_MOVES` `M_RECHARGE_x` |
| **twoturnmove** | semi-invuln | `M_ACTIVE_MOVE_ACTIONS_x` bit 14 (`ACTIVE_MOVE_ACTIONS_SEMI_INVUL`) |
| **leechseed** | (source side) | `F_LEECH_SEED_x` |
| **substitute** | hp | `F_SUBSTITUTE_x` |
| **perishsong** | `time` | `F_PERISH_COUNT_x` |
| **yawn** | `time` | `F_YAWN_TURNS_x` + `EXT_VOL_YAWN` |
| **destinybond** | (flag) | `F_DESTINY_BOND_x` |
| **attract** | (flag) | `EXT_VOL_ATTRACT` |
| **torment** | (flag) | `EXT_VOL_TORMENT` |
| **focusenergy** | (flag) | `EXT_VOL_FOCUS_ENERGY` |
| **trapped** / **meanlook** | (flag) | `EXT_VOL_MEAN_LOOK` |
| **curse** (ghost) | (flag) | `EXT_VOL_CURSE` |
| **ingrain** | (flag) | `EXT_VOL_INGRAIN` |
| **aquaring** | (flag) | `EXT_VOL_AQUA_RING` |
| **nightmare** | (flag) | `VOLATILE_NIGHTMARE` in volatile word |
| **protect** | consecutive uses | `F_PROTECT_x` via `bitpack` protect helpers |
| **choice lock** | move | `F_CHOICE_LOCK_x` |

## Field / side conditions

| Condition | Storage |
|-----------|---------|
| weather + duration | `F_WEATHER`, `M_WEATHER_TURNS` |
| terrain + duration | `F_TERRAIN`, `M_TERRAIN_TURNS` |
| trick room | `F_TRICK_ROOM` |
| turn counter | `F_TURN` |
| hazards (SR/spikes/tspikes/web) | `F_HAZARDS_0/1` packed via `bitpack` hazard helpers |
| screens (reflect/light screen/aurora veil/tailwind/safeguard/mist) | `F_SCREENS_0/1` |
| wish | `M_WISH_TURNS_*`, `M_WISH_HP_*` |
| future sight | `M_FUTURE_SIGHT`, `M_FUTURE_MOVE_*` |

## Extended volatile free bits

`F_EXTENDED_VOLATILE_*` uses bits defined in `constants.py`. All defined `EXT_VOL_*` masks are allocated; new gen9-only volatiles should reuse existing slots before adding bits. **NEEDS SLOT** only if a new gen5+ volatile has no mapping (not required for gen1–4 slice).

## Gen-specific notes

- **Gen1 slp**: wake on `onAfterMoveSelf`, not EOT decrement — handler in `sim/conditions.py` / `sim/mods/gen1.py`.
- **Gen3/4 slp**: decrement in `onBeforeMove` — same `+12` high byte.
- **Gen2/3 Quick Claw**: endTurn roll (legacy) or item handler gen4+ — item id at `+6`, roll at endTurn in mod.
