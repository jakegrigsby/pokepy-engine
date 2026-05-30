# Engine build contracts

Locked decisions for the modular pokepy engine (gens 1, 2, 3, 4, 9).

## State

- **Representation:** scalar `np.ndarray` int16 buffer (`MultiFormatState.battle_state`), layout in `pokepy/core/constants.py`.
- **Accessors:** `pokepy/core/bitpack.py` (boosts, status, hazards, protect, volatiles).
- **Wrapper:** `MultiFormatState` in `pokepy/core/state.py` holds team metadata, phase, side order, fog-of-war reveal flags.

## Data source

- **Runtime tables:** `pokepy/data/extracted/` (+ `extracted/gen{N}/` for gen < 9), loaded via `pokepy/data/loader.py`.
- **Move effects:** `pokepy/data/move_effects.py` via `load_move_effect_data(gen=...)`.
- **Type charts:** `pokepy/data/type_charts.py` via `load_type_chart_for_gen(gen)`.
- **NOT used at runtime:** `pokepy/data/showdown/*.json` (reference only).

## Effect call signature

Effect helpers in `pokepy/effects/*` are free functions mutating the flat buffer in place:

```python
fn(battle: np.ndarray, ..., gen5_prng: Gen5PRNG | None = None, game_data=..., **kwargs) -> Any
```

The dispatch registry binds `(table, id, event) -> fn` where `table` is one of
`ability`, `item`, `move`, `condition`, `volatile`. Handlers are invoked through
`BitpackBattleContext.single_event` / `run_event`.

## Public engine API (stable for env + tests)

| Symbol | Module | Purpose |
|--------|--------|---------|
| `init_battle_state` | `pokepy.env.battle_env` | Team -> `MultiFormatState` |
| `step_battle` | `pokepy.engine` | One turn, gen-keyed |
| `step_forced_switch_for_gen` | `pokepy.engine` | Forced switch subturn |
| `get_action_mask` | `pokepy.engine.action_mask` | Legality mask (10 battle actions) |
| `get_battle_action_mask` | `pokepy.engine.action_mask` | Alias |
| `ENGINE_REGISTRY` | `pokepy.engine` | `{gen: EngineEntry}` |

## Metamon bridge

- **Observation:** `pokepy_state_to_universal(state, side, game_data, mappings)` in `metamon/env/pokepy_battle/state_adapter.py` -> `UniversalState`.
- **Actions:** 13-action `UniversalAction` space via `metamon/env/pokepy_battle/action_adapter.py` (4 moves + 5 switches + tera variants).
- **Teams:** `team_set_to_pokepy_dict` in `team_adapter.py`.
- **Env:** `VectorizedPokepyEnv` in `vector_env.py` drives `step_battle` + forced-switch generator.

## Generation profiles

`GenProfile` in `pokepy/core/gen_profile.py` gates abilities, items, tera, terrain,
teampreview, hazard set, crit tables, para rolls, powder immunity. Gen-specific
branches live in `pokepy/engine/gen_mods.py` and are selected by `profile_for_gen`.

## Modular layout

```
pokepy/engine/
  dispatch.py      # runEvent / singleEvent / speedSort on bitpack state
  registry.py      # (table, id, event) -> effect fn
  queue.py         # turn action queue + speed ordering
  turn_loop.py     # start / commitChoices / runAction / endTurn / residual
  move_pipeline.py # runMove .. getDamage .. modifyDamage .. secondaries
  switch.py        # switch-in / faint / forced-switch + request model
  gen_mods.py      # gen 1-4 control-flow + mechanic branches
  battle_gen9.py   # legacy monolith (delegates to modular entry points)
```

Playability is the acceptance gate for this phase; PRNG frame-exact parity is deferred.
