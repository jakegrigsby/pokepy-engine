# Verbatim Showdown Port — Translation Guide (Phase B handoff)

This is the working manual for the **bulk translation phase** of the verbatim
Showdown port. Phase A (skeleton + dispatch + move pipeline + one bit-exact
gen9 vertical slice) is **done**. Your job in Phase B is breadth: translate the
large callback universe (moves / abilities / items / conditions) and the
per-generation mechanics, then wire the full adapter + env, until the live-diff
battery reaches the baseline in `BASELINE.md`.

**Do not refactor the dispatch core or the move pipeline.** They are a faithful
port of Showdown's `sim/battle.ts` + `sim/battle-actions.ts` and are validated
frame-exact (see "Why this works" below). Add *data* and *callbacks*; don't
restructure the engine.

---

## 1. Architecture map

```
pokepy/showdown/
  dex.py            Dex / DexTable / DexEntry / ConditionsTable + get_active_move
  util.py           trunc / clampIntRange / modify / chain  (Showdown arithmetic)
  field.py          Field (weather / terrain / pseudo-weather)
  pokemon.py        Pokemon (stats, types, status, boosts, hp, ability/item)
  side.py           Side (team / active / side conditions)
  battle_queue.py   BattleQueue (resolveAction / insertChoice / sort) + Action
  active_move.py    ActiveMove (mutable per-use copy of a move)
  battle.py         Battle: this-context + event dispatch + API shim + turn loop
  battle_actions.py BattleActions: runMove -> ... -> getDamage -> modifyDamage
  effects/__init__.py   register() decorator + handler registry  <-- YOUR WORK
  adapter.py        minimal parity driver (Phase A); full UniversalState = Phase B

pokepy/data/showdown/<gen>/   moves|species|abilities|items|conditions|natures|typechart .json
  (extracted from Showdown's own Dex by scripts/extract_dex_json.js,
   function bodies stripped — those become Python in effects/)
```

The data JSON carries every *data* field (basePower, type, secondary, boosts,
status, recoil, drain, accuracy, ...). The move pipeline reads those directly.
What it **cannot** carry is the function bodies (`onHit`, `onModifyMove`,
`basePowerCallback`, `onResidual`, ...). Those are dropped from the dump and are
exactly what you hand-translate into `effects/`.

---

## 2. How a translated callback gets called

Register `on_*` methods keyed by `(table, id)`, optional per-gen override:

```python
from pokepy.showdown.effects import register

@register("moves", "brickbreak")          # base (all gens)
class BrickBreak:
    @staticmethod
    def on_try_hit(battle, target, source, move):
        # `battle` is the this-context (Showdown's `this`); read it like JS.
        ...

@register("conditions", "brn", gen=1)      # gen1-specific override
class Gen1Burn:
    @staticmethod
    def on_residual(battle, pokemon): ...
```

- The `Dex` merges the registered handlers onto each entry as
  `entry.handlers` (`event_name -> callable`). The dispatch core
  (`find_event_handlers` / `single_event` / `run_event`) finds and calls them.
- **Method-name → event-name** is snake_case of `on<Event>`:
  `on_residual` → `onResidual`, `on_modify_atk` → `onModifyAtk`,
  `on_try_hit` → `onTryHit`, `on_base_power` → `onBasePower`.
- **Calling convention:** handlers are called as `callback(battle, *args)` where
  `battle` is the Battle (Showdown's `this`). For `single_event` the args are
  `(relay_var?, target, source, source_effect)`; for `run_event` likewise. Match
  the TypeScript signature minus `this`. A `None` return means "no change"
  (Showdown's `undefined`); return a value to override the relay var.
- **Order/priority hints:** put `onModifyAtkPriority = -1` etc. as plain class
  attributes if Showdown declares them; `resolve_priority` reads
  `handlers[f"{cb}Priority"]` / `Order` / `SubOrder`.

Data-driven fields need **no** callback. Only translate genuinely custom logic.

### Worked slice (A5 acceptance)

The slice test (`tests/test_showdown_engine_live_diff.py`) runs a **gen9 Alakazam
mirror** spamming Psychic (move id 94). No custom move/ability callbacks are
required for that path — damage, accuracy, crit, and secondaries are all
**data-driven** from `pokepy/data/showdown/gen9/moves.json`.

When you *do* need a callback, follow the burn example (already translated):

**Showdown** (`server/pokemon-showdown/data/conditions.ts`):

```typescript
brn: {
  onStart(target, source, sourceEffect) {
    if (sourceEffect && sourceEffect.id === 'flameorb') {
      this.add('-status', target, 'brn', '[from] item: Flame Orb');
    } else if (sourceEffect && sourceEffect.effectType === 'Ability') {
      this.add('-status', target, 'brn', '[from] ability: ' + sourceEffect.name, `[of] ${source}`);
    } else {
      this.add('-status', target, 'brn');
    }
  },
  onResidualOrder: 10,
  onResidual(pokemon) {
    this.damage(pokemon.baseMaxhp / 16);
  },
},
```

**Python** (`pokepy/showdown/effects/conditions_burn.py`):

```python
@register("conditions", "brn")
class Burn:
    onResidualOrder = 10

    @staticmethod
    def on_start(battle, target, source, source_effect):
        if source_effect and getattr(source_effect, "id", None) == "flameorb":
            battle.add("-status", target, "brn", "[from] item: Flame Orb")
        elif source_effect and getattr(source_effect, "effectType", None) == "Ability":
            battle.add("-status", target, "brn", "[from] ability: " + source_effect.name, f"[of] {source}")
        else:
            battle.add("-status", target, "brn")

    @staticmethod
    def on_residual(battle, pokemon):
        battle.damage(pokemon.baseMaxhp // 16, pokemon)
```

Mechanical rules visible here:
- `this.` → `battle.` (Battle is the Showdown `this`-context).
- `onStart` → `on_start` method; registry maps to `onStart` event name.
- Priority/order hints (`onResidualOrder`) become plain class attributes; numeric
  priority fields (`onModifyAtkPriority`) likewise.
- Integer division: Showdown's `/ 16` on integers → Python `// 16`.

Add new translations as sibling modules under `effects/`; they auto-import on
package load. Tick them off in `COVERAGE.md`.

### API shim vocabulary (`this` → `battle`)

These are the **supported** Showdown runtime methods on the Battle object and
its children. If a callback calls something not listed, port it to `battle.py`
(or `pokemon.py` / `side.py`) before using it — don't stub silently.

**Battle (`battle.*` / Showdown `this.*`):**

| Showdown | Python | Notes |
|----------|--------|-------|
| `this.random()` / `this.random(n)` | `battle.random()` | Gen5 LCG |
| `this.randomChance(a,b)` | `battle.random_chance(a,b)` | |
| `this.sample(arr)` | `battle.sample(arr)` | |
| `this.trunc` / `clampIntRange` / `modify` / `chain` / `chainModify` / `finalModify` | same snake_case | in `util.py`, bound on Battle |
| `this.randomizer(dmg)` | `battle.randomizer(dmg)` | damage roll |
| `this.getCategory(move)` | `battle.get_category(move)` | |
| `this.runEvent` / `this.singleEvent` / `this.eachEvent` | `run_event` / `single_event` / `each_event` | |
| `this.boost` | `battle.boost` | |
| `this.damage` / `this.heal` / `this.faint` | same | |
| `this.add(...)` | `battle.add(...)` | log lines (no-op in slice) |
| `this.dex.*` | `battle.dex.*` | see Dex below |
| `this.actions.runMove` etc. | `battle.actions.*` | move pipeline |
| `this.queue.*` | `battle.queue.*` | action queue |
| `this.field.*` | `battle.field.*` | weather/terrain |
| `this.sides` / active lookup | `battle.sides` | |
| `this.gen` / `this.turn` / `this.ended` | same | |

**Dex (`battle.dex.*`):**

| Showdown | Python |
|----------|--------|
| `dex.moves.get(id)` | `battle.dex.moves.get(id)` → `DexEntry` |
| `dex.species.get(id)` | same pattern |
| `dex.abilities.get` / `items.get` / `types.get` / `natures.get` | same |
| `dex.conditions.get(id)` / `getByID` | `conditions.get` / `get_by_id` |
| `dex.getActiveMove(move)` | `get_active_move` |
| `dex.getEffectiveness` / `getImmunity` | `get_effectiveness` / `get_immunity` |

**Pokemon (common in callbacks):**

`hp`, `maxhp`, `baseMaxhp`, `status`, `boosts`, `types`, `ability`, `item`,
`volatiles`, `fainted`, `isActive`, `side`, `getStat`, `getTypes`, `hasType`,
`hasAbility`, `hasItem`, `setStatus`, `clearStatus`, `boostBy`, `damage`, `heal`.

**Gap convention:** if you reach code that *must* call an unported callback, use:

```python
from pokepy.showdown.effects import unported
unported("moves", "return", "BasePower")  # raises NotImplementedError
```

The worklist of Showdown callbacks per entry lives in
`pokepy/data/showdown/genN/callbacks.json` (regenerate via
`node scripts/extract_dex_json.js`).

### Gen inherit-chain map

Showdown layers generations as mods on top of gen9 `BASE_MOD`:

```
gen9 (BASE_MOD)  ←  data/*.ts + sim/battle-actions.ts (modern pipeline)
  ↑ overridden by
gen8 … gen5       ←  data/mods/genN/ (mechanics deltas)
  ↑
gen4              ←  hit-step reorder, gen4 modifyDamage quirks
gen3              ←  accuracy scripts, old stat stages in places
gen2              ←  type chart, crit, sleep
gen1              ←  no team preview, special stat, badge boosts
```

In Python:
- **Data** differences: separate JSON per gen in `pokepy/data/showdown/genN/`.
- **Behavior** differences: `@register(table, id, gen=N)` overrides in `effects/`,
  plus `if battle.gen <= N` branches already in `battle.py` / `battle_actions.py`.
- **Gen scripts**: read `server/pokemon-showdown/data/mods/genN/scripts.ts` for
  Battle/BattleActions method overrides; port those as engine branches or gen-
  scoped effect registrations — do not invent new abstractions.

### Gen inheritance
`get_handlers(gen, table, id)` returns the base handlers merged under the gen
override. So gen9 mons use the base registration; a gen with different behavior
registers `gen=N` and only overrides the events that changed. This mirrors
Showdown's `mods/genN` inheritance chain.

---

## 3. The PRNG frame model — why this works (READ THIS)

Correctness here is **frame parity**: every `prng.random()` the engine consumes
must match Showdown's, in order. Showdown has a built-in trace
(`POKEPY_PRNG_TRACE=1` on the CLI emits `PRNGTRACE <raw32> ... <callstack>` to
stderr); pokepy's `Gen5PRNG` logs `(value, caller)` to
`pokepy.utils.gen5_prng._PRNG_TRACE_LOG` under the same env var. Diff the two raw
sequences to localize any divergence.

The validated gen9 singles mirror sequence (per `start()` + one `choose`):

```
startup (4 frames):
  1  shuffle   commitChoices.sort   (team-preview actions tie)
  2  random    switchIn.insertChoice (2nd lead's runSwitch inserted into a range)
  3  shuffle   runSwitch.speedSort  (allActive tie)
  4  shuffle   eachEvent('Update')  (runSwitch tail)
turn (per turn):
  -  shuffle   commitChoices.sort   (move actions tie = turn order)
  -  shuffle   eachEvent('BeforeTurn')
  -  shuffle   eachEvent('Update')  (beforeTurn tail)
  -  shuffle   gen8 queue re-sort   (next action is a move)
  per move:    accuracy, crit, damage-roll, secondary,
               + 2x eachEvent('Update') inside moveHitLoop,
               + 1x eachEvent('Update') in the runAction tail
  -  shuffle   eachEvent('Update')  (residual tail)
```

Key facts that bite if you forget them:
- **Speed ties consume a PRNG shuffle frame.** A mirror match ties on speed, so
  `eachEvent('Update')` (called all over the place) shuffles every time. The
  turn loop in `battle.py` reproduces Showdown's exact `eachEvent`/sort call
  sites — don't add or remove them.
- **`move.critRatio` defaults to 1, not 0**, so a normal move *does* roll for
  crit (`randomChance(1, 24)` in gen9) — that frame is real.
- Python truthiness traps: `False == 0` and `False in [0]` are `True` in Python
  but not JS. The pipeline uses `is False` / `is None` checks deliberately;
  keep that discipline when porting `=== false` / `!== 0`.
- `Pokemon.damage()` must **not** queue the faint; only `Battle.faint()` does
  (otherwise the faint never registers — see the Phase A bug history).

---

## 4. Adding a generation

1. Ensure `pokepy/data/showdown/genN/` exists (run `scripts/extract_dex_json.js`;
   it already emits gens 1/2/3/4/9 — add more gens there if needed).
2. Gen-specific *data* differences come for free from the per-gen JSON.
3. Gen-specific *mechanics* are either:
   - branch points already in the engine keyed on `battle.gen` (e.g. crit
     multipliers, gen<=6 hit-step swap, gen8 dynamic-speed re-sort), or
   - `@register(..., gen=N)` callback overrides.
4. **gen<5 has no team preview**: `start()` already branches (`turn_loop()`
   directly instead of the team-preview commit). The startup frame budget
   differs for gens 1-4 — re-derive it from the trace before chasing damage
   diffs.

---

## 5. Acceptance commands

```bash
cd metamon/env/pokepy-engine

# Phase-A vertical slice (must stay green): gen9 mirror, bit-exact vs Showdown
python -m pytest tests/test_showdown_engine_live_diff.py -q

# Localize a frame divergence (raw PRNG sequence, engine side):
POKEPY_PRNG_TRACE=1 python -c "import pokepy.utils.gen5_prng as P; ..."
# Showdown side:
POKEPY_PRNG_TRACE=1 server/pokemon-showdown/pokemon-showdown simulate-battle < script.txt 2>trace.txt

# Showdown ground truth helpers (engine-agnostic):
#   scripts/showdown_ref.py  -> team_to_showdown_packed / run_showdown /
#                               _parse_showdown_log / compare_battle_rows
```

The Phase-B target is the full multigen live-diff battery reaching the
`BASELINE.md` numbers (≥ 98/156 on `tests/test_pokepy_multigen_live_diff.py`
once that harness is re-pointed at the new engine — see `COVERAGE.md`).

---

## 6. Ground rules

- Translate **verbatim**. When in doubt, open the Showdown source
  (`server/pokemon-showdown/sim/` and `data/`) and port it line-for-line.
- Prefer many small `@register` translations over clever generalizations.
- Keep `battle`-as-`this`: a translated callback should read like the TS source
  with `this.` → `battle.` and `this.battle.` → `battle.`.
- Re-run the slice test after every batch; if a frame diverges, trace it.
