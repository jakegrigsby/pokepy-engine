# Port Coverage — what's done vs. what Phase B owes

Living checklist for the verbatim Showdown port. Update it as you translate.
"Done" means *ported verbatim and exercised by a passing test or the live-diff
slice*; "stub" means present but raises / returns a placeholder; "TODO" means
not started.

Target to declare Phase B complete: **≥ 98 / 156** on
`tests/test_pokepy_multigen_live_diff.py` across gens 1/2/3/4/9 (see `BASELINE.md`).

---

## Engine core (Phase A — DONE, do not refactor)

| Area | File | State |
|------|------|-------|
| Dex / DexTable / DexEntry / get_active_move | `dex.py` | done |
| Showdown arithmetic (trunc/clamp/modify/chain) | `util.py` | done |
| Field (weather/terrain/pseudo-weather) | `field.py` | done (data); custom weather callbacks = effects/ |
| Pokemon (stats/types/boosts/hp/status/item/ability) | `pokemon.py` | done (core); some `on*` hooks pending |
| Side (team/active/side conditions) | `side.py` | done (core) |
| BattleQueue (resolveAction/insertChoice/sort) | `battle_queue.py` | done |
| ActiveMove | `active_move.py` | done |
| Event dispatch (findEventHandlers/single/run/resolvePriority/speedSort) | `battle.py` | done |
| Battle API shim (this-context, add/damage/faint/boost/heal/...) | `battle.py` | partial — add shims as callbacks need them |
| Turn loop (start/choose/commit/turnLoop/runAction/residual) | `battle.py` | done, frame-exact for gen9 singles |
| Move pipeline runMove→...→getDamage→modifyDamage→secondaries→recoil | `battle_actions.py` | done |
| switch_in / run_switch | `battle_actions.py` | done (single-active); multi-switch request model = Phase B B3 |

## Data tables (extracted; function bodies are effects/ work)

| Gen | moves species abilities items conditions natures typechart |
|-----|------------------------------------------------------------|
| 9   | present |
| 4   | present |
| 3   | present |
| 2   | present |
| 1   | present |
| 5–8 | **TODO** — run `scripts/extract_dex_json.js` for these gens if the target battery needs them |

## Effect callbacks (`effects/` — THE BULK OF PHASE B)

Registry + decorator: **done**. Translated callbacks: **burn (`brn`) only** —
see `effects/conditions_burn.py` as the template.

**Callback worklist:** `pokepy/data/showdown/genN/callbacks.json` lists every
move/ability/item/condition id that has function bodies in Showdown's Dex for
that gen. Regenerate with `node scripts/extract_dex_json.js`. Tick entries in
this file as you translate them.

Translate, by rough priority (highest live-diff leverage first):

- [x] **conditions/brn** — `effects/conditions_burn.py`
- [x] **conditions/par, psn, tox, slp, frz** — `effects/conditions_status.py` (gen9 base)
- [ ] **conditions**: weather, terrain, volatiles, remaining edge cases
      `slp`, `frz`; weather (`sand`, `hail`/`snow`, `sun`, `rain`); terrains;
      `confusion`, `flinch`, `partiallytrapped`, `futuremove`, etc.
- [ ] **moves** with custom logic: `basePowerCallback` moves (e.g. `return`,
      `gyroball`, `lowkick`), `onModifyMove` / `onTryHit` / `onHit` /
      `onEffectiveness` / `onModifyType` moves, multi-hit special cases,
      protect family, hazards (`spikes`/`stealthrock`/...), recoil/drain edge
      cases the data fields don't cover.
- [ ] **abilities**: stat/​type/​power modifiers (`hugepower`, `levitate`,
      `intimidate`, `sturdy`, `multiscale`, weather setters, immunities, ...).
- [ ] **items**: berries, choice items, `lifeorb`, `leftovers`, type plates,
      `focussash`, `rockyhelmet`, gems, etc.

For each: open Showdown source (`server/pokemon-showdown/data/<table>.ts`,
gen mods in `data/mods/genN/`), translate the function bodies verbatim into a
`@register(table, id[, gen=N])` class, mapping `this.`→`battle.`. See
`TRANSLATION_GUIDE.md` §2.

## Generation mechanics

| Gen | State |
|-----|-------|
| 9 (singles) | mirror turn bit-exact (slice test); breadth = B1 |
| 4/3 | data present; mechanics branches + callbacks = B2 |
| 2/1 | data present; gen1/2 crit/stat/badge/typechart quirks = B2 (no team preview; re-derive startup frame budget) |

## Adapter / env contract (Phase B B3)

| Piece | File | State |
|-------|------|-------|
| Parity driver (heuristic + forced switch) | `adapter.py` | done — re-points multigen harness |
| Full UniversalState adapter | — | **TODO** |
| Request / forced-switch model | — | **TODO** |
| team / action internals rewrite | metamon glue | **TODO** (quarantined stubs raise) |
| vector_env wiring | metamon glue | **TODO** |

## Tests

| Test | State |
|------|-------|
| `tests/test_showdown_engine_live_diff.py` (gen9 mirror, bit-exact) | **passing** — keep green |
| `tests/test_pokepy_multigen_live_diff.py` (full battery) | re-point at new engine in B; target ≥98/156 |
| legacy packed-engine tests | obsolete; fixtures raise RuntimeError (see `tests/conftest.py`) |

---

### How to make progress measurable
1. Pick a gen + a failing live-diff case.
2. Run it, trace the PRNG divergence (`TRANSLATION_GUIDE.md` §3) **or** spot the
   missing data/callback in the log.
3. Translate the smallest callback that fixes it; re-run.
4. Keep the gen9 slice test green throughout.
