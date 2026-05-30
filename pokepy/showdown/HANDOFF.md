# Phase A → Phase B handoff

**Status: Phase A complete. Hard pause — switch to the cheaper bulk-translation agent.**

## What Phase A delivered

- Deleted the old hand-vectorized engine; new package at `pokepy/showdown/`.
- Faithful event dispatch + Battle API shim + gen9 move pipeline (verbatim port).
- Per-gen Dex JSON (gens 1/2/3/4/9) + callback manifest (`data/showdown/genN/callbacks.json`).
- One **bit-exact** vertical slice: gen9 Alakazam mirror vs live Showdown
  (`tests/test_showdown_engine_live_diff.py`, 4 tests, all green).
- One worked callback translation: burn (`effects/conditions_burn.py`).
- Minimal parity adapter (`adapter.run_engine_battle`); full `UniversalState` is Phase B.

## What Phase B must do

1. **B1** — Translate gen9 callbacks (moves/abilities/items/conditions) using the
   slice template; iterate until gen9 live-diff reaches baseline.
2. **B2** — Layer gen4/3/2/1 mods via `@register(..., gen=N)` + engine branches.
3. **B3** — Implement `adapter.showdown_state_to_universal`; rewrite
   `metamon/env/pokepy_battle/{vector_env,team_adapter,action_adapter}.py` internals.
4. **B4** — Reach **≥ 98 / 156** on the full multigen battery; flip default; delete stubs.

## Read these first

| File | Contents |
|------|----------|
| [`TRANSLATION_GUIDE.md`](TRANSLATION_GUIDE.md) | How to translate; PRNG frame model; API shim; inherit chain |
| [`COVERAGE.md`](COVERAGE.md) | Done / TODO checklist |
| [`BASELINE.md`](BASELINE.md) | Target numbers + pre-deletion capture |
| [`effects/conditions_burn.py`](effects/conditions_burn.py) | Worked TS → Python example |

## Acceptance commands

```bash
cd metamon/env/pokepy-engine

# Must stay green after every batch (Phase-A gate)
python -m pytest tests/test_showdown_engine_live_diff.py -q

# Phase-B target (re-point harness driver as breadth grows)
python -m pytest tests/test_pokepy_multigen_live_diff.py -n 4 -q --tb=no

# Regenerate Dex JSON + callback manifest after Showdown data changes
node scripts/extract_dex_json.js
```

## Rules (non-negotiable)

- **Do not refactor** `battle.py` dispatch or `battle_actions.py` pipeline.
- Translate **verbatim** from `server/pokemon-showdown/`.
- Missing custom logic → `effects.unported(table, id, event)` (loud `NotImplementedError`).
- Fix the **engine**, never the test, when parity fails (see `DOC.md` §1.7).
