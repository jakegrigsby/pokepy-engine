# pokepy-engine — Agent Documentation

Practical reference for AI agents working on the pokepy battle engine and its
Showdown parity tooling. This document grows over time; today it covers the
**parity test suite** and the **Showdown ground-truth cache**.

> Audience: coding agents (and humans) doing parity/faithfulness work against
> Pokémon Showdown. Keep this file accurate when you change the harness, the
> test battery, or the cache.

---

## 1. Parity test suite

### 1.1 What it is

The engine is validated by **live-diff parity tests**: we run a battle in
pokepy and the *same* battle (same seed, teams, and action sequence) in a
vendored Pokémon Showdown, then assert the per-turn state matches.

- Test file: [`tests/test_pokepy_multigen_live_diff.py`](tests/test_pokepy_multigen_live_diff.py)
- Harness: [`scripts/parity_heuristic_e2e.py`](scripts/parity_heuristic_e2e.py)
- Fast single-scenario probe: [`scripts/parity_probe.py`](scripts/parity_probe.py)
- Ground-truth oracle: `server/pokemon-showdown/` (the vendored CLI), invoked
  as `server/pokemon-showdown/pokemon-showdown`.

Showdown is the **oracle**. We do not edit tests to make them pass; we change
the engine to match Showdown's behavior (see the faithfulness rule in §1.7).

### 1.2 How a live-diff run works

`run_live_diff(team0, team1, seed, n_turns, gen=...)` in the harness:

1. Runs pokepy via `run_pokepy_battle(...)`. Both sides use a deterministic
   heuristic (first legal move slot, else first legal action). This produces a
   `choice_log` of the exact actions taken.
2. Runs Showdown via `run_showdown(...)`, replaying the **same** `choice_log`
   against the same packed teams and seed (subprocess: `simulate-battle`).
3. Compares per-turn rows with `compare_battle_rows(...)`.

Fields compared each turn (equality on every overlapping turn):

```
p0_hp, p0_max_hp, p0_status, p1_hp, p1_max_hp, p1_status
```

Status codes in rows: `brn=1, par=2, slp=3, frz=4, psn=5, tox=6`.

A test **passes** when: Showdown `returncode == 0`, no timeout, and
`compare_battle_rows` returns `None` (all shared turns match).

### 1.3 Battery structure (≈156 collected items)

| Group | Function(s) | Turns | Notes |
|-------|-------------|-------|-------|
| Unit / parser | `test_showdown_log_parser_*`, `test_compare_battle_rows_*`, `test_gen4_calc_damage_*` | n/a | No Showdown battle |
| Gen4 gate | `test_gen4_live_diff_alakazam_psychic_turn1` | 1 | Smoke |
| Mirror full battles | gen1–4 Alakazam/Starmie/Gengar mirrors | 20 | Core damaging-turn coverage |
| Tier A (switch/faint) | `test_live_diff_multi_mon_faint_switch_turn3` (4 gens × 4 seeds), `..._switch_mirror_turn5`, `..._dual_mon_switch_mirror_full_battle` | 3 / 5 / 20 | Faint-driven switch-ins |
| Tier B (status) | 7 status scenarios (see below) | `parity_n_turns()` (default 10) | 96 cases |

Standard seeds: `999, 12345, 424242, 500` (some mirror tests use a subset).

### 1.4 Tier B status scenarios

Defined by `_STATUS_SCENARIOS` in the test file:

| Key | Mon / move | Gens |
|-----|------------|------|
| `par_twave` | Starmie, Thunder Wave | 1,2,3,4 |
| `tox_toxic` | Muk, Toxic | 2,3,4 |
| `slp_hypnosis` | Gengar, Hypnosis | 1,2,3,4 |
| `slp_spore` | Parasect, Spore | 1,2,3,4 |
| `brn_fire_blast` | Charizard, Fire Blast | 1,2,3,4 |
| `frz_blizzard` | Jynx, Blizzard | 3,4 |
| `psn_sludge` | Tentacruel, Sludge Bomb | 2,3,4 |

Teams are **mirrors** (`team0 == team1`), so move order on tied-speed turns is
decided purely by PRNG speed-tie shuffles — which makes these tests sensitive
probes for PRNG frame alignment.

### 1.5 The engine under test

The shared turn loop lives in
[`pokepy/engine/battle_gen9.py`](pokepy/engine/battle_gen9.py) and drives gens
1–4 and 9 via a `GenProfile` (`pokepy/core/gen_profile.py`). `battle_gen1.py` /
`battle_gen2.py` are thin wrappers. The PRNG is a Showdown-compatible Gen5 LCG
in [`pokepy/utils/gen5_prng.py`](pokepy/utils/gen5_prng.py); every
`random()`/`randomChance()`/shuffle iteration consumes exactly one frame, in
lockstep with Showdown's `sim/prng.ts`.

### 1.6 Running the suite

```bash
cd metamon/env/pokepy-engine

# Full live-diff file (parallel via pytest-xdist)
pytest tests/test_pokepy_multigen_live_diff.py -n 4 -q --tb=no

# Just the Tier B status battery
pytest tests/test_pokepy_multigen_live_diff.py -k "status" -n 4 -q --tb=no

# One scenario / seed while debugging
pytest "tests/test_pokepy_multigen_live_diff.py::test_live_diff_status_slp_spore[999-3]" -q

# Shorten Tier B turn count while chasing early-turn drift
POKEPY_PARITY_TURNS=3 pytest ... -k "slp_spore"
```

Fast targeted probe (no pytest), prints per-turn rows for both engines:

```bash
python scripts/parity_probe.py --gen 3 --scenario slp_spore --seed 999 --turns 10 -v
```

`POKEPY_PARITY_TURNS` (default `10`) controls Tier B / probe turn counts.

### 1.7 Faithfulness rule (important)

When a parity test fails, fix the **engine**, not the test. Every PRNG frame we
add or remove must correspond to a specific Showdown source line. Do **not** add
case-specific skips or compensating constants to make a seed pass — that creates
hacks that cancel each other and drift later. Use the PRNG trace (§3) to align
frames against `server/pokemon-showdown/sim/*.ts` ground truth.

---

## 2. Showdown ground-truth cache

### 2.1 Why it exists

Each live-diff test shells out to `node` (the Showdown CLI) **twice**:

- `simulate-battle` — run the battle (~2s/call, dominated by node startup).
- `pack-team` — convert a team to Showdown's packed format (~2s/call).

Both are **pure functions of their inputs**, and the suite re-runs identical
battles constantly (same seeds/teams/actions across runs; the same handful of
mirror teams packed thousands of times). Without caching, a full file run is
minutes; almost all of that is repeated, deterministic Showdown work.

### 2.2 What it does

Implemented in [`scripts/parity_heuristic_e2e.py`](scripts/parity_heuristic_e2e.py):

- **`simulate-battle`**: memoized to disk, keyed on a hash of the exact stdin
  script (`>start` format+seed, both packed teams, and the full action log).
- **`pack-team`**: memoized in-process **and** on disk, keyed on the team export
  string.
- **Version token**: a content hash of every compiled `.js` under
  `dist/sim` and `dist/data`. This invalidates the cache on any real Showdown
  engine change, while ignoring the mtime churn the CLI causes by recompiling
  `dist/` on every invocation.

Cache location: `.cache/showdown_parity/` (gitignored). Writes are atomic
(temp file + `os.replace`), so it is safe under `pytest -n` xdist workers.

### 2.3 Performance

| Full live-diff file (≈156 items) | Time |
|----------------------------------|------|
| Cold (empty cache) | ~210s |
| Warm (steady state) | ~1.6s |

Results are identical cold vs warm (e.g. 103 passed / 53 failed either way).

### 2.4 Why it is correct (not "cheating")

Showdown's output depends **only** on its input script and its compiled code,
never on pokepy internals. Therefore:

- If a pokepy change alters the **action sequence**, the script changes, the key
  changes, and Showdown **re-runs** — fresh ground truth, automatically.
- If a pokepy change alters HP/status but **not** the actions, Showdown's result
  is genuinely unchanged, so the cache hit is valid.
- If **Showdown itself** is edited and rebuilt (e.g. `data/mods/gen3/conditions.ts`),
  the compiled `dist` content changes, the version token changes, and the whole
  cache busts.

The cache can never silently serve a stale result: any input that affects
Showdown's output is part of the key.

### 2.5 Controls (environment variables)

| Variable | Effect |
|----------|--------|
| `POKEPY_SHOWDOWN_CACHE=0` | Disable cache read+write (always run live) |
| `POKEPY_SHOWDOWN_CACHE_DIR=<path>` | Override cache location |
| `POKEPY_SHOWDOWN_CACHE_VERSION=<str>` | Extra token to force-invalidate manually |
| `POKEPY_PRNG_TRACE=1` | Bypass cache entirely (need live stderr trace; see §3) |

To clear the cache: `rm -rf .cache/showdown_parity`.

### 2.6 Gotchas

- The Showdown CLI **recompiles `dist/` on every run**, bumping file mtimes.
  The version token hashes file **content**, not mtimes, so this does not bust
  the cache. (Compiled output is stable across rebuilds of unchanged source.)
- The first run after any Showdown rebuild is cold (token changed) — expected.
- Only successful (`returncode == 0`) `simulate-battle` runs are cached; errors
  and timeouts always re-run.

---

## 3. PRNG frame trace (debugging aid)

For root-causing frame drift, both PRNGs can dump their consumed frames,
gated by `POKEPY_PRNG_TRACE=1` (inert otherwise):

- pokepy: [`pokepy/utils/gen5_prng.py`](pokepy/utils/gen5_prng.py) appends
  `(raw_value, caller)` to `gen5_prng._PRNG_TRACE_LOG`.
- Showdown: `server/pokemon-showdown/sim/prng.ts` writes `PRNGTRACE <raw>
  <float> <stack>` to stderr (requires a `node build` after editing).

Because both use the same Gen5 LCG, the raw 32-bit values form an identical
sequence; align the two traces by raw value to see exactly which consumer
(accuracy roll, crit, damage, speed-tie shuffle, sleep duration, …) reads each
frame, and where pokepy and Showdown diverge.

> Note: the trace hooks are debugging instrumentation. The Showdown-side edit
> modifies vendored source; remove it if a pristine oracle checkout is needed.
