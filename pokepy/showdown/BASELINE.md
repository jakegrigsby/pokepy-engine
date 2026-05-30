# Pre-deletion parity baseline (Phase A, step A0)

Captured before deleting the old hand-vectorized engine, per the Verbatim
Showdown Port plan. This is the **match-or-beat target for Phase B**.

## Target (the number Phase B must reach or exceed)

The old engine's demonstrated peak, recorded in `DOC.md` sec 4.2 after the
SetStatus speedSort fix (May 2026):

| Battery            | Pass / total |
|--------------------|--------------|
| Full live-diff     | **98 / 156** |
| Tier A + mirror    | **55 / 56**  |
| Tier B status      | **39 / 96**  |

Phase B (the new `pokepy/showdown/` engine) is "done enough to cut over" when
it reaches or exceeds **98 / 156** on the full live-diff suite across the
target gens (1/2/3/4/9), measured with the same harness.

## Live re-measurement on this branch (`refactor`), at deletion time

Command (warm Showdown cache, on by default):

```bash
cd metamon/env/pokepy-engine
python -m pytest tests/test_pokepy_multigen_live_diff.py -n 4 -q --tb=no
```

Result captured: **18 passed / 138 failed** (156 total).

### Why the live number is far below the documented peak

The `refactor` branch carries in-progress frame-counting edits (to
`pokepy/sim/moves.py`, `battle.py`, `queue.py`, `helpers.py`) that chased
gen1 PRNG alignment at the expense of broad cross-gen parity. The divergences
are real engine state mismatches (e.g. `psn_sludge_bomb[999-2]`:
`pokepy=247 showdown=248`, `returncode==0`), confirmed with the Showdown cache
disabled - not a stale-cache artifact.

This regression *is the motivation for the verbatim port*: piecemeal
frame-counting was net-negative. Phase B targets the documented **98 / 156**
peak, not this mid-investigation low point.

## Acceptance commands (for Phase B)

```bash
cd metamon/env/pokepy-engine

# Full live-diff suite
python -m pytest tests/test_pokepy_multigen_live_diff.py -n 4 -q --tb=no

# Tier B status battery only
python -m pytest tests/test_pokepy_multigen_live_diff.py -k status -n 4 -q --tb=no

# Parity regressions
python -m pytest tests/test_pokepy_parity_regressions.py -q
```
