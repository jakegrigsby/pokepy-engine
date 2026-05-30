# Parity-later phase (deferred)

PRNG frame-exact parity against Showdown is **not** a gate for the playable-engine phase.

When ready:

1. Use `showdown-reference` branch (`pokepy/showdown/*`) + `scripts/sim_frame_trace.py` + `POKEPY_PRNG_TRACE`.
2. Run `tests/test_pokepy_multigen_live_diff.py` (target >= 98/156).
3. Frame-trace each divergence to the first diverging draw; fix dispatch order or bind missing effect handlers.
4. Iterate gen9 -> 4 -> 3 -> 2 -> 1.
5. Retire legacy `_preroll_*` / `_consume_*` hooks in `battle_gen9.py` as structural dispatch subsumes them.

Reference branch: `showdown-reference` @ `31a00cd`.
