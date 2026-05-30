"""pokepy: in-process Python Pokemon battle simulator.

The engine is a near-verbatim port of Pokemon Showdown's simulator (object
model + runtime event/effect dispatch) living under ``pokepy.showdown``. The
old hand-vectorized packed-state engine was removed in the verbatim-port
refactor; see ``pokepy/showdown/`` and the plan in
``.cursor/plans/verbatim_showdown_port_*.plan.md``.
"""

from pokepy.utils.gen5_prng import Gen5PRNG

__all__ = ["Gen5PRNG"]
