"""pokepy.showdown - near-verbatim Python port of Pokemon Showdown's simulator.

Object model (Battle / BattleActions / BattleQueue / Pokemon / Side / Field) +
runtime event/effect dispatch, driven by per-gen data extracted from Showdown's
own Dex (see ``pokepy/data/showdown/``) and the matched Gen5 PRNG.

This package replaces the old hand-vectorized packed-state engine. See the plan
``.cursor/plans/verbatim_showdown_port_*.plan.md`` and ``TRANSLATION_GUIDE.md``.
"""

from pokepy.showdown.dex import Dex, get_dex, to_id

__all__ = ["Dex", "get_dex", "to_id"]
