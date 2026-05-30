"""Engine registry.

The old packed-state engine (``battle_gen9`` / ``engine_adapter`` /
``action_mask`` / ``switch_requests``) was removed in the verbatim-port
refactor. The new object-model engine lives under ``pokepy.showdown``; this
module will re-expose the gen-keyed entry points once the port registers them.
"""

__all__: list[str] = []
