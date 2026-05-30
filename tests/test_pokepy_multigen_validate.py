"""Validation gate documentation (T3b deferred to manual eval)."""

from pokepy.core.gen_profile import registered_gens


def test_all_target_gens_registered():
    """T1/T2 prerequisite: gens 1-4 and 9 must be in the engine registry."""
    for gen in (1, 2, 3, 4, 9):
        assert gen in registered_gens()
