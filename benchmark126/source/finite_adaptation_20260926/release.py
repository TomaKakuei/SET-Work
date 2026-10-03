"""Recommended reduced-query entry; other registered modes remain explicit."""
from .solver import solve as solve_candidate


def solve(problem, network, outer_config, mode='linear_fast'):
    """Use the separately evaluated fast path by default, with one shared model.

    The confidence-interpolation experiment remains opt-in because the bounded
    screen found a 1.30 percent multiframe regression. No production policy is
    changed by importing or calling this module.
    """
    return solve_candidate(problem, network, outer_config, mode=mode)
