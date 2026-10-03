"""Two reviewed research models; task-independent configuration dispatch."""
from .path_stable import solve as solve_path
from .search_completion import solve as solve_quadratic

MODELS = {
    'path_line_complete': dict(implementation='path_stable', evidence_method='path_line_complete'),
    'quadratic_usable8_complete': dict(implementation='search_completion', evidence_method='quadratic_polished8'),
}


def solve(problem, network, engine, outer_config, model='path_line_complete'):
    """Return (solver_result, diagnostic_packet), preserving five outer steps.

    The caller supplies the registered shared network and observation problem.
    No score, task identifier, ground truth, or per-case policy is accepted.
    The diagnostic packet is optional for users, retained in registered runs.
    """
    if model == 'path_line_complete':
        return solve_path(problem,network,engine,outer_config)
    if model == 'quadratic_usable8_complete':
        return solve_quadratic(problem,network,engine,outer_config,'quadratic_polished8')
    raise ValueError(f'Unknown release model: {model}. Choose one of {tuple(MODELS)}')
