"""Public solve(problem, network, engine, outer_config, model_config) entry."""
import numpy as np
from .core import ModelConfig
from .hooks import make_hook
from . import backbone
from run_fiber_system_20260924 import PhysicalAudit


def solve(problem, network, engine, outer_config, model_config):
    if not isinstance(model_config, ModelConfig):
        raise TypeError('model_config must be ModelConfig')
    if outer_config.steps != 5:
        raise ValueError('Registered release uses five outer steps')
    native = problem.native
    original = native.least_squares
    cache = {}
    hits = 0
    def cached(x, jacobian=True):
        nonlocal hits
        key = np.asarray(x,dtype=np.float64).tobytes()
        if key in cache:
            value = cache[key]
            if not jacobian or value[1] is not None:
                hits += 1
                return value if jacobian else (value[0],None)
        value = original(x,jacobian)
        cache[key] = value
        return value
    native.least_squares = cached
    audit = PhysicalAudit(native)
    packet = {}
    hook = make_hook(model_config,problem,audit,packet)
    failures = []
    def guarded(*args):
        try:
            return hook(*args)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            failures.append(dict(kind=type(exc).__name__,message=str(exc),iteration=int(args[0])+1))
            packet.clear()
            audit.phase = 'solver'
            return args[2],args[4],False
    try:
        result = backbone.solve(problem,network,engine,curve_mode='norm_mean_envelope',
                                config=outer_config,hook=guarded)
        result.update(physical_counts=audit.counts(),cache_hits=hits,
                      refinement_failures=failures,model_config=model_config.to_dict())
    finally:
        audit.close()
        native.least_squares = original
    return result,packet
