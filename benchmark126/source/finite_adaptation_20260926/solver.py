"""Observation-only quadratic adaptation, one intervention in five steps.

The old release and checkpoint stay immutable. Four predeclared ablations
separate controller consistency, early linear exit, and trusted extra candidates.
"""
from dataclasses import dataclass, asdict
from types import SimpleNamespace
import time
import numpy as np
import torch
from finite_models_20260925.core import ModelConfig
from finite_models_20260925.hooks import finish, quadratic_design, poly
from finite_models_20260925.search_completion import coordinate_polish, polished_hook
import optimize_dimension_utility_20260924 as legacy_q
from stage9_backends import engine_for, CountedEngine
from . import backbone


@dataclass(frozen=True)
class AdaptationConfig:
    linear_tolerance: float = 1e-8
    model_error_target: float = .01
    minimum_scale: float = .125
    max_trusted_candidates: int = 3
    max_joint_jacobians: int = 21

    def __post_init__(self):
        if not (0 < self.linear_tolerance < 1 and
                0 < self.model_error_target < 1 and
                0 < self.minimum_scale <= 1 and
                0 <= self.max_trusted_candidates <= 3 and
                self.max_joint_jacobians == 21):
            raise ValueError('Invalid bounded adaptation configuration')


MODES = {
    'legacy': (False, False, False),
    'controller': (True, False, False),
    'linear_fast': (False, True, False),
    'trusted': (False, False, True),
    'combined': (True, True, True),
}


def relative_defect(actual, predicted, reference):
    return float(np.linalg.norm(actual-predicted)/max(float(reference), 1e-15))


def trust_scale(error, config):
    if not np.isfinite(error) or error < 0:
        raise ValueError('Nonfinite model diagnostic')
    return float(np.clip(np.sqrt(config.model_error_target/max(error, 1e-30)),
                         config.minimum_scale, 1.))


def adaptive_hook(problem, audit, packet, fast, trusted, config):
    acceptance_config = ModelConfig('quadratic_anchored8')

    def hook(iteration, origin, proposal, before, after, metric, base, V, step):
        if iteration != 1:
            return proposal, after, False
        tic = time.perf_counter()
        audit.phase = 'intervention'
        native = problem.native
        P, chart = legacy_q.chart(origin, proposal, metric, base, V, step, 'usable', 8)
        k = P.shape[1]
        x0, xp = origin.numpy(), proposal.numpy()
        r0, J = native.least_squares(x0, True)
        JP0 = np.asarray(J@P)
        normalizer = max(float(np.linalg.norm(r0)), 1e-15)
        E1 = poly.powers(k, 1)
        C1 = np.array([r0 if not e.sum() else JP0[:, np.flatnonzero(e)[0]] for e in E1])
        R1 = np.linalg.qr(C1.T, mode='r')/normalizer
        calls = dict(value=0, jacobian=0)
        linear = legacy_q.search(E1, R1, [np.zeros(k)], calls)[0]
        zlin = linear['z']
        rlin, _ = native.least_squares(x0+P@zlin, False)
        E, nodes, A = quadratic_design(k)
        gate = dict(enabled=fast, taken=False, tolerance=config.linear_tolerance,
                    linear_defect=relative_defect(rlin, r0+JP0@zlin, normalizer),
                    certificate_scope='Finite witnesses only; not domain-wide linearity')
        witness = None
        if fast and linear['status'] == 0 and gate['linear_defect'] <= config.linear_tolerance:
            rp, _ = native.least_squares(xp, False)
            gate['proposal_defect'] = relative_defect(rp, r0+J@(xp-x0), normalizer)
            if gate['proposal_defect'] <= config.linear_tolerance:
                z = nodes[1]
                rw, Jw = native.least_squares(x0+P@z, True)
                witness = (rw, np.asarray(Jw@P))
                gate['witness_value_defect'] = relative_defect(rw, r0+JP0@z, normalizer)
                gate['witness_J_defect'] = relative_defect(witness[1], JP0, np.linalg.norm(JP0))
                gate['taken'] = max(gate['witness_value_defect'], gate['witness_J_defect']) <= config.linear_tolerance
        if gate['taken']:
            states = [x0, x0+P@zlin, x0+P@nodes[1]]
            residuals = [r0, rlin, witness[0]]
            selected, cost, changed, info = finish(native, proposal, after, states,
                residuals, ['origin', 'linear', 'linear_witness'], acceptance_config)
            packet.update(arrays=dict(origin=x0, proposal=xp, P=P,
                selected=selected, selected_cost=cost, candidate_states=np.array(states),
                candidate_residuals=np.array(residuals)),
                detail=dict(chart=chart, linear_gate=gate, fit=None,
                    trust_candidates=[], internal_status=[linear['status']],
                    surrogate_calls=calls, seconds=time.perf_counter()-tic, **info))
            audit.phase = 'solver'
            return torch.from_numpy(selected).to(proposal), torch.from_numpy(cost).to(after), changed

        rr, jj = [], []
        for i, z in enumerate(nodes):
            if i == 1 and witness is not None:
                r, JP = witness
            else:
                r, j = native.least_squares(x0+P@z, True)
                JP = np.asarray(j@P)
            rr.append(r); jj.append(JP)
        rr, jj = np.array(rr), np.array(jj)
        observations = np.concatenate([np.vstack([a, b.T]) for a, b in zip(rr, jj)])
        C, _, rank, sv = np.linalg.lstsq(A, observations, rcond=None)
        if rank != len(E) or not np.isfinite(C).all():
            raise ValueError('Quadratic design lost rank or has nonfinite coefficients')
        R = np.linalg.qr(C.T, mode='r')
        rng = np.random.default_rng(92495300+k)
        starts = np.vstack([np.zeros(k), zlin, rng.uniform(-1, 1, (14, k))])
        internal = legacy_q.search(E, R/normalizer, starts, calls)
        chosen = []
        for row in internal:
            if all(np.linalg.norm(row['z']-z) > 1e-5 for z in chosen):
                chosen.append(row['z'])
            if len(chosen) == 3:
                break
        while len(chosen) < 3:
            chosen.append(zlin)
        states = [x0, x0+P@zlin]+[x0+P@z for z in nodes]
        residuals = [r0, rlin]+list(rr)
        labels = ['origin', 'linear']+['probe']*len(nodes)
        candidates = []
        for z in chosen:
            x = x0+P@z; residual = native.least_squares(x, False)[0]
            states.append(x); residuals.append(residual); labels.append('legacy_quadratic')
            candidates.append((z, x, residual))
        polishing, newcoords = [], []
        for z in chosen:
            polished, info = coordinate_polish(z, E, R/normalizer)
            polishing.append(info)
            if np.linalg.norm(polished-z) > 1e-8 and all(np.linalg.norm(polished-v) > 1e-8 for v in newcoords):
                x = x0+P@polished; residual = native.least_squares(x, False)[0]
                states.append(x); residuals.append(residual); labels.append('polished_quadratic')
                newcoords.append(polished); candidates.append((polished, x, residual))
        # Held-out zlin value does not enter the fitted value/J observations.
        heldout = relative_defect(rlin, poly.basis(zlin, E)@C, normalizer)
        center_J_error = relative_defect(C.T@poly.dbasis(np.zeros(k), E), JP0, np.linalg.norm(JP0))
        trust_info, unique = [], []
        if trusted:
            for z, x, res in sorted(candidates, key=lambda t:float(t[2]@t[2])):
                if len(trust_info) >= config.max_trusted_candidates:
                    break
                if any(np.linalg.norm(z-old) <= 1e-5 for old in unique):
                    continue
                unique.append(z)
                error = max(heldout, relative_defect(res, poly.basis(z, E)@C, normalizer))
                scale = trust_scale(error, config)
                if scale >= 1. or np.linalg.norm(x-xp) <= 1e-14:
                    continue
                candidate = xp+scale*(x-xp)
                residual = native.least_squares(candidate, False)[0]
                states.append(candidate); residuals.append(residual); labels.append('trusted_interpolation')
                trust_info.append(dict(scale=scale, error=error, cost=.5*float(residual@residual)))
        selected, cost, changed, acceptance = finish(native, proposal, after, states,
            residuals, labels, acceptance_config)
        packet.update(arrays=dict(origin=x0, proposal=xp, P=P, r0=r0, JP0=JP0,
            E=E, nodes=nodes, A=A, C=C, R=R, zlin=zlin, rlin=rlin,
            selected=selected, selected_cost=cost, candidate_states=np.array(states),
            candidate_residuals=np.array(residuals)),
            detail=dict(chart=chart, linear_gate=gate,
                fit=dict(rank=int(rank), condition=float(sv[0]/sv[-1]),
                    relative_fit=relative_defect(A@C, observations, np.linalg.norm(observations)),
                    center_J_error=center_J_error, heldout_linear_error=heldout),
                trust_candidates=trust_info, polishing=polishing,
                internal_status=[x['status'] for x in internal], surrogate_calls=calls,
                seconds=time.perf_counter()-tic, **acceptance))
        audit.phase = 'solver'
        return torch.from_numpy(selected).to(proposal), torch.from_numpy(cost).to(after), changed
    return hook


def solve(problem, network, outer_config, mode='combined', config=None):
    if mode not in MODES or outer_config.steps != 5:
        raise ValueError('Unknown adaptation mode or non-five-step configuration')
    config = config or AdaptationConfig()
    sync, fast, trusted = MODES[mode]
    native = engine_for(problem)
    original = native.least_squares
    cache, packet, failures = {}, {}, []
    query = dict(J=0, F=0, hits=0)

    def cached(x, jacobian=True):
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key in cache and (not jacobian or cache[key][1] is not None):
            query['hits'] += 1
            return cache[key] if jacobian else (cache[key][0], None)
        # Reserve five joint derivative calls for the outer linearizations.
        if jacobian and query['J'] >= config.max_joint_jacobians-5:
            raise RuntimeError('Registered total derivative budget would be exceeded')
        query['J' if jacobian else 'F'] += 1
        value = original(x, jacobian)
        cache[key] = value
        return value

    native.least_squares = cached
    proxy = SimpleNamespace(native=native)
    audit = SimpleNamespace(phase='solver')
    hook = (adaptive_hook(proxy, audit, packet, fast, trusted, config) if fast or trusted
            else polished_hook(proxy, audit, packet))

    def guarded(*args):
        try:
            return hook(*args)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append(dict(kind=type(exc).__name__, message=str(exc), iteration=int(args[0])+1))
            packet.clear(); audit.phase = 'solver'
            return args[2], args[4], False
    try:
        result = backbone.solve(problem, network, CountedEngine(native),
            curve_mode='norm_mean_envelope', config=outer_config, hook=guarded,
            sync_intervention_damping=sync)
    finally:
        native.least_squares = original
    # All runs keep five outer linearizations, whether implemented by AD or native.
    if result['counts']['linearizations'] % 2:
        raise AssertionError('Unpaired view linearization accounting')
    total_j = query['J']+result['counts']['linearizations']//2
    if total_j > config.max_joint_jacobians:
        raise AssertionError('Physical derivative accounting exceeds registered cap')
    result.update(LS_requests=query, actual_joint_J=total_j,
        refinement_failures=failures, adaptation_mode=mode,
        adaptation_config=asdict(config))
    return result, packet
