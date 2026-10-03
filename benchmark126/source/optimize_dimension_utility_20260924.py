"""Exact registered chart/design/search functions with portable imports."""
from functools import lru_cache
from scipy.optimize import minimize
import numpy as np
import torch
import dimension_utility_20260924 as u
d = u.d


@lru_cache(None)
def design(k):
    E = d.poly.powers(k, 3)
    m = {1:2, 2:4, 3:6, 4:8, 5:11, 6:13, 7:16, 8:20}[k]
    rng = np.random.default_rng(92495200+k)
    best = None
    for _ in range(16):
        nodes = np.vstack([np.zeros(k), rng.uniform(-1, 1, (m-1, k))])
        norm = np.linalg.norm(nodes, axis=1)
        nodes = nodes/np.maximum(1, norm[:, None]/2)
        A = d.poly.design(nodes, E)
        sv = np.linalg.svd(A, compute_uv=False)
        quality = sv[-1]/sv[0]
        if best is None or quality > best[0]:
            best = quality, nodes, A
    quality, nodes, A = best
    assert np.linalg.matrix_rank(A) == len(E)
    return E, nodes, A, float(quality)


def chart(origin, proposal, metric, base, V, step, kind, k):
    white = metric.apply(proposal-origin, inverse=False)
    raw = metric.apply(step, inverse=False)
    radius = max(float(white.norm()), .1*float(raw.norm()), 1e-8)
    if kind == 'usable':
        Z = u.projected(base, V)
        U, S, sp = u.spectrum(Z, V.norm())
        directions = U[:, :sp['rank']]
    else:
        directions = V
    candidates = [white]+list(directions.unbind(1))+list(base['Q'].unbind(1))
    columns = []
    for v in candidates:
        q = v.clone()
        ref = float(q.norm())
        if ref < 1e-20:
            continue
        for _ in range(2):
            if columns:
                W = torch.stack(columns, 1)
                q = q-W@(W.T@q)
        if float(q.norm()) > 1e-9*ref:
            columns.append(q/q.norm())
        if len(columns) == k:
            break
    W = torch.stack(columns, 1)
    P = metric.apply(W)*radius
    return P.numpy(), dict(requested=k, actual=W.shape[1], radius=radius,
                           source=kind,
                           orthogonality_error=float((W.T@W-torch.eye(W.shape[1])).abs().max()))


def search(E, R, starts, calls):
    k = E.shape[1]
    rows = []
    def fun(z):
        calls['value'] += 1
        res = R@d.poly.basis(z, E)
        return float(res@res/2)
    def jac(z):
        calls['jacobian'] += 1
        res = R@d.poly.basis(z, E)
        return (R@d.poly.dbasis(z, E)).T@res
    for start in starts:
        z = np.clip(start, -1, 1)
        z = z/max(1, np.linalg.norm(z)/2)
        result = minimize(fun, z, jac=jac, method='SLSQP', bounds=[(-1., 1.)]*k,
                          constraints=[dict(type='ineq', fun=lambda z:4-z@z,
                                            jac=lambda z:-2*z)],
                          options=dict(maxiter=50, ftol=1e-11))
        z = np.clip(result.x, -1, 1)
        z = z/max(1, np.linalg.norm(z)/2)
        rows.append(dict(z=z, cost=fun(z), status=int(result.status),
                         iterations=int(result.nit)))
    return sorted(rows, key=lambda q:q['cost'])
