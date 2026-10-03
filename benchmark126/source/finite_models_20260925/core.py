"""Numerical kernels independent of task names, scores and trained weights."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ModelConfig:
    method: str
    intervention_step: int = 2
    directions: int = 8
    acceptance_relative: float = 1e-12
    path_witness_tolerance: float = 1e-6
    path_starts: int = 6
    path_iterations: int = 70
    quadratic_starts: int = 16
    quadratic_iterations: int = 50

    def __post_init__(self):
        if self.method not in ('path_compact', 'path_verified', 'quadratic_anchored8', 'quadratic_verified8'):
            raise ValueError(self.method)
        if self.intervention_step != 2 or self.directions != 8:
            raise ValueError('This registered release fixes step2 and maximum8 directions')

    def to_dict(self):
        return asdict(self)


def project_box_ball(x, radius=2.0):
    x = np.asarray(x, dtype=float)
    y = np.clip(x, -1., 1.)
    if np.linalg.norm(y) <= radius:
        return y
    lo, hi = 0., max(np.linalg.norm(x)/radius, 1.)
    for _ in range(64):
        mid = .5*(lo+hi)
        if np.linalg.norm(np.clip(x/(1+mid), -1, 1)) > radius:
            lo = mid
        else:
            hi = mid
    return np.clip(x/(1+hi), -1, 1)


def quadratic_ball(H, g, radius=1.):
    """Convex full-space ball QP by a scalar secular equation; no SLSQP."""
    H = .5*(H+H.T)
    e, U = np.linalg.eigh(H)
    scale = max(float(np.max(abs(e))), 1e-30)
    if e.min() < -1e-10*scale:
        raise ValueError('quadratic_ball requires positive semidefinite curvature')
    e = np.maximum(e, np.finfo(float).eps*scale)
    a = U.T@g
    def point(lam):
        return -a/(e+lam)
    lam = 0.
    if np.linalg.norm(point(0.)) > radius:
        lo, hi = 0., max(np.linalg.norm(a)/radius, scale)
        for _ in range(80):
            mid = .5*(lo+hi)
            if np.linalg.norm(point(mid)) > radius:
                lo = mid
            else:
                hi = mid
        lam = hi
    x = U@point(lam)
    return x, dict(multiplier=float(lam), norm=float(np.linalg.norm(x)),
                   kkt_relative=float(np.linalg.norm(H@x+g+lam*x)/max(np.linalg.norm(g), 1e-30)))


def polynomial_segment(coefficients, lower=0., upper=1.):
    """Enumerate real stationary roots and endpoints, with finite/flat guards."""
    c = np.asarray(coefficients, dtype=float)
    if not np.isfinite(c).all() or lower > upper:
        raise ValueError('invalid finite polynomial segment')
    derivative = np.arange(1, len(c))*c[1:]
    scale = float(np.max(abs(derivative))) if len(derivative) else 0.
    candidates = [float(lower), float(upper)]
    if scale > np.finfo(float).tiny:
        d = derivative/scale
        while len(d) > 1 and abs(d[-1]) < 32*np.finfo(float).eps:
            d = d[:-1]
        roots = np.polynomial.polynomial.polyroots(d)
        for x in roots:
            if abs(x.imag) <= 1e-8*(1+abs(x.real)) and lower-1e-10 <= x.real <= upper+1e-10:
                candidates.append(float(np.clip(x.real, lower, upper)))
    candidates = np.unique(candidates)
    values = np.polynomial.polynomial.polyval(candidates, c)
    index = int(np.argmin(values))
    return float(candidates[index]), dict(candidates=candidates.tolist(), values=values.tolist(),
                                        degree=len(c)-1, certificate_scope='Fitted polynomial on this segment; floating-point root enumeration')


def squared_polynomial(vector_coefficients):
    a = np.asarray(vector_coefficients)
    c = np.zeros(2*len(a)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(a):
            c[i+j] += .5*float(x@y)
    return c


def hermite_coefficients(r0, r1, v0, v1):
    """Residual cubic satisfying physical endpoint values and derivatives."""
    return np.array([r0, v0, 3*(r1-r0)-2*v0-v1, 2*(r0-r1)+v0+v1])


class CompactAnchor:
    """Evaluate a full-input, rank<=2 nonlinear-output model without dense B2."""
    def __init__(self, r, J, C, D, pairs, mu):
        self.n = J.shape[1]
        self.H = J.T@J + mu*np.eye(self.n)
        self.g = J.T@r
        self.constant = .5*float(r@r)
        self.cr = C.T@r
        self.cJ = C.T@J
        self.gram = C.T@C
        self.forms = np.zeros((len(D), self.n, self.n))
        for k, (i, j) in enumerate(pairs):
            self.forms[:, i, j] = D[:, k]
            self.forms[:, j, i] = D[:, k]
        self.calls = 0

    def terms(self, x):
        dx = np.einsum('qij,j->qi', self.forms, x, optimize=True)
        q = .5*dx@x
        return q, dx

    def fun(self, x):
        self.calls += 1
        q, _ = self.terms(x)
        value = self.constant + self.g@x + .5*x@self.H@x
        return float(value+(self.cr+self.cJ@x)@q+.5*q@self.gram@q)

    def jac(self, x):
        q, dq = self.terms(x)
        return self.g+self.H@x+self.cJ.T@q+dq.T@(self.cr+self.cJ@x+self.gram@q)

    def solve(self, gn, iterations=70):
        if not np.any(self.forms):
            x, diag = quadratic_ball(self.H, self.g)
            return x, dict(status='convex_linear_ball', numerical=diag, starts=0, iterations=0, calls=self.calls)
        rng = np.random.default_rng(92499503+self.n)
        random = rng.normal(size=(3, self.n))
        random *= .5/np.linalg.norm(random, axis=1)[:, None]
        starts = np.vstack([np.zeros(self.n), gn, -gn, random])
        candidates = [np.zeros(self.n), gn]
        statuses, counts = [], []
        for x in starts:
            opt = minimize(self.fun, x, jac=self.jac, method='SLSQP',
                           constraints=[dict(type='ineq', fun=lambda z:1-z@z, jac=lambda z:-2*z)],
                           options=dict(maxiter=iterations, ftol=1e-11))
            statuses.append(int(opt.status)); counts.append(int(opt.nit))
            if np.isfinite(opt.x).all():
                candidates.append(opt.x/max(1., np.linalg.norm(opt.x)))
        best = min(candidates, key=self.fun)
        grad = self.jac(best)
        trial = best-grad
        projected = trial/max(1., np.linalg.norm(trial))
        return best, dict(status='bounded_multistart', statuses=statuses, iterations=counts,
                           projected_gradient_norm=float(np.linalg.norm(best-projected)), calls=self.calls,
                           global_certificate=False)


def anchored_fit(E, A, observations, r0, JP0):
    """Keep exactly observed center value/J; fit only quadratic coefficients."""
    linear = E.sum(1) <= 1
    C = np.zeros((len(E), len(r0)))
    for i, e in enumerate(E):
        if e.sum() == 0:
            C[i] = r0
        elif e.sum() == 1:
            C[i] = JP0[:, int(np.flatnonzero(e)[0])]
    X = A[:, ~linear]
    remaining = observations-A@C
    q, _, rank, sv = np.linalg.lstsq(X, remaining, rcond=1e-12)
    C[~linear] = q
    if rank != X.shape[1] or not np.isfinite(C).all():
        raise ValueError('Quadratic interpolation is rank deficient or nonfinite')
    diag = dict(rank=int(rank), unknowns=X.shape[1], condition=float(sv[0]/sv[-1]),
                relative_fit=float(np.linalg.norm(A@C-observations)/max(np.linalg.norm(observations),1e-30)),
                center_value_error=0., center_J_error=0.)
    return C, diag


def polynomial_search(E, R, starts, basis, dbasis, iterations=50, polish=False):
    calls = dict(value=0, jacobian=0)
    def fun(z):
        calls['value'] += 1
        r = R@basis(z, E)
        return .5*float(r@r)
    def jac(z):
        calls['jacobian'] += 1
        r = R@basis(z, E)
        return (R@dbasis(z, E)).T@r
    answer = []
    for start in starts:
        start = project_box_ball(start)
        opt = minimize(fun, start, jac=jac, method='SLSQP', bounds=[(-1.,1.)]*len(start),
                       constraints=[dict(type='ineq', fun=lambda z:4-z@z, jac=lambda z:-2*z)],
                       options=dict(maxiter=iterations, ftol=1e-11))
        z = project_box_ball(opt.x) if np.isfinite(opt.x).all() else start
        if fun(start) < fun(z):
            z = start
        if polish:
            # Two deterministic feasible chords; each quartic solved by roots.
            for _ in range(2):
                end = project_box_ball(z-jac(z))
                if np.linalg.norm(end-z) < 1e-12:
                    break
                r0, rm, r1 = [R@basis(z+t*(end-z), E) for t in [0.,.5,1.]]
                a2 = 2*(r1-2*rm+r0); a1 = r1-r0-a2
                t, _ = polynomial_segment(squared_polynomial([r0,a1,a2]))
                candidate = z+t*(end-z)
                if fun(candidate) < fun(z):
                    z = candidate
        grad = jac(z)
        answer.append(dict(z=z, cost=fun(z), status=int(opt.status), iterations=int(opt.nit),
                           projected_gradient_norm=float(np.linalg.norm(z-project_box_ball(z-grad)))))
    return sorted(answer, key=lambda x:x['cost']), calls
