"""Frozen native GN operators, measured Schur elimination and two-view minimax."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable
import time
import torch
from torch import Tensor

@dataclass
class CSNConfig:
    steps: int = 5
    measured_directions: int = 2
    completion_directions: int = 2
    damping: float = 1e-3
    backtracks: int = 5
    armijo: float = 1e-4
    rank_tolerance: float = 1e-9

def orth(columns: Tensor, tolerance: float = 1e-9) -> Tensor:
    """Twice-reorthogonalized MGS; drop rank-deficient proposals explicitly."""
    result = []
    reference = torch.linalg.vector_norm(columns).detach().clamp_min(1e-30)
    for v in columns.unbind(1):
        for _ in range(2):
            for q in result:
                v = v - q * (q @ v)
        size = torch.linalg.vector_norm(v)
        if float(size.detach()) > tolerance * float(reference):
            result.append(v / size)
    return torch.stack(result, 1) if result else columns[:, :0]

def sym(matrix):
    return 0.5 * (matrix + matrix.T)

def minimax(Ha, Hb, ba, bb, counters=None, iterations=36):
    """Envelope gradient: solve the scalar dual without differentiating its argmin."""
    dH, db = Ha - Hb, ba - bb
    def evaluate(t, detach=False):
        H, b = Hb + t*dH, bb + t*db
        if detach:
            H, b = H.detach(), b.detach()
        a = torch.linalg.solve(H, b)
        if counters is not None:
            counters["small_linear_solves"] += 1
        return a, 2*(db.detach() @ a) - a @ dH.detach() @ a
    with torch.no_grad():
        if float(evaluate(0., True)[1]) >= 0:
            weight = 0.
        elif float(evaluate(1., True)[1]) <= 0:
            weight = 1.
        else:
            lo, hi = 0., 1.
            for _ in range(iterations):
                mid = (lo+hi)*0.5
                if float(evaluate(mid, True)[1]) > 0:
                    hi = mid
                else:
                    lo = mid
            weight = (lo+hi)*0.5
    a, _ = evaluate(weight)
    gain = 0.5*((bb+weight*db) @ a)
    changes = torch.stack((-ba@a+0.5*a@Ha@a, -bb@a+0.5*a@Hb@a))
    return a, weight, gain, changes.max()+gain, changes

class GNView:
    """One immutable linearization. Every Bv is one actual JVP and one VJP."""
    def __init__(self, problem, x, view, damping, counts):
        self.x = x.detach().clone()
        self.counts = counts
        self.damping = damping
        counts["weight_evaluations"] += 1
        if problem.metadata and problem.metadata.get("robust"):
            counts["residual_evaluations"] += 1
        weights = problem.weights(self.x, view).detach().sqrt()
        def residual(z):
            counts["residual_evaluations"] += 1
            return (problem.raw(z, view)*weights).reshape(-1)
        self.residual = residual
        self.r, self.pullback = torch.func.vjp(residual, self.x)
        self.g = self.pullback(self.r)[0].detach()
        counts["vjp_directions"] += 1
        counts["linearizations"] += 1

    def __call__(self, v):
        jv = torch.func.jvp(self.residual, (self.x,), (v,))[1]
        self.counts["jvp_directions"] += 1
        self.counts["vjp_directions"] += 1
        self.counts["curvature_products"] += 1
        return self.pullback(jv)[0] + self.damping*v

    def columns(self, V):
        if V.shape[1] == 0:
            return V.clone()
        return torch.stack([self(v) for v in V.unbind(1)], 1)

    def full_jacobian(self):
        basis = torch.eye(self.x.numel(), dtype=self.x.dtype, device=self.x.device)
        # Batched forward AD, with every direction counted (not one free batch).
        J = torch.vmap(lambda v: torch.func.jvp(self.residual, (self.x,), (v,))[1])(basis).T
        self.counts["jvp_directions"] += self.x.numel()
        self.counts["jacobian_evaluations"] += 1
        return J

def measured_base(ga, gb, opa, opb, history=None, k=2):
    g = (ga+gb)*0.5
    candidates = [g]
    if history is not None and float(history.norm()) > 1e-15:
        candidates.append(history)
    candidates.append(ga-gb)
    Q = orth(torch.stack(candidates, 1))[:, :k]
    Ya, Yb = opa.columns(Q), opb.columns(Q)
    Y = (Ya+Yb)*0.5
    if Q.shape[1]:
        A = sym(Q.T@Y)
        c = torch.linalg.solve(A, Q.T@g)
        p0 = -Q@c
    else:
        A, c, p0 = Q.T@Q, g[:0], torch.zeros_like(g)
    e = g-Y@c
    return dict(g=g, Q=Q, Ya=Ya, Yb=Yb, Y=Y, A=A, c=c, p0=p0, e=e)

def completion(base, ga, gb, opa, opb, V, consensus=True, counts=None):
    Q, A, Y = base["Q"], base["A"], base["Y"]
    if Q.shape[1] and counts is not None:
        counts["small_linear_solves"] += 1
    Z = V-Q@torch.linalg.solve(A, Y.T@V) if Q.shape[1] else V
    Z = orth(Z)
    p0 = base["p0"]
    if Z.shape[1] == 0:
        zero = ga.sum()*0.
        return p0, zero, dict(rank=0, weight=.5, gap=0., gain=0., changes=[0.,0.], lock_error=0.)
    BZa, BZb = opa.columns(Z), opb.columns(Z)
    Ha, Hb = sym(Z.T@BZa), sym(Z.T@BZb)
    ba = Z.T@(ga-base["Ya"]@base["c"])
    bb = Z.T@(gb-base["Yb"]@base["c"])
    if consensus:
        a, weight, gain, gap, changes = minimax(Ha,Hb,ba,bb,counts)
    else:
        a = torch.linalg.solve((Ha+Hb)*.5, (ba+bb)*.5)
        if counts is not None:
            counts["small_linear_solves"] += 1
        weight = .5
        gain = .5*((ba+bb)*.5@a)
        changes = torch.stack((-ba@a+.5*a@Ha@a, -bb@a+.5*a@Hb@a))
        gap = changes.max()+gain
    p = p0-Z@a
    remaining = base["g"]-Y@base["c"]-(BZa+BZb)*.5@a
    lock = torch.linalg.vector_norm(Q.T@remaining) / base["g"].norm().clamp_min(1e-30)
    diagnostics = dict(rank=Z.shape[1], weight=weight, gain=float(gain.detach()),
                       gap=float(gap.detach()), changes=changes.detach().tolist(),
                       lock_error=float(lock.detach()))
    return p, gain, diagnostics

def analytic_proposals(base, ga, gb, s=2):
    """Residual and measured curvature response; no learned parameters, same probes."""
    e = base["e"]
    Y = base["Y"]
    response = Y[:,0] if Y.shape[1] else ga-gb
    return torch.stack((e, response),1)[:,:s]

def initial_counters():
    return dict(residual_evaluations=0, objective_evaluations=0, weight_evaluations=0, jvp_directions=0,
                vjp_directions=0, curvature_products=0, network_calls=0,
                small_linear_solves=0, dense_linear_solves=0,
                jacobian_evaluations=0, linearizations=0, evaluated_states=1)

def solve(problem, model=None, method="csn", config=None):
    config = config or CSNConfig()
    counts = initial_counters()
    x = problem.initial.detach().clone()
    history = None
    trajectory = []
    damping = config.damping
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    started = time.perf_counter()
    def costs(z):
        counts["objective_evaluations"] += 2
        counts["residual_evaluations"] += 2
        return torch.stack([problem.cost(z, view) for view in (0,1)]).detach()
    initial_cost = costs(x)
    for iteration in range(config.steps):
        opa = GNView(problem,x,0,damping,counts)
        opb = GNView(problem,x,1,damping,counts)
        ga, gb = opa.g, opb.g
        g = (ga+gb)*.5
        before = costs(x)
        diag = dict(rank=0, gain=0., gap=0., changes=[0.,0.], lock_error=0., weight=.5)
        if method in ("csn","analytic_csn","mean_schur","base"):
            base = measured_base(ga,gb,opa,opb,history,config.measured_directions)
            counts["small_linear_solves"] += 1
            if method == "base":
                p = base["p0"]
            else:
                if model is not None and method in ("csn","mean_schur"):
                    with torch.no_grad():
                        proposals = model(base,ga,gb,problem.block_id)
                    counts["network_calls"] += 1
                else:
                    proposals = analytic_proposals(base,ga,gb,config.completion_directions)
                p, _, diag = completion(base,ga,gb,opa,opb,proposals,
                                        consensus=method!="mean_schur",counts=counts)
        elif method == "native_lm":
            Ja, Jb = opa.full_jacobian(), opb.full_jacobian()
            B = (Ja.T@Ja+Jb.T@Jb)*.5 + damping*torch.eye(x.numel(),dtype=x.dtype)
            p = torch.linalg.solve(B,-g)
            counts["dense_linear_solves"] += 1
        elif method == "pcg4":
            # Unpreconditioned CG is a Krylov control with exactly four B products/view.
            p, r = torch.zeros_like(x), -g
            d = r.clone()
            for _ in range(config.measured_directions+config.completion_directions):
                Bd = (opa(d)+opb(d))*.5
                rr = r@r
                alpha = rr/(d@Bd).clamp_min(1e-30)
                p = p+alpha*d
                r_new = r-alpha*Bd
                d = r_new+(r_new@r_new)/rr.clamp_min(1e-30)*d
                r = r_new
        elif method in ("adam","gd"):
            if method == "adam":
                m = .9*m+.1*g
                v = .999*v+.001*g.square()
                p = -.03*(m/(1-.9**(iteration+1)))/((v/(1-.999**(iteration+1))).sqrt()+1e-8)
            else:
                p = -.1*g/g.norm().clamp_min(1e-30)
        else:
            raise ValueError(method)
        p = p.detach()
        directional = float(g@p)
        accepted, step_scale, after = False, 0., before
        for bt in range(config.backtracks):
            alpha = .5**bt
            candidate = x+alpha*p
            counts["evaluated_states"] += 1
            candidate_cost = costs(candidate)
            bound = before.mean()+config.armijo*alpha*directional
            if torch.isfinite(candidate_cost).all() and float(candidate_cost.mean()) <= float(bound)+1e-14:
                history = candidate-x
                x = candidate.detach()
                accepted, step_scale, after = True, alpha, candidate_cost
                break
        if not accepted:
            history = None
            damping *= 10.
        trajectory.append(dict(iteration=iteration, cost_before=before.tolist(),cost_after=after.tolist(),
                               accepted=accepted, step_scale=step_scale,damping=damping,
                               directional_derivative=directional,**diag))
    final_cost = costs(x).tolist()
    seconds = time.perf_counter()-started
    return dict(theta=x, seconds=seconds, counts=counts, trajectory=trajectory,
                config=asdict(config), initial_cost=initial_cost.tolist(),
                final_cost=final_cost)

