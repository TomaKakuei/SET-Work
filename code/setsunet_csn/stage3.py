"""Auditable solver repairs; frozen Stage 2 and its checkpoint stay unchanged.

Exact block Jacobi is an explicitly charged diagnostic/reference option. It
materializes Jacobians and is not a claim of scalable matrix-free whitening.
"""
from dataclasses import asdict, dataclass
import time
import torch

from .core import completion, sym
from .stage2 import (
    Stage2Config, BatchedGNView, BlockWhitener, WhitenedOperator,
    initial_counters, estimate_block_whitener, measured_base,
    analytic_proposals, _block_edges, _cg,
)


@dataclass
class RepairConfig(Stage2Config):
    whitening: str = "identity"


def make_whitener(ga, gb, a, b, ids, edges, config, counts):
    if config.whitening == "probe":
        return estimate_block_whitener(ga, gb, a, b, ids, edges, config.whitening_probes, counts)
    if config.whitening not in ("identity", "exact_block"):
        raise ValueError(config.whitening)
    ja = jb = None
    if config.whitening == "exact_block":
        ja, jb = a.full_jacobian(), b.full_jacobian()
    roots, inverses, values_all = [], [], []
    for block in range(int(ids.max()) + 1):
        selected = ids == block
        identity = torch.eye(int(selected.sum()), dtype=ga.dtype, device=ga.device)
        if ja is None:
            root = inverse = identity
            values = identity.diagonal()
        else:
            local_a, local_b = ja[:, selected], jb[:, selected]
            normal = .5 * (local_a.T @ local_a + local_b.T @ local_b) + a.damping * identity
            values, vectors = torch.linalg.eigh(sym(normal))
            values = values.clamp_min(torch.finfo(ga.dtype).eps * values.max().clamp_min(1.))
            root = (vectors * values.sqrt()) @ vectors.T
            inverse = (vectors * values.rsqrt()) @ vectors.T
        roots.append(root)
        inverses.append(inverse)
        values_all.extend(values.detach().tolist())
    return BlockWhitener(ids, inverses, roots, {
        "mode": config.whitening, "blocks": len(roots),
        "minimum_eigenvalue": min(values_all), "maximum_eigenvalue": max(values_all),
        "condition_estimate": max(values_all) / min(values_all),
        "materialized_jacobians": ja is not None,
    })


def predicted_reduction(directional, quadratic, alpha=1.):
    """Reduction of the actual scaled quadratic model, including damping."""
    return -alpha * directional - .5 * alpha * alpha * quadratic


def solve(problem, model=None, method="csn", config=None):
    config = config or RepairConfig()
    if method not in ("csn", "mean_schur", "analytic_csn", "native_lm", "block_pcg"):
        raise ValueError(method)
    counts = initial_counters()
    x = problem.initial.detach().clone()
    ids, history, trajectory = problem.block_id.to(x.device), None, []
    edges = _block_edges(problem, ids)
    damping = config.damping
    started = time.perf_counter()

    def costs(z):
        counts["objective_evaluations"] += 2
        counts["residual_evaluations"] += 2
        return torch.stack([problem.cost(z, v) for v in (0, 1)]).detach()

    initial = costs(x)
    for iteration in range(config.steps):
        a, b = [BatchedGNView(problem, x, v, damping, counts) for v in (0, 1)]
        g = .5 * (a.g + b.g)
        before = costs(x)
        diagnostics = {"rank": 0, "gain": 0., "gap": 0., "lock_error": 0., "changes": [0., 0.]}
        white_diagnostics = {"mode": "unused"}
        if method == "native_lm":
            # LM has no use for whitening probes or a learned measured base.
            ja, jb = a.full_jacobian(), b.full_jacobian()
            normal = .5 * (ja.T @ ja + jb.T @ jb) + damping * torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            step = torch.linalg.solve(normal, -g)
            counts["dense_linear_solves"] += 1
            action = normal @ step
        else:
            white = make_whitener(a.g, b.g, a, b, ids, edges, config, counts)
            white_diagnostics = white.diagnostics
            wa, wb = WhitenedOperator(a, white), WhitenedOperator(b, white)
            ga, gb = white.to_white_gradient(a.g), white.to_white_gradient(b.g)
            if method == "block_pcg":
                wstep = _cg(lambda v: .5 * (wa(v) + wb(v)), .5 * (ga + gb), config.cg_iterations)
            else:
                base = measured_base(ga, gb, wa, wb, ids, edges,
                    history=None if history is None else white.to_white_step(history), count=config.measured_directions)
                counts["small_linear_solves"] += 1
                if method == "analytic_csn":
                    proposals = analytic_proposals(base, config.completion_directions)
                else:
                    if model is None:
                        raise ValueError("CSN requires a model")
                    with torch.no_grad():
                        proposals = model.to(x)(base, ga, gb, ids, edges, directions=config.completion_directions)
                    counts["network_calls"] += 1
                wstep, _, diagnostics = completion(base, ga, gb, wa, wb, proposals,
                    consensus=method != "mean_schur", counts=counts)
            step = white.to_parameter(wstep).detach()
            action = .5 * (a(step) + b(step))
        directional, quadratic = float(g @ step), float(step @ action)
        accepted, alpha, after, rho = False, 0., before, None
        for backtrack in range(config.backtracks):
            scale = .5 ** backtrack
            candidate = x + scale * step
            counts["evaluated_states"] += 1
            observed = costs(candidate)
            if torch.isfinite(observed).all() and float(observed.mean()) <= float(before.mean()) + config.armijo * scale * directional + 1.e-14:
                actual = float(before.mean() - observed.mean())
                prediction = predicted_reduction(directional, quadratic, scale)
                rho = actual / max(prediction, 1.e-30)
                history, x = candidate - x, candidate.detach()
                accepted, alpha, after = True, scale, observed
                break
        old_damping = damping
        if accepted:
            if rho > .75 and alpha == 1.:
                damping = max(config.damping_min, damping * .3)
            elif rho < .25:
                damping = min(config.damping_max, damping * 3.)
        else:
            history = None
            damping = min(config.damping_max, damping * 10.)
        trajectory.append({"iteration": iteration, "cost_before": before.tolist(), "cost_after": after.tolist(),
            "accepted": accepted, "step_scale": alpha, "damping_before": old_damping, "damping_after": damping,
            "actual_to_predicted_ratio": rho, "directional_derivative": directional, "quadratic": quadratic,
            "predicted_reduction": predicted_reduction(directional, quadratic),
            "scaled_predicted_reduction": predicted_reduction(directional, quadratic, alpha),
            "whitening": white_diagnostics, **diagnostics})
    final = costs(x)
    return {"theta": x, "seconds": time.perf_counter() - started, "counts": counts,
        "trajectory": trajectory, "config": asdict(config), "initial_cost": initial.tolist(), "final_cost": final.tolist()}
