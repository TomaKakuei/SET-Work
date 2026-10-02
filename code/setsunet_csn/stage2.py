"""Second-stage SETSUNET consensus Schur--Newton solver.

This module completes the planned engineering changes without modifying the
frozen pilot implementation: batched curvature actions, measured block
whitening, graph-aware probe generation, adaptive damping, and stronger
classical controls.  The learned network still makes exactly one proposal per
outer round and never selects a step length.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import torch
from torch import Tensor

from .core import completion, minimax, orth, sym


@dataclass
class Stage2Config:
    steps: int = 5
    measured_directions: int = 4
    completion_directions: int = 4
    whitening_probes: int = 2
    damping: float = 1.0e-3
    damping_min: float = 1.0e-7
    damping_max: float = 1.0e5
    backtracks: int = 6
    armijo: float = 1.0e-4
    rank_tolerance: float = 1.0e-9
    nystrom_rank: int = 8
    cg_iterations: int = 8


def initial_counters() -> dict[str, int]:
    return {
        "residual_evaluations": 0,
        "objective_evaluations": 0,
        "weight_evaluations": 0,
        "jvp_directions": 0,
        "vjp_directions": 0,
        "curvature_products": 0,
        "network_calls": 0,
        "small_linear_solves": 0,
        "dense_linear_solves": 0,
        "jacobian_evaluations": 0,
        "linearizations": 0,
        "evaluated_states": 1,
        "whitening_products": 0,
    }


class BatchedGNView:
    """An immutable robust GN linearization with batched matrix-free actions."""

    def __init__(self, problem, x: Tensor, view: int, damping: float, counts: dict) -> None:
        self.x = x.detach().clone()
        self.damping = float(damping)
        self.counts = counts
        counts["weight_evaluations"] += 1
        if problem.metadata and problem.metadata.get("robust"):
            counts["residual_evaluations"] += 1
        weights = problem.weights(self.x, view).detach().sqrt()

        def residual(z: Tensor) -> Tensor:
            return (problem.raw(z, view) * weights).reshape(-1)

        self.residual = residual
        counts["residual_evaluations"] += 1
        self._reverse_jacobian = None
        try:
            self.r, self.pullback = torch.func.vjp(residual, self.x)
        except RuntimeError as exc:
            if "functorch transforms" not in str(exc):
                raise
            working = self.x.detach().requires_grad_(True)
            self.r = residual(working).detach()
            self._reverse_jacobian = torch.autograd.functional.jacobian(
                residual, working, vectorize=False,
            ).detach()
            self.pullback = lambda vector: (self._reverse_jacobian.T @ vector,)
            counts["jacobian_evaluations"] += 1
        self.g = self.pullback(self.r)[0].detach()
        counts["vjp_directions"] += 1
        counts["linearizations"] += 1

    def columns(self, vectors: Tensor) -> Tensor:
        if vectors.ndim != 2:
            raise ValueError("curvature directions must have shape [parameters, directions]")
        if vectors.shape[1] == 0:
            return vectors.clone()

        if self._reverse_jacobian is not None:
            jacobian = self._reverse_jacobian
            result = jacobian.T @ (jacobian @ vectors) + self.damping * vectors
            count = int(vectors.shape[1])
            self.counts["jvp_directions"] += count
            self.counts["vjp_directions"] += count
            self.counts["curvature_products"] += count
            return result

        def action(vector: Tensor) -> Tensor:
            jv = torch.func.jvp(self.residual, (self.x,), (vector,))[1]
            return self.pullback(jv)[0] + self.damping * vector

        try:
            result = torch.vmap(action)(vectors.T).T
        except NotImplementedError:
            # A few imaging kernels (notably grid_sample) still lack forward
            # AD.  Reverse AD remains native and yields the same batched GN
            # product once per frozen linearization.
            if self._reverse_jacobian is None:
                self._reverse_jacobian = torch.func.jacrev(self.residual)(self.x).detach()
                self.counts["jacobian_evaluations"] += 1
            jacobian = self._reverse_jacobian
            result = jacobian.T @ (jacobian @ vectors) + self.damping * vectors
        count = int(vectors.shape[1])
        self.counts["residual_evaluations"] += 1
        self.counts["jvp_directions"] += count
        self.counts["vjp_directions"] += count
        self.counts["curvature_products"] += count
        return result

    def __call__(self, vector: Tensor) -> Tensor:
        return self.columns(vector[:, None])[:, 0]

    def full_jacobian(self) -> Tensor:
        if self._reverse_jacobian is not None:
            return self._reverse_jacobian
        basis = torch.eye(self.x.numel(), dtype=self.x.dtype, device=self.x.device)
        try:
            jacobian = torch.vmap(
                lambda vector: torch.func.jvp(self.residual, (self.x,), (vector,))[1]
            )(basis).T
        except NotImplementedError:
            jacobian = torch.func.jacrev(self.residual)(self.x).detach()
            self._reverse_jacobian = jacobian
        self.counts["residual_evaluations"] += 1
        self.counts["jvp_directions"] += self.x.numel()
        self.counts["jacobian_evaluations"] += 1
        return jacobian


def _block_edges(problem, block_id: Tensor) -> Tensor:
    metadata = problem.metadata or {}
    value = metadata.get("block_edges")
    if value is None:
        blocks = int(block_id.max()) + 1
        if blocks <= 1:
            return torch.empty(2, 0, dtype=torch.long, device=block_id.device)
        value = torch.stack((torch.arange(blocks - 1), torch.arange(1, blocks)))
    value = torch.as_tensor(value, dtype=torch.long, device=block_id.device)
    if value.numel() == 0:
        return value.reshape(2, 0)
    if value.ndim != 2 or value.shape[0] != 2:
        raise ValueError("block_edges must have shape [2, edges]")
    valid = (value >= 0).all(dim=0) & (value < int(block_id.max()) + 1).all(dim=0)
    value = value[:, valid]
    keep = value[0] != value[1]
    return value[:, keep]


def _segment_sum(values: Tensor, ids: Tensor, blocks: int) -> Tensor:
    result = values.new_zeros((blocks,) + values.shape[1:])
    return result.index_add(0, ids, values)


def structured_directions(
    ga: Tensor,
    gb: Tensor,
    block_id: Tensor,
    block_edges: Tensor,
    count: int,
    history: Tensor | None = None,
) -> Tensor:
    """Generate up to eight coordinate-equivariant, topology-aware probes."""
    ids = block_id.to(device=ga.device, dtype=torch.long)
    blocks = int(ids.max()) + 1
    pooled = 0.5 * (ga + gb)
    disagreement = ga - gb
    candidates = [pooled]
    if history is not None and float(history.norm().detach()) > 1.0e-15:
        candidates.append(history)
    candidates.append(disagreement)
    energy = _segment_sum(pooled.square()[:, None], ids, blocks)[:, 0]
    neighbor = energy.new_zeros(blocks)
    degree = energy.new_zeros(blocks)
    if block_edges.numel():
        source, target = block_edges
        both_source = torch.cat((source, target))
        both_target = torch.cat((target, source))
        neighbor.index_add_(0, both_target, energy[both_source])
        degree.index_add_(0, both_target, torch.ones_like(both_target, dtype=energy.dtype))
    graph_weight = neighbor / degree.clamp_min(1.0)
    graph_weight = graph_weight / graph_weight.mean().clamp_min(1.0e-15)
    candidates.append(pooled * graph_weight[ids])
    mode = 1
    while len(candidates) < max(count, 1):
        phase = torch.sin((torch.arange(blocks, device=ga.device, dtype=ga.dtype) + 1) * (mode * 1.61803398875))
        mask = torch.where(phase >= 0, torch.ones_like(phase), -torch.ones_like(phase))
        candidates.append((pooled if mode % 2 else disagreement) * mask[ids])
        mode += 1
    return orth(torch.stack(candidates, dim=1))[:, :count]


class BlockWhitener:
    """Probe-estimated SPD block metric and its square-root coordinate maps."""

    def __init__(self, block_id: Tensor, inverse_sqrt: list[Tensor], sqrt: list[Tensor], diagnostics: dict) -> None:
        self.block_id = block_id
        self.inverse_sqrt = inverse_sqrt
        self.sqrt = sqrt
        self.diagnostics = diagnostics

    def _apply(self, value: Tensor, matrices: list[Tensor]) -> Tensor:
        was_vector = value.ndim == 1
        matrix = value[:, None] if was_vector else value
        output = torch.zeros_like(matrix)
        for block, transform in enumerate(matrices):
            selected = self.block_id == block
            output[selected] = transform @ matrix[selected]
        return output[:, 0] if was_vector else output

    def to_parameter(self, white_value: Tensor) -> Tensor:
        return self._apply(white_value, self.inverse_sqrt)

    def to_white_gradient(self, gradient: Tensor) -> Tensor:
        return self._apply(gradient, self.inverse_sqrt)

    def to_white_step(self, parameter_step: Tensor) -> Tensor:
        return self._apply(parameter_step, self.sqrt)


def estimate_block_whitener(
    ga: Tensor,
    gb: Tensor,
    opa: BatchedGNView,
    opb: BatchedGNView,
    block_id: Tensor,
    block_edges: Tensor,
    probes: int,
    counts: dict,
) -> BlockWhitener:
    ids = block_id.to(device=ga.device, dtype=torch.long)
    directions = structured_directions(ga, gb, ids, block_edges, max(1, probes))
    responses = 0.5 * (opa.columns(directions) + opb.columns(directions))
    counts["whitening_products"] += 2 * int(directions.shape[1])
    global_scale = (directions * responses).sum() / directions.square().sum().clamp_min(1.0e-30)
    global_scale = global_scale.detach().abs().clamp_min(1.0e-8)
    inverse_sqrt: list[Tensor] = []
    roots: list[Tensor] = []
    eigenvalues = []
    for block in range(int(ids.max()) + 1):
        selected = ids == block
        q, y = directions[selected], responses[selected]
        width = int(q.shape[0])
        identity = torch.eye(width, dtype=q.dtype, device=q.device)
        ridge = 1.0e-3 * torch.trace(q @ q.T).clamp_min(1.0e-12) / max(width, 1)
        estimate = y @ q.T @ torch.linalg.inv(q @ q.T + ridge * identity)
        estimate = 0.5 * (estimate + estimate.T)
        estimate = 0.85 * estimate + 0.15 * global_scale * identity
        values, vectors = torch.linalg.eigh(estimate)
        floor = global_scale * 1.0e-3
        ceiling = global_scale * 1.0e3
        values = values.clamp(floor, ceiling)
        roots.append((vectors * values.sqrt()) @ vectors.T)
        inverse_sqrt.append((vectors * values.rsqrt()) @ vectors.T)
        eigenvalues.extend(values.detach().cpu().tolist())
    diagnostics = {
        "blocks": int(ids.max()) + 1,
        "probes": int(directions.shape[1]),
        "minimum_eigenvalue": float(min(eigenvalues)),
        "maximum_eigenvalue": float(max(eigenvalues)),
        "condition_estimate": float(max(eigenvalues) / max(min(eigenvalues), 1.0e-30)),
    }
    return BlockWhitener(ids, inverse_sqrt, roots, diagnostics)


class WhitenedOperator:
    def __init__(self, operator: BatchedGNView, whitener: BlockWhitener) -> None:
        self.operator = operator
        self.whitener = whitener

    def columns(self, vectors: Tensor) -> Tensor:
        parameter_vectors = self.whitener.to_parameter(vectors)
        return self.whitener.to_parameter(self.operator.columns(parameter_vectors))

    def __call__(self, vector: Tensor) -> Tensor:
        return self.columns(vector[:, None])[:, 0]


def measured_base(
    ga: Tensor,
    gb: Tensor,
    opa,
    opb,
    block_id: Tensor,
    block_edges: Tensor,
    *,
    history: Tensor | None,
    count: int,
) -> dict[str, Tensor]:
    pooled = 0.5 * (ga + gb)
    Q = structured_directions(ga, gb, block_id, block_edges, count, history)
    Ya, Yb = opa.columns(Q), opb.columns(Q)
    Y = 0.5 * (Ya + Yb)
    if Q.shape[1]:
        A = sym(Q.T @ Y)
        regularizer = torch.finfo(A.dtype).eps * torch.trace(A).abs().clamp_min(1.0)
        coefficients = torch.linalg.solve(A + regularizer * torch.eye(A.shape[0], dtype=A.dtype, device=A.device), Q.T @ pooled)
        p0 = -Q @ coefficients
    else:
        A, coefficients, p0 = Q.T @ Q, pooled[:0], torch.zeros_like(pooled)
    e = pooled - Y @ coefficients
    return {
        "g": pooled, "Q": Q, "Ya": Ya, "Yb": Yb, "Y": Y,
        "A": A, "c": coefficients, "p0": p0, "e": e,
        "history": torch.zeros_like(pooled) if history is None else history,
    }


def analytic_proposals(base: dict[str, Tensor], directions: int) -> Tensor:
    candidates = [base["e"]]
    Ya, Yb, Q = base["Ya"], base["Yb"], base["Q"]
    for index in range(Q.shape[1]):
        candidates.extend((0.5 * (Ya[:, index] + Yb[:, index]), Ya[:, index] - Yb[:, index]))
    return orth(torch.stack(candidates, dim=1))[:, :directions]


def _cg(operator, gradient: Tensor, iterations: int) -> Tensor:
    step = torch.zeros_like(gradient)
    residual = -gradient
    direction = residual.clone()
    for _ in range(iterations):
        action = operator(direction)
        rr = residual @ residual
        alpha = rr / (direction @ action).clamp_min(1.0e-30)
        step = step + alpha * direction
        next_residual = residual - alpha * action
        beta = (next_residual @ next_residual) / rr.clamp_min(1.0e-30)
        direction = next_residual + beta * direction
        residual = next_residual
    return step


def _nystrom_step(operator, gradient: Tensor, rank: int, block_id: Tensor) -> Tensor:
    dimension = gradient.numel()
    rank = min(max(1, rank), dimension)
    ids = block_id.to(gradient)
    columns = []
    for index in range(rank):
        phase = torch.sin((torch.arange(dimension, device=gradient.device, dtype=gradient.dtype) + 1) * (index + 1) * 0.754877666)
        block_phase = torch.cos((ids + 1) * (index + 1) * 1.324717957)
        columns.append(torch.sign(phase + 0.35 * block_phase))
    omega = orth(torch.stack(columns, dim=1))
    response = operator.columns(omega)
    basis = orth(response)
    projected = sym(basis.T @ operator.columns(basis))
    eigenvalues = torch.linalg.eigvalsh(projected)
    residual_scale = eigenvalues.median().clamp_min(1.0e-8)
    parallel = basis @ torch.linalg.solve(projected, basis.T @ gradient)
    perpendicular = gradient - basis @ (basis.T @ gradient)
    return -(parallel + perpendicular / residual_scale)


def _block_schur_step(
    normal: Tensor,
    gradient: Tensor,
    split: int,
    block_id: Tensor,
    counts: dict,
) -> Tensor:
    """Eliminate independent factor-variable blocks before the camera solve."""
    if split <= 0 or split >= gradient.numel():
        raise ValueError("Schur split must partition the parameter vector")
    Hcc = normal[:split, :split]
    Hcl = normal[:split, split:]
    gc, gl = gradient[:split], gradient[split:]
    landmark_ids = block_id[split:]
    schur = Hcc.clone()
    rhs = -gc.clone()
    inverse_cross: dict[int, Tensor] = {}
    inverse_gradient: dict[int, Tensor] = {}
    for block in torch.unique(landmark_ids, sorted=True).tolist():
        local = (landmark_ids == block).nonzero().flatten() + split
        Hll = normal[local][:, local]
        Hlc = normal[local, :split]
        solved_cross = torch.linalg.solve(Hll, Hlc)
        solved_gradient = torch.linalg.solve(Hll, gradient[local])
        counts["small_linear_solves"] += 2
        schur = schur - Hcl[:, local - split] @ solved_cross
        rhs = rhs + Hcl[:, local - split] @ solved_gradient
        inverse_cross[int(block)] = solved_cross
        inverse_gradient[int(block)] = solved_gradient
    camera_step = torch.linalg.solve(sym(schur), rhs)
    counts["dense_linear_solves"] += 1
    step = torch.zeros_like(gradient)
    step[:split] = camera_step
    for block in torch.unique(landmark_ids, sorted=True).tolist():
        local = (landmark_ids == block).nonzero().flatten() + split
        step[local] = -(inverse_gradient[int(block)] + inverse_cross[int(block)] @ camera_step)
    return step


def solve(problem, model=None, method: str = "csn", config: Stage2Config | None = None) -> dict:
    config = config or Stage2Config()
    counts = initial_counters()
    x = problem.initial.detach().clone()
    block_id = problem.block_id.to(x.device)
    edges = _block_edges(problem, block_id)
    history = None
    trajectory = []
    damping = float(config.damping)
    started = time.perf_counter()

    def costs(candidate: Tensor) -> Tensor:
        counts["objective_evaluations"] += 2
        counts["residual_evaluations"] += 2
        return torch.stack([problem.cost(candidate, view) for view in (0, 1)]).detach()

    initial_cost = costs(x)
    for iteration in range(config.steps):
        opa = BatchedGNView(problem, x, 0, damping, counts)
        opb = BatchedGNView(problem, x, 1, damping, counts)
        ga, gb = opa.g, opb.g
        pooled_gradient = 0.5 * (ga + gb)
        before = costs(x)
        whitener = estimate_block_whitener(
            ga, gb, opa, opb, block_id, edges, config.whitening_probes, counts
        )
        white_a, white_b = WhitenedOperator(opa, whitener), WhitenedOperator(opb, whitener)
        white_ga = whitener.to_white_gradient(ga)
        white_gb = whitener.to_white_gradient(gb)
        white_history = None if history is None else whitener.to_white_step(history)
        base = measured_base(
            white_ga, white_gb, white_a, white_b, block_id, edges,
            history=white_history, count=config.measured_directions,
        )
        counts["small_linear_solves"] += 1
        diag = {"rank": 0, "gain": 0.0, "gap": 0.0, "changes": [0.0, 0.0], "lock_error": 0.0, "weight": 0.5}
        if method in ("csn", "mean_schur", "analytic_csn", "base"):
            if method == "base":
                white_step = base["p0"]
            else:
                if method in ("csn", "mean_schur"):
                    if model is None:
                        raise ValueError("learned CSN requires a model")
                    with torch.no_grad():
                        proposals = model.to(device=x.device, dtype=x.dtype)(
                            base, white_ga, white_gb, block_id, edges,
                            directions=config.completion_directions,
                        )
                    counts["network_calls"] += 1
                else:
                    proposals = analytic_proposals(base, config.completion_directions)
                white_step, _, diag = completion(
                    base, white_ga, white_gb, white_a, white_b, proposals,
                    consensus=method != "mean_schur", counts=counts,
                )
        elif method in ("cg", "block_pcg"):
            average = lambda vector: 0.5 * (white_a(vector) + white_b(vector))
            white_step = _cg(average, 0.5 * (white_ga + white_gb), config.cg_iterations)
        elif method == "nystrom":
            class Average:
                def columns(self, vectors):
                    return 0.5 * (white_a.columns(vectors) + white_b.columns(vectors))
                def __call__(self, vector):
                    return self.columns(vector[:, None])[:, 0]
            white_step = _nystrom_step(Average(), 0.5 * (white_ga + white_gb), config.nystrom_rank, block_id)
        elif method in ("native_lm", "schur_lm"):
            jacobian_a, jacobian_b = opa.full_jacobian(), opb.full_jacobian()
            normal = 0.5 * (jacobian_a.T @ jacobian_a + jacobian_b.T @ jacobian_b)
            normal = normal + damping * torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            if method == "schur_lm":
                split = int((problem.metadata or {}).get("schur_split", 0))
                parameter_step = _block_schur_step(normal, pooled_gradient, split, block_id, counts)
            else:
                parameter_step = torch.linalg.solve(normal, -pooled_gradient)
                counts["dense_linear_solves"] += 1
            white_step = whitener.to_white_step(parameter_step)
        else:
            raise ValueError(f"unknown stage-2 method: {method}")
        parameter_step = whitener.to_parameter(white_step).detach()
        white_average_action = 0.5 * (white_a(white_step.detach()) + white_b(white_step.detach()))
        predicted = float((-(0.5 * (white_ga + white_gb)) @ white_step - 0.5 * white_step @ white_average_action).detach())
        directional = float(pooled_gradient @ parameter_step)
        accepted, scale, after, rho = False, 0.0, before, float("nan")
        for backtrack in range(config.backtracks):
            alpha = 0.5**backtrack
            candidate = x + alpha * parameter_step
            counts["evaluated_states"] += 1
            candidate_cost = costs(candidate)
            bound = before.mean() + config.armijo * alpha * directional
            if torch.isfinite(candidate_cost).all() and float(candidate_cost.mean()) <= float(bound) + 1.0e-14:
                actual = float(before.mean() - candidate_cost.mean())
                scaled_prediction = max(alpha * predicted, 1.0e-30)
                rho = actual / scaled_prediction
                history = candidate - x
                x = candidate.detach()
                after = candidate_cost
                accepted, scale = True, alpha
                break
        damping_before = damping
        if accepted:
            if rho > 0.75 and scale == 1.0:
                damping = max(config.damping_min, damping * 0.3)
            elif rho < 0.25:
                damping = min(config.damping_max, damping * 3.0)
        else:
            history = None
            damping = min(config.damping_max, damping * 10.0)
        trajectory.append(
            {
                "iteration": iteration,
                "cost_before": before.tolist(), "cost_after": after.tolist(),
                "accepted": accepted, "step_scale": scale,
                "damping_before": damping_before, "damping_after": damping,
                "actual_to_predicted_ratio": rho,
                "predicted_reduction": predicted,
                "directional_derivative": directional,
                "whitening": whitener.diagnostics,
                **diag,
            }
        )
    return {
        "theta": x,
        "seconds": time.perf_counter() - started,
        "counts": counts,
        "trajectory": trajectory,
        "config": asdict(config),
        "initial_cost": initial_cost.tolist(),
        "final_cost": costs(x).tolist(),
    }


def solve_adam(problem, *, steps: int = 100, learning_rate: float = 0.03, time_budget: float | None = None) -> dict:
    """Monotone Adam control, optionally stopped at an observed wall-clock budget."""
    x = problem.initial.detach().clone()
    m, v = torch.zeros_like(x), torch.zeros_like(x)
    start = time.perf_counter()
    initial = [float(problem.cost(x, view)) for view in (0, 1)]
    history = []
    evaluations = 1
    for iteration in range(steps):
        if time_budget is not None and iteration > 0 and time.perf_counter() - start >= time_budget:
            break
        variable = x.detach().requires_grad_(True)
        costs = torch.stack([problem.cost(variable, view) for view in (0, 1)])
        gradient = torch.autograd.grad(costs.mean(), variable)[0].detach()
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient.square()
        proposal = -learning_rate * (m / (1 - 0.9 ** (iteration + 1))) / ((v / (1 - 0.999 ** (iteration + 1))).sqrt() + 1.0e-8)
        accepted = False
        for backtrack in range(6):
            alpha = 0.5**backtrack
            candidate = x + alpha * proposal
            candidate_cost = torch.stack([problem.cost(candidate, view) for view in (0, 1)]).detach()
            evaluations += 1
            if torch.isfinite(candidate_cost).all() and float(candidate_cost.mean()) <= float(costs.detach().mean()):
                x = candidate.detach()
                accepted = True
                break
        history.append({"iteration": iteration, "accepted": accepted, "step_scale": alpha if accepted else 0.0})
    return {
        "theta": x,
        "seconds": time.perf_counter() - start,
        "steps": len(history),
        "evaluated_states": evaluations,
        "trajectory": history,
        "initial_cost": initial,
        "final_cost": [float(problem.cost(x, view)) for view in (0, 1)],
    }


__all__ = [
    "BatchedGNView", "BlockWhitener", "Stage2Config", "analytic_proposals",
    "estimate_block_whitener", "initial_counters", "measured_base", "solve",
    "solve_adam", "structured_directions",
]
