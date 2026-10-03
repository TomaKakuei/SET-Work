# Isolated copy of frozen finite_models_20260925/backbone.py; opt-in adaptation.
# Original SHA256: 74717a79206ccf17f51cb6144f769a65a5f4c0f80c48bd90781cf3e8670bcf16
"""Registered curve-envelope ablations under one five-step outer controller.

This research module leaves the active solver untouched.  It crosses two
registered tangent constructions with five finite realizations while keeping
the rho/Armijo controller, inputs, initial states, and step count fixed.
"""
from dataclasses import asdict
import time

import torch

from setsunet_csn.curvature_stage4 import BlockMetric
from setsunet_csn.curvature_stage5 import BlockLayout, linearize
from setsunet_csn.stage2 import _cg
from setsunet_csn.stage3 import predicted_reduction
from setsunet_csn.stage5 import Stage5Config, WhiteView, complete, counters, measured_base
from setsunet_csn.stage5_rnc_proxy import _rnc2, _wrapped
from .controller import intervention_damping


CURVE_MODES = (
    'straight', 'raw_rnc2', 'norm_only',
    'norm_mean_envelope', 'norm_view_envelope',
)


def _acceleration(engine, x, velocity, damping, mode):
    if mode == 'straight':
        return torch.zeros_like(velocity), dict(
            raw_acceleration_ratio=0.0, curve_trust_scale=0.0,
            trusted_acceleration_ratio=0.0)
    acceleration, raw_ratio, extra = _rnc2(engine, x, velocity, damping)
    scale = 1.0
    if mode != 'raw_rnc2':
        scale = min(1.0, 2.0 / max(raw_ratio, 1e-300))
    if not torch.isfinite(acceleration).all():
        acceleration = torch.zeros_like(velocity)
        scale = 0.0
    acceleration = acceleration * scale
    extra.update(
        raw_acceleration_ratio=raw_ratio,
        curve_trust_scale=scale,
        trusted_acceleration_ratio=float(
            acceleration.norm() / velocity.norm().clamp_min(1e-300)),
    )
    return acceleration, extra


def _select_curve(mode, straight_cost, curved_cost):
    if mode == 'straight':
        return False
    if mode in ('raw_rnc2', 'norm_only'):
        return bool(torch.isfinite(curved_cost).all())
    tolerance = 1e-14 * straight_cost.abs().clamp_min(1e-30)
    if mode == 'norm_mean_envelope':
        return bool(torch.isfinite(curved_cost).all() and
                    curved_cost.mean() <= straight_cost.mean() + tolerance.mean())
    if mode == 'norm_view_envelope':
        return bool(torch.isfinite(curved_cost).all() and
                    torch.all(curved_cost <= straight_cost + tolerance))
    raise ValueError(mode)


def solve(problem, model, engine, tangent='csn', curve_mode='straight', config=None,
          hook=None, sync_intervention_damping=False):
    """Run a registered tangent/curve pair for exactly ``config.steps``."""
    if tangent not in ('csn', 'dense_same_curvature'):
        raise ValueError(tangent)
    if curve_mode not in CURVE_MODES:
        raise ValueError(curve_mode)
    problem, original_dtype = _wrapped(problem)
    x = problem.initial.detach().to(torch.float64).clone()
    ids = problem.block_id.to(x.device)
    layout = BlockLayout(ids) if hasattr(problem, 'native') else None
    if model is not None:
        model.to(device=x.device, dtype=x.dtype)
    config = config or Stage5Config()
    counts = counters()
    history = None
    damping = config.damping
    trajectory, snapshots = [], {}
    started = time.perf_counter()

    def costs(z):
        counts['objective_evaluations'] += 2
        fn = problem.native.cost if hasattr(problem, 'native') else problem.cost
        if hasattr(problem, 'native'):
            counts['factor_evaluations'] = counts.get('factor_evaluations', 0) + 2
            if getattr(problem.native, 'evaluates_in_c', False):
                counts['native_calls'] += 2
        return torch.stack([fn(z, view) for view in (0, 1)]).detach()

    initial = costs(x)
    before = initial
    for iteration in range(config.steps):
        diag = dict(rank=0, gain=0., gap=0., lock_error=0.,
                    certificate_fallback=False)
        actual_rank = 0
        a, b, edges, curvature = linearize(problem, x, damping, counts, config)
        g = .5 * (a.g + b.g)
        if a.H is None or b.H is None:
            raise ValueError('curve ablation requires dense registered curvature')
        if tangent == 'dense_same_curvature':
            H = .5 * (a.H + b.H)
            step = torch.linalg.solve(H, -g)
            action = H @ step
            counts['dense_linear_solves'] += 1
            directional, quadratic = float(g @ step), float(step @ action)
        else:
            normal = .5 * (a.H + b.H)
            metric = layout.metric(normal) if layout is not None else BlockMetric(ids, normal=normal)
            wa, wb = WhiteView(a, metric, counts), WhiteView(b, metric, counts)
            base = measured_base(
                wa, wb, ids, edges, config.measured_directions,
                None if history is None else metric.apply(history, inverse=False),
                config, counts)
            actual_rank = base['Q'].shape[1]
            if model is None:
                raise ValueError('missing shared CSN model')
            with torch.no_grad():
                V = model(base, wa.g, wb.g, ids, edges,
                          directions=config.completion_directions)
            if getattr(model, 'is_learned_proposal', True):
                counts['network_calls'] += 1
            wstep, waction, _, diag = complete(
                base, wa, wb, V, config, counts, consensus=True)
            if config.residual_cg_iterations:
                remaining = base['g'] + waction
                op = lambda v: .5 * (wa(v) + wb(v))
                delta = _cg(op, remaining, config.residual_cg_iterations)
                delta_action = op(delta)
                gain_cg = float(-(remaining @ delta + .5 * delta @ delta_action))
                accepted_cg = bool(torch.isfinite(delta).all() and gain_cg >= 0.)
                if accepted_cg:
                    wstep, waction = wstep + delta, waction + delta_action
                diag.update(
                    residual_cg_accepted=accepted_cg,
                    residual_cg_predicted_gain=gain_cg,
                    residual_before_cg=float(remaining.norm()),
                    residual_after_cg=float((base['g'] + waction).norm()),
                )
                counts['residual_cg_iterations'] = (
                    counts.get('residual_cg_iterations', 0) +
                    config.residual_cg_iterations)
            step = metric.apply(wstep).detach()
            directional = float(.5 * (wa.g + wb.g) @ wstep.detach())
            quadratic = float(wstep.detach() @ waction.detach())

        acceleration, curve_diag = _acceleration(
            engine, x, step, damping, curve_mode)
        for key, value in curve_diag.items():
            if key.startswith('curve_') and key.endswith('evaluations'):
                counts[key] = counts.get(key, 0) + value
            elif key == 'curve_linear_solves':
                counts[key] = counts.get(key, 0) + value

        accepted = False
        alpha, after, rho, selected_curve = 0., before, None, False
        straight_cost = curved_cost = None
        origin = x
        for backtrack in range(config.backtracks):
            scale = .5 ** backtrack
            straight = origin + scale * step
            straight_observed = costs(straight)
            if curve_mode == 'straight':
                curved = straight
                curved_observed = straight_observed
                counts['evaluated_states'] += 1
            else:
                curved = straight + .5 * scale * scale * acceleration
                curved_observed = costs(curved)
                counts['evaluated_states'] += 2
            use_curve = _select_curve(curve_mode, straight_observed, curved_observed)
            candidate = curved if use_curve else straight
            observed = curved_observed if use_curve else straight_observed
            predicted = predicted_reduction(directional, quadratic, scale)
            actual = float(before.mean() - observed.mean())
            trial_rho = actual / max(predicted, 1e-300)
            armijo_bound = (float(before.mean()) + config.armijo * scale * directional +
                            1e-14 * max(float(before.abs().mean()), 1e-30))
            if (torch.isfinite(observed).all() and predicted > 0 and
                    float(observed.mean()) <= armijo_bound and trial_rho > 1e-3):
                history = candidate - origin
                x = candidate.detach()
                accepted, alpha, after, rho = True, scale, observed, trial_rho
                selected_curve = use_curve
                straight_cost = straight_observed.tolist()
                curved_cost = curved_observed.tolist()
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
        fiber_changed = False
        controller_override = None
        if hook is not None:
            x, after, fiber_changed = hook(iteration, origin, x, before, after, metric, base, V, step)
            if fiber_changed:
                history = x-origin
                accepted = True
                if sync_intervention_damping:
                    damping, controller_override = intervention_damping(
                        history, g, .5*(a.H+b.H), before, after,
                        old_damping, config)
                    controller_override.update(original_ratio=rho,
                        original_step_scale=alpha)
                    rho = controller_override['ratio']
                    counts['cached_matrix_products'] += 1
        trajectory.append(dict(
            iteration=iteration, accepted=accepted, step_scale=alpha, fiber_override=fiber_changed,
            cost_before=before.tolist(), cost_after=after.tolist(),
            damping_before=old_damping, damping_after=damping,
            actual_to_predicted_ratio=rho, actual_measured_rank=actual_rank,
            controller_override=controller_override,
            curvature=curvature, tangent=tangent, curve=curve_mode,
            curve_selected=selected_curve, straight_trial_cost=straight_cost,
            curved_trial_cost=curved_cost,
            curve_correction_norm=float((.5 * alpha * alpha * acceleration).norm()),
            **curve_diag, **diag))
        before = after
        if iteration + 1 in config.snapshot_steps:
            snapshots[str(iteration + 1)] = dict(
                theta=x.tolist(), iterations=iteration + 1,
                final_cost=before.tolist())
    return dict(
        theta=x.to(original_dtype), seconds=time.perf_counter() - started,
        counts=counts, trajectory=trajectory, snapshots=snapshots,
        initial_cost=initial.tolist(), final_cost=before.tolist(),
        iterations=len(trajectory), termination='step_limit',
        config=asdict(config), implementation=dict(
            optimizer='research_curve_ablation_'+tangent+'_'+curve_mode,
            active_solver_changed=False))
