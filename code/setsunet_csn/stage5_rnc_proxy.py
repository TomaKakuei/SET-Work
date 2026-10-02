"""Research-only finite-curve controls for Stage5 CSN and LM proxies.

This module leaves the active solver untouched.  The straight CSN route is an
executable reproduction control.  The RNC2 routes keep the registered tangent
proposal and change only its finite realization to x+t*v+t^2*a/2.
"""
from dataclasses import asdict,replace
from contextlib import nullcontext
import time
import numpy as np
import torch

from .curvature_stage4 import BlockMetric
from .curvature_stage5 import BlockLayout,linearize
from .stage2 import _cg
from .stage3 import predicted_reduction
from .stage5 import Stage5Config,WhiteView,complete,counters,measured_base,orthogonalize


def _wrapped(problem):
    dtype=problem.initial.dtype
    if dtype==torch.float64:return problem,dtype
    original=problem
    return replace(original,
        raw_function=lambda z,v:original.raw(z.to(dtype),v).to(z.dtype),
        cost_function=lambda z,v:original.cost(z.to(dtype),v).to(z.dtype),
        weight_function=lambda z,v:original.weights(z.to(dtype),v).to(z.dtype)),dtype


def _residual_linearization(engine,x):
    r,J=engine.least_squares(x.detach().cpu().numpy(),True)
    J=J.toarray() if hasattr(J,'toarray') else np.asarray(J)
    return (torch.as_tensor(r,dtype=torch.float64,device=x.device),
            torch.as_tensor(J,dtype=torch.float64,device=x.device))


def _rnc2(engine,x,velocity,damping,r=None,J=None,h=.02):
    made_linearization=r is None
    if made_linearization:r,J=_residual_linearization(engine,x)
    diagonal=(J*J).sum(0).clamp_min(1e-30)
    G=J.T@J+damping*torch.diag(diagonal)
    trial=x+h*velocity
    forward,_=engine.least_squares(trial.detach().cpu().numpy(),False)
    forward=torch.as_tensor(forward,dtype=torch.float64,device=x.device)
    fvv=2*(forward-r-h*(J@velocity))/(h*h)
    try:acceleration=torch.linalg.solve(G,-J.T@fvv)
    except torch.linalg.LinAlgError:acceleration=torch.linalg.lstsq(G,-J.T@fvv).solution
    ratio=float(acceleration.norm()/velocity.norm().clamp_min(1e-300))
    return acceleration.detach(),ratio,dict(curve_residual_evaluations=1,curve_linear_solves=1,
        curve_jacobian_evaluations=int(made_linearization))


def solve(problem,model,engine,tangent='csn',curve='straight',config=None):
    """Five-step research route with a fixed tangent/curve pairing.

    tangent: csn, lm_proxy, or dense_same_curvature.
    curve: straight or rnc2.  RNC2 uses the explicit shared residual map and a
    More-scaled damped GN metric.  It is an RNC-inspired second-order control,
    not a claim of reproducing the paper's arbitrary-order RNC-LM.
    """
    if tangent not in ('csn','lm_proxy','dense_same_curvature'):raise ValueError(tangent)
    if curve not in ('straight','rnc2'):raise ValueError(curve)
    problem,original_dtype=_wrapped(problem);x=problem.initial.detach().to(torch.float64).clone()
    ids=problem.block_id.to(x.device);layout=BlockLayout(ids) if hasattr(problem,'native') else None
    if model is not None:model.to(device=x.device,dtype=x.dtype)
    config=config or Stage5Config();counts=counters();history=None;damping=config.damping
    trajectory=[];snapshots={};started=time.perf_counter()

    def costs(z):
        counts['objective_evaluations']+=2
        fn=problem.native.cost if hasattr(problem,'native') else problem.cost
        if hasattr(problem,'native'):
            counts['factor_evaluations']=counts.get('factor_evaluations',0)+2
            if getattr(problem.native,'evaluates_in_c',False):counts['native_calls']+=2
        return torch.stack([fn(z,v) for v in (0,1)]).detach()

    initial=costs(x);before=initial
    for iteration in range(config.steps):
        diag=dict(rank=0,gain=0.,gap=0.,lock_error=0.,certificate_fallback=False)
        actual_rank=0;r=J=None
        if tangent=='lm_proxy':
            r,J=_residual_linearization(engine,x);counts['jacobian_evaluations']+=1
            diagonal=(J*J).sum(0).clamp_min(1e-30)
            H=J.T@J+damping*torch.diag(diagonal);g=J.T@r
            try:step=torch.linalg.solve(H,-g)
            except torch.linalg.LinAlgError:step=torch.linalg.lstsq(H,-g).solution
            action=H@step;counts['dense_linear_solves']+=1
            directional,quadratic=float(g@step),float(step@action)
            curvature='explicit residual GN + More diagonal damping'
        else:
            a,b,edges,curvature=linearize(problem,x,damping,counts,config);g=.5*(a.g+b.g)
            if a.H is None or b.H is None:raise ValueError('dense research control requires dense curvature')
            if tangent=='dense_same_curvature':
                H=.5*(a.H+b.H);step=torch.linalg.solve(H,-g);action=H@step
                counts['dense_linear_solves']+=1
                directional,quadratic=float(g@step),float(step@action)
            else:
                normal=.5*(a.H+b.H)
                metric=layout.metric(normal) if layout is not None else BlockMetric(ids,normal=normal)
                wa,wb=WhiteView(a,metric,counts),WhiteView(b,metric,counts)
                base=measured_base(wa,wb,ids,edges,config.measured_directions,
                    None if history is None else metric.apply(history,inverse=False),config,counts)
                actual_rank=base['Q'].shape[1]
                if model is None:raise ValueError('missing shared CSN model')
                with torch.no_grad():V=model(base,wa.g,wb.g,ids,edges,directions=config.completion_directions)
                if getattr(model,'is_learned_proposal',True):counts['network_calls']+=1
                wstep,waction,gain,diag=complete(base,wa,wb,V,config,counts,consensus=True)
                if config.residual_cg_iterations:
                    remaining=base['g']+waction;op=lambda v:.5*(wa(v)+wb(v))
                    delta=_cg(op,remaining,config.residual_cg_iterations);delta_action=op(delta)
                    gain_cg=float(-(remaining@delta+.5*delta@delta_action))
                    accepted_cg=bool(torch.isfinite(delta).all() and gain_cg>=0.)
                    if accepted_cg:wstep,waction=wstep+delta,waction+delta_action
                    diag.update(residual_cg_accepted=accepted_cg,residual_cg_predicted_gain=gain_cg,
                        residual_before_cg=float(remaining.norm()),residual_after_cg=float((base['g']+waction).norm()))
                    counts['residual_cg_iterations']=counts.get('residual_cg_iterations',0)+config.residual_cg_iterations
                step=metric.apply(wstep).detach()
                directional=float(.5*(wa.g+wb.g)@wstep.detach());quadratic=float(wstep.detach()@waction.detach())

        acceleration=torch.zeros_like(step);acceleration_ratio=0.;curve_counts={}
        if curve=='rnc2':
            acceleration,acceleration_ratio,curve_counts=_rnc2(engine,x,step,damping,r,J)
            for key,value in curve_counts.items():counts[key]=counts.get(key,0)+value
        accepted=False;alpha=0.;after=before;rho=None
        origin=x
        for backtrack in range(config.backtracks):
            scale=.5**backtrack
            candidate=origin+scale*step+.5*scale*scale*acceleration
            observed=costs(candidate);counts['evaluated_states']+=1
            predicted=predicted_reduction(directional,quadratic,scale)
            actual=float(before.mean()-observed.mean())
            trial_rho=actual/max(predicted,1e-300)
            armijo_bound=float(before.mean())+config.armijo*scale*directional+1e-14*max(float(before.abs().mean()),1e-30)
            if torch.isfinite(observed).all() and predicted>0 and float(observed.mean())<=armijo_bound and trial_rho>1e-3:
                history=candidate-origin;x=candidate.detach();accepted=True;alpha=scale;after=observed;rho=trial_rho;break
        old=damping
        if accepted:
            if rho>.75 and alpha==1.:damping=max(config.damping_min,damping*.3)
            elif rho<.25:damping=min(config.damping_max,damping*3.)
        else:history=None;damping=min(config.damping_max,damping*10.)
        trajectory.append(dict(iteration=iteration,accepted=accepted,step_scale=alpha,cost_before=before.tolist(),
            cost_after=after.tolist(),damping_before=old,damping_after=damping,actual_to_predicted_ratio=rho,
            actual_measured_rank=actual_rank,curvature=curvature,tangent=tangent,curve=curve,
            acceleration_ratio=acceleration_ratio,curve_correction_norm=float((.5*alpha*alpha*acceleration).norm()),**diag))
        before=after
        if iteration+1 in config.snapshot_steps:snapshots[str(iteration+1)]=dict(theta=x.tolist(),iterations=iteration+1,final_cost=before.tolist())
    return dict(theta=x.to(original_dtype),seconds=time.perf_counter()-started,counts=counts,trajectory=trajectory,
        snapshots=snapshots,initial_cost=initial.tolist(),final_cost=before.tolist(),iterations=len(trajectory),
        termination='step_limit',config=asdict(config),implementation=dict(
            optimizer='research_'+tangent+'_'+curve,
            curve_scope='second-order RNC-inspired finite update; not arbitrary-order RNC-LM',
            tangent_scope='complete registered CSN(+CG) proposal' if tangent=='csn' else tangent,
            active_solver_changed=False))
