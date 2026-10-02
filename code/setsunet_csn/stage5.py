"""Stage 5 native analytic inference; preserves the Stage 4 learned subspace algorithm."""
from dataclasses import dataclass,asdict,replace
from contextlib import nullcontext
import time
import torch
from .core import sym, minimax
from .curvature_stage4 import CurvatureConfig,BlockMetric,DenseView
from .curvature_stage5 import BlockLayout,linearize
from .model_stage2 import _segment_sum,_neighbor_mean
from .stage2 import _cg
from .stage3 import predicted_reduction


@dataclass
class Stage5Config(CurvatureConfig):
    steps:int=5
    measured_directions:int=8
    completion_directions:int=8
    damping:float=1e-3
    damping_min:float=1e-9
    damping_max:float=1e6
    backtracks:int=6
    armijo:float=1e-4
    rank_tolerance:float=1e-9
    certificate_tolerance:float=1e-7
    cg_iterations:int=8
    backend:str='c'
    time_limit:float|None=None
    fused_subspace:bool=True
    snapshot_steps:tuple=()
    residual_cg_iterations:int=0


def counters():
    return dict(residual_evaluations=0,objective_evaluations=0,weight_evaluations=0,
        jvp_directions=0,vjp_directions=0,curvature_products=0,cached_matrix_products=0,
        hessian_vector_products=0,jacobian_evaluations=0,linearizations=0,small_linear_solves=0,
        dense_linear_solves=0,network_calls=0,evaluated_states=1,native_calls=0)


def orthogonalize(V,tolerance=1e-9,reference=None,backend='torch'):
    if V.shape[1]==0: return V
    if backend=='c' and not V.requires_grad:
        from .native_stage4 import orthogonalize_c
        return orthogonalize_c(V,tolerance,reference)
    threshold=tolerance*(V.norm().detach() if reference is None else reference)
    columns=[]
    for v in V.unbind(1):
        for _ in range(2):
            if columns:
                Q=torch.stack(columns,dim=1)
                v=v-Q@(Q.T@v)
        norm=v.norm()
        if float(norm.detach())>float(threshold): columns.append(v/norm)
    return torch.stack(columns,dim=1) if columns else V[:,:0]


class WhiteView:
    def __init__(self,view,metric,counts):
        self.view,self.metric,self.counts=view,metric,counts
        self.H=metric.normal(view.H) if view.H is not None else None
        self.g=metric.apply(view.g)
    def columns(self,V):
        if self.H is not None:
            self.counts['curvature_products']+=V.shape[1]
            self.counts['cached_matrix_products']+=V.shape[1]
            return self.H@V
        return self.metric.apply(self.view.columns(self.metric.apply(V)))
    def __call__(self,v): return self.columns(v[:,None])[:,0]


def measured_base(a,b,ids,edges,count,history,config,counts):
    """Block-label invariant seeds; extra directions are measured Krylov actions."""
    ga,gb=a.g,b.g
    g=.5*(ga+gb)
    blocks=int(ids.max())+1
    energy=_segment_sum(g.square()[:,None],ids,blocks)[:,0]
    total=energy.mean().clamp_min(torch.finfo(g.dtype).tiny**.25)
    graph=_neighbor_mean((energy/total)[:,None],edges)[:,0]
    seeds=[g,ga-gb]
    if history is not None: seeds.append(history)
    seeds.extend((g*torch.tanh(graph[ids]),g*torch.tanh(energy[ids]/total)))
    # All candidates depend on tensors and graph structure, never numeric IDs.
    # The fused C measured kernel has eight coefficient slots. Wider solver
    # spaces use the existing general reference implementation.
    if a.H is None or b.H is None or not config.fused_subspace or count>8:
        from .stage4 import measured_base as reference_base
        return reference_base(a,b,ids,edges,count,history,config,counts)
    from .native_stage5 import measured
    Q,Ya,Yb=measured(a.H,b.H,torch.stack(seeds,dim=1),count,config.rank_tolerance)
    counts['curvature_products']+=2*Q.shape[1]
    counts['cached_matrix_products']+=2*Q.shape[1]
    counts['native_calls']+=1
    Y=.5*(Ya+Yb)
    A=sym(Q.T@Y)
    if Q.shape[1]:
        c=torch.linalg.solve(A,Q.T@g)
        counts['small_linear_solves']+=1
    else: c=g[:0]
    return dict(g=g,Q=Q,Ya=Ya,Yb=Yb,Y=Y,A=A,c=c,p0=-Q@c,e=g-Y@c,
        history=torch.zeros_like(g) if history is None else history)


def complete(base,a,b,V,config,counts,consensus=True):
    Q,A,Y=base['Q'],base['A'],base['Y']
    reference=V.norm().detach()
    Z=V
    if Q.shape[1]:
        Z=Z-Q@torch.linalg.solve(A,Y.T@Z)
        counts['small_linear_solves']+=1
        # Remove cancellation left by an ill-conditioned oblique projection.
        U=torch.linalg.qr(Y,mode='reduced')[0]
        for _ in range(2): Z=Z-U@(U.T@Z)
    Z=orthogonalize(Z,config.rank_tolerance,reference,config.backend)
    c=base['c']
    if not Z.shape[1]:
        pa,pb=-base['Ya']@c,-base['Yb']@c
        return base['p0'],.5*(pa+pb),base['g'].sum()*0,dict(rank=0,gain=0.,gap=0.,lock_error=0.,certificate_fallback=False,changes=[0.,0.])
    Za,Zb=a.columns(Z),b.columns(Z)
    Ha,Hb=sym(Z.T@Za),sym(Z.T@Zb)
    ba=Z.T@(a.g-base['Ya']@c)
    bb=Z.T@(b.g-base['Yb']@c)
    if consensus:
        if config.backend=='c':
            from .native_stage4 import minimax_c
            alpha,weight,gain,gap,changes=minimax_c(Ha,Hb,ba,bb,counts)
        else: alpha,weight,gain,gap,changes=minimax(Ha,Hb,ba,bb,counts)
    else:
        alpha=torch.linalg.solve(.5*(Ha+Hb),.5*(ba+bb))
        counts['small_linear_solves']+=1
        gain=.25*(ba+bb)@alpha
        changes=torch.stack((.5*alpha@Ha@alpha-ba@alpha,.5*alpha@Hb@alpha-bb@alpha))
        gap=changes.max()+gain
    action=-Y@c-.5*(Za+Zb)@alpha
    step=base['p0']-Z@alpha
    remaining=base['g']+action
    lock=(Q.T@remaining).norm()/base['g'].norm().clamp_min(1e-300)
    scale=max(float(gain.detach().abs()),float((base['g']@base['g']).detach()),1e-30)
    reject=not torch.isfinite(step).all() or float(lock.detach())>config.certificate_tolerance
    if consensus: reject=reject or float(changes.detach().max())>config.certificate_tolerance*scale or abs(float(gap.detach()))>config.certificate_tolerance*scale
    diag=dict(rank=Z.shape[1],gain=float(gain.detach()),gap=float(gap.detach()),lock_error=float(lock.detach()),
        changes=changes.detach().tolist(),certificate_fallback=bool(reject))
    if reject: return base['p0'],-Y@c,gain*0,diag
    return step,action,gain,diag


def solve(problem,model=None,method='csn',config=None,*,training=False):
    config=config or Stage5Config()
    if method not in ('csn','analytic','mean','lm','pcg'): raise ValueError(method)
    if training: raise ValueError('Stage 5 fused native path is inference-only; train with Stage 4')
    original_dtype=problem.initial.dtype
    if original_dtype!=torch.float64:
        # Legacy imaging kernels may have captured float32 filters/parameters.
        # Keep their declared forward dtype while differentiating through casts
        # and doing every curvature accumulation/subspace solve in float64.
        original=problem
        problem=replace(original,
            raw_function=lambda z,v:original.raw(z.to(original_dtype),v).to(z.dtype),
            cost_function=lambda z,v:original.cost(z.to(original_dtype),v).to(z.dtype),
            weight_function=lambda z,v:original.weights(z.to(original_dtype),v).to(z.dtype))
    x=problem.initial.detach().to(torch.float64).clone()
    ids=problem.block_id.to(x.device)
    if model is not None: model.to(device=x.device,dtype=x.dtype)
    counts=counters()
    times=dict(linearization=0.,whitening=0.,probes=0.,network=0.,analytic_proposal=0.,completion=0.,line_search=0.,classical=0.)
    trajectory,losses,snapshots=[],[],{}
    history=None
    damping=config.damping
    started=time.perf_counter()
    layout=BlockLayout(ids) if hasattr(problem,'native') else None
    def costs(z):
        counts['objective_evaluations']+=2
        fn=problem.native.cost if hasattr(problem,'native') else problem.cost
        if hasattr(problem,'native'):
            counts['factor_evaluations']=counts.get('factor_evaluations',0)+2
            if getattr(problem.native,'evaluates_in_c',False):counts['native_calls']+=2
        return torch.stack([fn(z,v) for v in (0,1)]).detach()
    initial=costs(x)
    before=initial
    for iteration in range(config.steps):
        if iteration and config.time_limit is not None and time.perf_counter()-started>=config.time_limit:break
        t=time.perf_counter()
        a,b,edges,curvature=linearize(problem,x,damping,counts,config)
        times['linearization']+=time.perf_counter()-t
        g=.5*(a.g+b.g)
        diag=dict(rank=0,gain=0.,gap=0.,lock_error=0.,certificate_fallback=False)
        actual_rank=0
        if method=='lm':
            t=time.perf_counter()
            if a.H is None or b.H is None: raise ValueError('Dense LM comparison requires dense curvature')
            H=.5*(a.H+b.H)
            step=torch.linalg.solve(H,-g)
            action=H@step
            counts['dense_linear_solves']+=1
            times['classical']+=time.perf_counter()-t
        else:
            t=time.perf_counter()
            normal=.5*(a.H+b.H) if a.H is not None and b.H is not None else None
            local=None if normal is not None else [.5*(ha+hb) for ha,hb in zip(a.local_blocks,b.local_blocks)]
            metric=layout.metric(normal) if layout is not None else BlockMetric(ids,normal=normal,local=local)
            wa,wb=WhiteView(a,metric,counts),WhiteView(b,metric,counts)
            times['whitening']+=time.perf_counter()-t
            if method=='pcg':
                t=time.perf_counter()
                wstep=_cg(lambda v:.5*(wa(v)+wb(v)),.5*(wa.g+wb.g),config.cg_iterations)
                waction=.5*(wa(wstep)+wb(wstep))
                times['classical']+=time.perf_counter()-t
            else:
                t=time.perf_counter()
                base=measured_base(wa,wb,ids,edges,config.measured_directions,
                    None if history is None else metric.apply(history,inverse=False),config,counts)
                actual_rank=base['Q'].shape[1]
                times['probes']+=time.perf_counter()-t
                t=time.perf_counter()
                if method=='analytic':
                    V=orthogonalize(torch.cat((base['e'][:,None],base['Y'],base['Ya']-base['Yb']),dim=1),
                        config.rank_tolerance,backend=config.backend)[:,:config.completion_directions]
                else:
                    if model is None: raise ValueError('Missing normalized network')
                    with nullcontext() if training else torch.no_grad():
                        V=model(base,wa.g,wb.g,ids,edges,directions=config.completion_directions)
                    if getattr(model,'is_learned_proposal',True):counts['network_calls']+=1
                learned=method!='analytic' and getattr(model,'is_learned_proposal',True)
                times['network' if learned else 'analytic_proposal']+=time.perf_counter()-t
                t=time.perf_counter()
                wstep,waction,gain,diag=complete(base,wa,wb,V,config,counts,consensus=method!='mean')
                if training:
                    residual=(base['g']+waction).square().sum()/base['g'].square().sum().clamp_min(1e-30)
                    unit=V/V.norm(dim=0,keepdim=True).clamp_min(1e-20)
                    gram=unit.T@unit
                    diversity=(gram-torch.diag_embed(gram.diagonal())).square().mean()
                    losses.append(residual+.005*diversity)
                times['completion']+=time.perf_counter()-t
                if config.residual_cg_iterations:
                    t=time.perf_counter()
                    remaining=base['g']+waction
                    op=lambda v:.5*(wa(v)+wb(v))
                    delta=_cg(op,remaining,config.residual_cg_iterations)
                    delta_action=op(delta)
                    gain_cg=float(-(remaining@delta+.5*delta@delta_action))
                    accepted_cg=bool(torch.isfinite(delta).all() and gain_cg>=0.)
                    if accepted_cg:wstep,waction=wstep+delta,waction+delta_action
                    diag.update(residual_cg_accepted=accepted_cg,residual_cg_predicted_gain=gain_cg,
                        residual_before_cg=float(remaining.norm()),residual_after_cg=float((base['g']+waction).norm()),
                        certificate_scope='CSN completion only; subsequent mean-objective CG is not consensus-certified')
                    counts['residual_cg_iterations']=counts.get('residual_cg_iterations',0)+config.residual_cg_iterations
                    times['classical']+=time.perf_counter()-t
            step=metric.apply(wstep).detach()
            # g'p and p'Bp are identical in the whitened coordinates; reuse
            # measured actions instead of another AD curvature evaluation.
            directional=float(.5*(wa.g+wb.g)@wstep.detach())
            quadratic=float(wstep.detach()@waction.detach())
        if method=='lm': directional,quadratic=float(g@step),float(step@action)
        t=time.perf_counter()
        accepted,alpha,after,rho=False,0.,before,None
        for backtrack in range(config.backtracks):
            scale=.5**backtrack
            candidate=x+scale*step
            observed=costs(candidate)
            counts['evaluated_states']+=1
            if torch.isfinite(observed).all() and float(observed.mean())<=float(before.mean())+config.armijo*scale*directional+1e-14*max(float(before.abs().mean()),1e-30):
                actual=float(before.mean()-observed.mean())
                predicted=predicted_reduction(directional,quadratic,scale)
                rho=actual/max(predicted,1e-300)
                history,x=candidate-x,candidate.detach()
                accepted,alpha,after=True,scale,observed
                break
        old=damping
        if accepted:
            if rho>.75 and alpha==1.: damping=max(config.damping_min,damping*.3)
            elif rho<.25: damping=min(config.damping_max,damping*3.)
        else: history=None; damping=min(config.damping_max,damping*10.)
        times['line_search']+=time.perf_counter()-t
        trajectory.append(dict(iteration=iteration,accepted=accepted,step_scale=alpha,cost_before=before.tolist(),cost_after=after.tolist(),
            damping_before=old,damping_after=damping,actual_to_predicted_ratio=rho,actual_measured_rank=actual_rank,
            curvature=curvature,**diag))
        before=after
        if iteration+1 in config.snapshot_steps:
            snapshots[str(iteration+1)]={'theta':x.tolist(),'seconds':time.perf_counter()-started,
                'iterations':iteration+1,'initial_cost':initial.tolist(),'final_cost':before.tolist()}
    result=dict(theta=x.to(original_dtype),seconds=time.perf_counter()-started,counts=counts,timing=times,trajectory=trajectory,
        initial_cost=initial.tolist(),final_cost=before.tolist(),config=asdict(config),forward_dtype=str(original_dtype),algebra_dtype='torch.float64')
    if training: result['loss']=torch.stack(losses).mean()
    result['iterations']=len(trajectory)
    result['termination']='time_limit' if len(trajectory)<config.steps else 'step_limit'
    result['implementation']={
        'optimizer':method,'evaluator_backend':getattr(getattr(problem,'native',None),'evaluator_backend','torch_autodiff'),
        'shared_interface_credit':'SETSUNET-CSN project factor evaluation and outer-loop implementation',
        'subspace_backend_option':config.backend,'fused_measured_subspace':config.fused_subspace,
        'backend_option_scope':'subspace kernels only; does not switch factor evaluator or model implementation',
        'native_calls_scope':'C factor evaluations plus fused measured calls; excludes C feature/proposal/orthogonalization/minimax calls',
        'network_calls_scope':'learned forward calls; analytic proposal modules excluded',
        'timing_scope':'internal solver timer; use external wall time for cross-implementation comparison'}
    if config.snapshot_steps:result['snapshots']=snapshots
    return result


class SETSUNETCSNStage5:
    def __init__(self,model,config=None): self.model,self.config=model,config or Stage5Config(measured_directions=8,completion_directions=8,backend='c')
    @classmethod
    def from_checkpoint(cls,path,config=None):
        from .model_stage5 import NormalizedSchurNet
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!='setsunet-csn-stage4-v1': raise ValueError('Stage 4 requires trained normalized features and all direction heads')
        model=NormalizedSchurNet(**payload['architecture']).double()
        model.load_state_dict(payload['state_dict'])
        return cls(model.eval(),config)
    @classmethod
    def from_profile(cls,family,budget='five_steps',*,selection_path=None,checkpoint=None):
        """Load the frozen per-family validation profile, never inspect a query."""
        from pathlib import Path
        import json,hashlib
        root=Path(__file__).resolve().parents[1]
        path=Path(selection_path) if selection_path is not None else root/'results_stage5/selection_final.json'
        selection=json.loads(path.read_text(encoding='utf-8'))
        rows=[r for r in selection['selected'] if r['family']==family and r['budget']==budget and r['configuration']['method']=='csn']
        if len(rows)!=1:raise ValueError(f'No frozen CSN profile for {family}/{budget}')
        chosen=dict(rows[0]['configuration']);chosen.pop('method');chosen.pop('id',None);k=chosen.pop('directions')
        iteration_cap=chosen.pop('iteration_cap',10000 if budget=='equal_time_20ms' else 5)
        checkpoint=Path(checkpoint) if checkpoint is not None else root/'checkpoints_stage4/csn_stage4_selected.pt'
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=selection['checkpoint_sha256']:
            raise ValueError('Profile/checkpoint fingerprint mismatch')
        config=Stage5Config(measured_directions=k,completion_directions=k,
            steps=iteration_cap,time_limit=.020 if budget=='equal_time_20ms' else None,**chosen)
        return cls.from_checkpoint(checkpoint,config)
    def __call__(self,problem): return solve(problem,self.model,'csn',self.config)
