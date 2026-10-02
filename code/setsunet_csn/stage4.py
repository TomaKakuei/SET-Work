"""Normalized CSN with identified curvature, equivariant probes and certified completion."""
from dataclasses import dataclass,asdict,replace
from contextlib import nullcontext
import time
import torch
from .core import sym, minimax
from .curvature_stage4 import CurvatureConfig,BlockMetric,DenseView,linearize
from .model_stage2 import _segment_sum,_neighbor_mean
from .stage2 import _cg
from .stage3 import predicted_reduction


@dataclass
class Stage4Config(CurvatureConfig):
    steps:int=5
    measured_directions:int=4
    completion_directions:int=4
    damping:float=1e-3
    damping_min:float=1e-9
    damping_max:float=1e6
    backtracks:int=6
    armijo:float=1e-4
    rank_tolerance:float=1e-9
    certificate_tolerance:float=1e-7
    cg_iterations:int=8
    backend:str='torch'


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
    Qs,Yas,Ybs=[],[],[]
    candidates=iter(seeds)
    exhausted=False
    krylov_index=0
    while len(Qs)<min(count,g.numel()):
        if not exhausted:
            try: v=next(candidates)
            except StopIteration: exhausted=True; continue
        else:
            if krylov_index>=len(Qs): break
            v=.5*(Yas[krylov_index]+Ybs[krylov_index])
            krylov_index+=1
        original=v.norm().detach()
        for _ in range(2):
            if Qs:
                Q=torch.stack(Qs,dim=1)
                v=v-Q@(Q.T@v)
        length=v.norm()
        if float(length)<=config.rank_tolerance*max(float(original),1e-300): continue
        q=v/length
        Qs.append(q)
        Yas.append(a(q)); Ybs.append(b(q))
    Q=torch.stack(Qs,dim=1) if Qs else g.new_zeros(g.numel(),0)
    Ya=torch.stack(Yas,dim=1) if Yas else Q
    Yb=torch.stack(Ybs,dim=1) if Ybs else Q
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
    config=config or Stage4Config()
    if method not in ('csn','analytic','mean','lm','pcg'): raise ValueError(method)
    if training and method!='csn': raise ValueError('Training requires learned proposals')
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
    times=dict(linearization=0.,whitening=0.,probes=0.,network=0.,completion=0.,line_search=0.,classical=0.)
    trajectory,losses=[],[]
    history=None
    damping=config.damping
    started=time.perf_counter()
    def costs(z):
        counts['objective_evaluations']+=2
        return torch.stack([problem.cost(z,v) for v in (0,1)]).detach()
    initial=costs(x)
    before=initial
    for iteration in range(config.steps):
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
            metric=BlockMetric(ids,normal=normal,local=local)
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
                    counts['network_calls']+=1
                times['network']+=time.perf_counter()-t
                t=time.perf_counter()
                wstep,waction,gain,diag=complete(base,wa,wb,V,config,counts,consensus=method!='mean')
                if training:
                    residual=(base['g']+waction).square().sum()/base['g'].square().sum().clamp_min(1e-30)
                    unit=V/V.norm(dim=0,keepdim=True).clamp_min(1e-20)
                    gram=unit.T@unit
                    diversity=(gram-torch.diag_embed(gram.diagonal())).square().mean()
                    losses.append(residual+.005*diversity)
                times['completion']+=time.perf_counter()-t
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
    result=dict(theta=x.to(original_dtype),seconds=time.perf_counter()-started,counts=counts,timing=times,trajectory=trajectory,
        initial_cost=initial.tolist(),final_cost=before.tolist(),config=asdict(config),forward_dtype=str(original_dtype),algebra_dtype='torch.float64')
    if training: result['loss']=torch.stack(losses).mean()
    return result


class SETSUNETCSNStage4:
    def __init__(self,model,config=None): self.model,self.config=model,config or Stage4Config(measured_directions=8,completion_directions=8,backend='c')
    @classmethod
    def from_checkpoint(cls,path,config=None):
        from .model_stage4 import NormalizedSchurNet
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!='setsunet-csn-stage4-v1': raise ValueError('Stage 4 requires trained normalized features and all direction heads')
        model=NormalizedSchurNet(**payload['architecture']).double()
        model.load_state_dict(payload['state_dict'])
        return cls(model.eval(),config)
    def __call__(self,problem): return solve(problem,self.model,'csn',self.config)
