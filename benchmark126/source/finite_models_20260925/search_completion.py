"""Second registered round: finish searches without replacing fitted models.

This module is separate from the first frozen four-variant implementation.
"""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from .core import ModelConfig, polynomial_segment, squared_polynomial
from .hooks import path_hook, finish, quadratic_design, poly
from . import backbone
import optimize_dimension_utility_20260924 as legacy_q
from run_fiber_system_20260924 import PhysicalAudit

METHODS=('path_triangle','quadratic_polished8')


def coordinate_polish(z,E,R,sweeps=3):
    """Exact quartic minimization per feasible coordinate; fixed bounded work."""
    z=np.array(z,copy=True);trace=[]
    def residual(x):return R@poly.basis(x,E)
    initial=.5*float(residual(z)@residual(z))
    for sweep in range(sweeps):
        gradient=(R@poly.dbasis(z,E)).T@residual(z)
        for j in np.argsort(-abs(gradient),kind='stable'):
            center=z.copy();center[j]=0.
            radius=min(1.,np.sqrt(max(0.,4-center@center)))
            e=np.zeros(len(z));e[j]=1.
            a=residual(center);minus=residual(center-e);plus=residual(center+e)
            b=.5*(plus-minus);c=.5*(plus+minus)-a
            t,diag=polynomial_segment(squared_polynomial([a,b,c]),-radius,radius)
            candidate=center+t*e
            oldr,newr=residual(z),residual(candidate)
            if newr@newr < oldr@oldr:z=candidate
        trace.append(.5*float(residual(z)@residual(z)))
    return z,dict(initial=initial,final=trace[-1],cost_trace=trace,sweeps=sweeps,
                  global_certificate=False,coordinate_minimizers='quartic stationary roots plus interval boundaries')


def triangle_coefficients(rr,J,B,pairs,gn,anchor,mu):
    import dictionary_matching_core_20260924 as c
    g2=B@c.monomials(gn,pairs);a2=B@c.monomials(anchor,pairs)
    cross=B@c.monomials(gn+anchor,pairs)-g2-a2
    C=np.column_stack([rr,J@gn,J@anchor,g2,cross,a2])
    regularizer=np.zeros((len(gn),6));regularizer[:,1]=np.sqrt(mu)*gn;regularizer[:,2]=np.sqrt(mu)*anchor
    R=np.linalg.qr(np.vstack([C,regularizer]),mode='r')
    return R


def triangle_search(R,old_t):
    def phi(x):
        a,b=x;return np.array([1,a,b,a*a,a*b,b*b])
    def dphi(x):
        a,b=x;return np.array([[0,0],[1,0],[0,1],[2*a,0],[b,a],[0,2*b]])
    def fun(x):v=R@phi(x);return .5*float(v@v)
    def jac(x):return (R@dphi(x)).T@(R@phi(x))
    points=[np.array(x,dtype=float) for x in [(0,0),(1,0),(0,1),(1/3,1/3),(1-old_t,old_t),(.5,0)]]
    statuses=[]
    for x in list(points):
        sol=minimize(fun,x,jac=jac,method='SLSQP',bounds=[(0.,1.)]*2,
            constraints=[dict(type='ineq',fun=lambda z:1-z.sum(),jac=lambda z:-np.ones(2))],
            options=dict(maxiter=50,ftol=1e-12))
        statuses.append(int(sol.status))
        if np.isfinite(sol.x).all():
            y=np.maximum(sol.x,0.);y/=max(1.,y.sum());points.append(y)
    # Check every boundary exactly, including the original GN-to-anchor chord.
    for start,end in [(points[0],points[1]),(points[0],points[2]),(points[1],points[2])]:
        r0,rm,r1=[R@phi(start+t*(end-start)) for t in [0.,.5,1.]]
        c=2*(r1-2*rm+r0);b=r1-r0-c
        t,_=polynomial_segment(squared_polynomial([r0,b,c]));points.append(start+t*(end-start))
    best=min(points,key=fun)
    return best,dict(alpha=float(best[0]),beta=float(best[1]),cost=fun(best),statuses=statuses,
                     boundaries_checked=3,global_certificate=False)


def triangle_hook(problem,audit,packet):
    config=ModelConfig('path_compact');parent_packet={}
    parent=path_hook(config,problem,audit,parent_packet)
    def hook(*args):
        if args[0]!=1:return args[2],args[4],False
        parent(*args)
        audit.phase='intervention';tic=time.perf_counter();a=parent_packet['arrays'];native=problem.native
        R=triangle_coefficients(a['rr'],a['J'],a['full'],a['pairs'],a['gn'],a['seed'],a['mu'])
        uv,info=triangle_search(R,parent_packet['detail']['line_solve']['t'])
        delta=uv[0]*a['gn']+uv[1]*a['seed']
        states=list(a['candidate_states']);residuals=list(a['candidate_residuals'])
        labels=['path','path_half','path_quarter','GN','GN_half','GN_quarter','anchor','anchor_half','anchor_quarter']
        for scale in [1.,.5,.25]:
            x=a['proposal']+a['T']@(scale*delta);states.append(x)
            residuals.append(native.least_squares(x,False)[0]);labels.append('triangle')
        selected,cost,changed,acceptance=finish(native,args[2],args[4],states,residuals,labels,config)
        a.update(selected=selected,selected_cost=cost,candidate_states=np.array(states),
                 candidate_residuals=np.array(residuals),triangle_R=R,triangle_uv=uv)
        detail=dict(parent_packet['detail']);detail.update(method='path_triangle',triangle=info,**acceptance)
        detail['seconds']+=time.perf_counter()-tic
        packet.update(arrays=a,detail=detail);audit.phase='solver'
        return torch.from_numpy(selected).to(args[2]),torch.from_numpy(cost).to(args[4]),changed
    return hook


def polished_hook(problem,audit,packet):
    config=ModelConfig('quadratic_anchored8')
    def hook(iteration,origin,proposal,before,after,metric,base,V,step):
        if iteration!=1:return proposal,after,False
        tic=time.perf_counter();audit.phase='intervention';native=problem.native
        P,chart=legacy_q.chart(origin,proposal,metric,base,V,step,'usable',8)
        k=P.shape[1];x0=origin.numpy();r0,J=native.least_squares(x0,True);JP0=J@P
        normalizer=max(np.linalg.norm(r0),1e-15);E1=poly.powers(k,1)
        C1=np.array([r0 if not e.sum() else JP0[:,np.flatnonzero(e)[0]] for e in E1])
        R1=np.linalg.qr(C1.T,mode='r')/normalizer;calls=dict(value=0,jacobian=0)
        linear=legacy_q.search(E1,R1,[np.zeros(k)],calls)[0];zlin=linear['z']
        rlin,_=native.least_squares(x0+P@zlin,False)
        E,nodes,A=quadratic_design(k);rr=[];jj=[]
        for z in nodes:
            a,b=native.least_squares(x0+P@z,True);rr.append(a);jj.append(b@P)
        rr=np.array(rr);jj=np.array(jj);observations=np.concatenate([np.vstack([a,b.T]) for a,b in zip(rr,jj)])
        C,_,rank,sv=np.linalg.lstsq(A,observations,rcond=None)
        if rank!=len(E):raise ValueError('Historical quadratic design lost rank')
        R=np.linalg.qr(C.T,mode='r');rng=np.random.default_rng(92495300+k)
        starts=np.vstack([np.zeros(k),zlin,rng.uniform(-1,1,(14,k))])
        internal=legacy_q.search(E,R/normalizer,starts,calls);chosen=[]
        for row in internal:
            if all(np.linalg.norm(row['z']-z)>1e-5 for z in chosen):chosen.append(row['z'])
            if len(chosen)==3:break
        while len(chosen)<3:chosen.append(zlin)
        states=[x0,x0+P@zlin]+[x0+P@z for z in nodes]
        residuals=[r0,rlin]+list(rr);labels=['origin','linear']+['probe']*len(nodes)
        for z in chosen:
            x=x0+P@z;states.append(x);residuals.append(native.least_squares(x,False)[0]);labels.append('legacy_quadratic')
        polishing=[];newcoords=[]
        for z in chosen:
            polished,info=coordinate_polish(z,E,R/normalizer);polishing.append(info)
            if np.linalg.norm(polished-z)>1e-8 and all(np.linalg.norm(polished-x)>1e-8 for x in newcoords):
                x=x0+P@polished;states.append(x);residuals.append(native.least_squares(x,False)[0]);labels.append('polished_quadratic');newcoords.append(polished)
        selected,cost,changed,acceptance=finish(native,proposal,after,states,residuals,labels,config)
        center=poly.basis(np.zeros(k),E)@C;centerJ=C.T@poly.dbasis(np.zeros(k),E)
        packet.update(arrays=dict(origin=x0,proposal=proposal.numpy(),P=P,r0=r0,JP0=JP0,E=E,nodes=nodes,A=A,
            observations=observations,C=C,R=R,rr=rr,JP=jj,zlin=zlin,rlin=rlin,legacy_coordinates=np.array(chosen),
            selected=selected,selected_cost=cost,candidate_states=np.array(states),candidate_residuals=np.array(residuals)),
            detail=dict(method='quadratic_polished8',chart=chart,fit=dict(rank=int(rank),condition=float(sv[0]/sv[-1]),
                relative_fit=float(np.linalg.norm(A@C-observations)/max(np.linalg.norm(observations),1e-30)),
                center_value_error=float(np.linalg.norm(center-r0)/normalizer),
                center_J_error=float(np.linalg.norm(centerJ-JP0)/max(np.linalg.norm(JP0),1e-30))),
                polishing=polishing,internal_status=[x['status'] for x in internal],
                surrogate_calls=calls,**acceptance,seconds=time.perf_counter()-tic))
        audit.phase='solver'
        return torch.from_numpy(selected).to(proposal),torch.from_numpy(cost).to(after),changed
    return hook


def solve(problem,network,engine,outer_config,method):
    if method not in METHODS:raise ValueError(method)
    if outer_config.steps!=5:raise ValueError('Only registered five-step execution')
    native=problem.native;original=native.least_squares;cache={};hits=0
    def cached(x,jacobian=True):
        nonlocal hits
        key=np.asarray(x,dtype=np.float64).tobytes()
        if key in cache and (not jacobian or cache[key][1] is not None):
            hits+=1;return cache[key] if jacobian else (cache[key][0],None)
        value=original(x,jacobian);cache[key]=value;return value
    native.least_squares=cached;audit=PhysicalAudit(native);packet={};failures=[]
    hook=triangle_hook(problem,audit,packet) if method=='path_triangle' else polished_hook(problem,audit,packet)
    def guarded(*args):
        try:return hook(*args)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures.append(dict(kind=type(exc).__name__,message=str(exc),iteration=int(args[0])+1))
            packet.clear();audit.phase='solver';return args[2],args[4],False
    try:
        result=backbone.solve(problem,network,engine,curve_mode='norm_mean_envelope',config=outer_config,hook=guarded)
        result.update(physical_counts=audit.counts(),cache_hits=hits,refinement_failures=failures)
    finally:
        audit.close();native.least_squares=original
    return result,packet
