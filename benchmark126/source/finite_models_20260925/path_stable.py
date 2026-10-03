"""Stable completion: compact nonlinear evaluation, original linear trajectory."""
import time
import numpy as np
import torch
from .core import ModelConfig, CompactAnchor
from .hooks import finish, legacy_p
from . import backbone
from run_fiber_system_20260924 import PhysicalAudit

def path_hook(config, problem, audit, packet):
    def hook(iteration, origin, proposal, before, after, metric, base, Vnet, step):
        if iteration != config.intervention_step-1:
            return proposal, after, False
        tic = time.perf_counter(); audit.phase = 'intervention'; native = problem.native
        xp = proposal.numpy(); n = len(xp)
        radius = max(float(metric.apply(proposal-origin, inverse=False).norm()),
                     .1*float(metric.apply(step,inverse=False).norm()), 1e-8)
        T = metric.apply(torch.eye(n,dtype=torch.float64)).numpy()*radius
        rp, Jp = native.least_squares(xp, True)
        normalizer = max(np.linalg.norm(rp),1e-15)
        rr, J = rp/normalizer, np.asarray(Jp@T)/normalizer
        mu = .001*max(float(np.mean(np.sum(J*J,axis=0))),1e-30)
        gn = -np.linalg.solve(J.T@J+mu*np.eye(n), J.T@rr)
        gn *= min(1.,1/max(np.linalg.norm(gn),1e-30))
        nodes, mask, probe = legacy_p.aligned.probes(J, gn)
        jacs = []
        for s in nodes:
            _, j = native.least_squares(xp+T@s, True)
            jacs.append(np.asarray(j@T)/normalizer)
        pairs, full, groups, recovery = legacy_p.c.recover(J,nodes,np.array(jacs),mask)
        _, C, D, comp = legacy_p.anchor.compress(full,pairs,rr,J,gn)
        if np.any(full):
            model = CompactAnchor(rr,J,C,D,pairs,mu)
            seed, seedsolve = model.solve(gn, config.path_iterations)
        else:
            # Preserve the historically evaluated finite-iteration linear route.
            # The exact ball replacement is kept as an ablation, not silently adopted.
            seed, seedsolve = legacy_p.c.TensorModel(rr,J,pairs,full,mu).solve(gn)
        delta, linesolve = legacy_p.line_solve(rr,J,full,pairs,mu,gn,seed)
        steps = [delta,.5*delta,.25*delta,gn,.5*gn,.25*gn,seed,.5*seed,.25*seed]
        labels = ['path','path_half','path_quarter','GN','GN_half','GN_quarter','anchor','anchor_half','anchor_quarter']
        states = [xp+T@s for s in steps]
        residuals = [native.least_squares(x,False)[0] for x in states]
        witness = dict(queried=False, repaired=False, reason='control_arm')
        hermite = None
        v = seed-gn
        if config.method == 'path_verified':
            if np.linalg.norm(v) <= 1e-8 or np.linalg.norm(full) <= 1e-12*max(np.linalg.norm(J),1e-30):
                witness['reason'] = 'degenerate_or_observed_linear'
            else:
                mid = .5*(gn+seed)
                rm, _ = native.least_squares(xp+T@mid,False)
                states.append(xp+T@mid); residuals.append(rm); labels.append('witness_midpoint')
                model_res = lambda s:rr+J@s+full@legacy_p.c.monomials(s,pairs)
                endpoint_errors = [np.linalg.norm(residuals[i]/normalizer-model_res(s)) for i,s in [(3,gn),(6,seed)]]
                mid_error = float(np.linalg.norm(rm/normalizer-model_res(mid)))
                witness = dict(queried=True,repaired=False,model_endpoint_errors=list(map(float,endpoint_errors)),
                               model_midpoint_error=mid_error,reason='model_consistent')
                if max(endpoint_errors+[mid_error]) > config.path_witness_tolerance:
                    r0,G0 = native.least_squares(xp+T@gn,True)
                    r1,G1 = native.least_squares(xp+T@seed,True)
                    h = hermite_coefficients(r0/normalizer,r1/normalizer,G0@(T@v)/normalizer,G1@(T@v)/normalizer)
                    hmid = np.array([1,.5,.25,.125])@h
                    herror = float(np.linalg.norm(hmid-rm/normalizer))
                    witness.update(hermite_midpoint_error=herror,reason='hermite_witness_rejected')
                    # Same withheld midpoint for both models. It never enters the fit.
                    if herror <= mid_error+config.path_witness_tolerance:
                        coeff = squared_polynomial(h)
                        coeff[:3] += .5*mu*np.array([gn@gn,2*gn@v,v@v])
                        t,sol = polynomial_segment(coeff)
                        z = gn+t*v
                        for scale in [1.,.5,.25]:
                            x = xp+T@(scale*z)
                            states.append(x); residuals.append(native.least_squares(x,False)[0]); labels.append('physical_hermite')
                        witness.update(repaired=True,reason='hermite_witness_passed',t=t,solve=sol)
                        hermite = h
        selected,checked,changed,acceptance = finish(native,proposal,after,states,residuals,labels,config)
        arrays = dict(origin=origin.numpy(),proposal=xp,selected=selected,selected_cost=checked,T=T,
                      rr=rr,J=J,mu=mu,pairs=pairs,full=full,C=C,D=D,gn=gn,seed=seed,delta=delta,
                      nodes=nodes,mask=mask,query_jacobians=jacs,normalizer=normalizer,
                      candidate_states=np.array(states),candidate_residuals=np.array(residuals))
        if hermite is not None: arrays['hermite']=hermite
        packet.update(arrays=arrays,detail=dict(method=config.method,n=n,radius=radius,probe=probe,
            recovery=recovery,compression=comp,anchor_solve=seedsolve,line_solve=linesolve,witness=witness,
            **acceptance,seconds=time.perf_counter()-tic))
        audit.phase='solver'
        return torch.from_numpy(selected).to(proposal),torch.from_numpy(checked).to(after),changed
    return hook



def solve(problem,network,engine,outer_config,method='path_line_complete'):
    if method!='path_line_complete' or outer_config.steps!=5:raise ValueError('Registered radial five-step route only')
    native=problem.native;original=native.least_squares;cache={};hits=0
    def cached(x,jacobian=True):
        nonlocal hits
        key=np.asarray(x,dtype=np.float64).tobytes()
        if key in cache and (not jacobian or cache[key][1] is not None):
            hits+=1;return cache[key] if jacobian else (cache[key][0],None)
        value=original(x,jacobian);cache[key]=value;return value
    native.least_squares=cached;audit=PhysicalAudit(native);packet={};failures=[];hook=path_hook(ModelConfig('path_compact'),problem,audit,packet)
    def guarded(*args):
        try:return hook(*args)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures.append(dict(kind=type(exc).__name__,message=str(exc),iteration=int(args[0])+1));packet.clear();audit.phase='solver';return args[2],args[4],False
    try:
        result=backbone.solve(problem,network,engine,curve_mode='norm_mean_envelope',config=outer_config,hook=guarded)
        result.update(physical_counts=audit.counts(),cache_hits=hits,refinement_failures=failures)
        if packet:packet['detail']['method']='path_line_complete'
    finally:audit.close();native.least_squares=original
    return result,packet
