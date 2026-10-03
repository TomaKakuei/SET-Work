"""One registered intervention; inputs contain observations, never scores/truth."""
import time
import numpy as np
import torch
import optimize_dimension_utility_20260924 as legacy_q
import path_dictionary_20260925 as legacy_p
from .core import (CompactAnchor, anchored_fit, hermite_coefficients, polynomial_segment,
                   squared_polynomial, polynomial_search)

poly = legacy_q.d.poly


def quadratic_design(k):
    # Exactly the historical quadratic design, so fit changes use matched queries.
    _, nodes, _, _ = legacy_q.design(k)
    nodes = nodes[:k+1]
    E = poly.powers(k, 2)
    return E, nodes, poly.design(nodes, E)


def finish(native, proposal, after, states, residuals, labels, config):
    """Finite best observed candidate, strict incumbent protection, no labels."""
    costs = [float(after.mean())]+[.5*float(r@r) if np.isfinite(r).all() else float('inf') for r in residuals]
    index = int(np.argmin(costs))
    changed = index > 0 and costs[index] < costs[0]-config.acceptance_relative*max(abs(costs[0]),1e-30)
    index = index if changed else 0
    selected = proposal.numpy() if not index else states[index-1]
    checked = np.array([float(native.cost(torch.from_numpy(selected), v)) for v in (0,1)])
    if not np.isfinite(checked).all() or not np.isclose(checked.mean(), costs[index], rtol=1e-9, atol=1e-12):
        raise ValueError('Residual and two-view objective contracts disagree')
    return selected, checked, bool(changed), dict(selected_index=index,
        selected_source='incumbent' if index==0 else labels[index-1],
        original_cost=costs[0], selected_cost=float(checked.mean()), candidate_costs=costs,
        changed=bool(changed))


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
        model = CompactAnchor(rr,J,C,D,pairs,mu)
        seed, seedsolve = model.solve(gn, config.path_iterations)
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


def quadratic_hook(config,problem,audit,packet):
    def hook(iteration,origin,proposal,before,after,metric,base,V,step):
        if iteration != config.intervention_step-1:
            return proposal,after,False
        tic=time.perf_counter();audit.phase='intervention';native=problem.native
        P,chart=legacy_q.chart(origin,proposal,metric,base,V,step,'usable',config.directions)
        k=P.shape[1];x0=origin.numpy();r0,J=native.least_squares(x0,True);JP0=J@P
        normalizer=max(np.linalg.norm(r0),1e-15)
        E1=poly.powers(k,1)
        C1=np.array([r0 if not e.sum() else JP0[:,np.flatnonzero(e)[0]] for e in E1])
        R1=np.linalg.qr(C1.T,mode='r')/normalizer
        lin,lin_calls=polynomial_search(E1,R1,[np.zeros(k)],poly.basis,poly.dbasis,
                                        iterations=config.quadratic_iterations)
        zlin=lin[0]['z'];rlin,_=native.least_squares(x0+P@zlin,False)
        E,nodes,A=quadratic_design(k);rr=[];jj=[]
        for z in nodes:
            a,b=native.least_squares(x0+P@z,True);rr.append(a);jj.append(b@P)
        rr=np.array(rr);jj=np.array(jj)
        observations=np.concatenate([np.vstack([a,b.T]) for a,b in zip(rr,jj)])
        C,fit=anchored_fit(E,A,observations,r0,JP0)
        linear=E.sum(1)<=1
        predicted=poly.basis(zlin,E)@C-r0-JP0@zlin
        observed=rlin-r0-JP0@zlin
        alpha=1.
        if config.method=='quadratic_verified8' and np.linalg.norm(predicted)>1e-12*normalizer:
            alpha=float(np.clip((observed@predicted)/(predicted@predicted),0.,1.))
            C[~linear]*=alpha
        R=np.linalg.qr(C.T,mode='r')/normalizer
        rng=np.random.default_rng(92495300+k)
        starts=np.vstack([np.zeros(k),zlin,rng.uniform(-1,1,(config.quadratic_starts-2,k))])
        internal,calls=polynomial_search(E,R,starts,poly.basis,poly.dbasis,
                iterations=config.quadratic_iterations,polish=config.method=='quadratic_verified8')
        chosen=[]
        for row in internal:
            if all(np.linalg.norm(row['z']-z)>1e-5 for z in chosen): chosen.append(row['z'])
            if len(chosen)==3: break
        states=[x0,x0+P@zlin]+[x0+P@z for z in nodes]
        residuals=[r0,rlin]+list(rr);labels=['origin','linear']+['probe']*len(nodes)
        for z in chosen:
            x=x0+P@z;states.append(x);residuals.append(native.least_squares(x,False)[0]);labels.append('quadratic')
        selected,checked,changed,acceptance=finish(native,proposal,after,states,residuals,labels,config)
        changes=(jj-JP0).reshape(-1,k);sv=np.linalg.svd(changes,compute_uv=False)
        nonlinear_rank=int(sum(sv>1e-9*max(float(np.linalg.norm(jj)),1e-30)))
        nonlinear95=0 if not nonlinear_rank else int(np.searchsorted(np.cumsum(sv*sv),.95*sum(sv*sv)))+1
        packet.update(arrays=dict(origin=x0,proposal=proposal.numpy(),P=P,r0=r0,JP0=JP0,
            E=E,nodes=nodes,A=A,observations=observations,C=C,R=R,rr=rr,JP=jj,zlin=zlin,rlin=rlin,
            selected=selected,selected_cost=checked,candidate_states=np.array(states),
            candidate_residuals=np.array(residuals),nonlinear_singular_values=sv),
            detail=dict(method=config.method,chart=chart,fit=fit,nonlinearity_scale=alpha,
                witness_relative_error=float(np.linalg.norm(observed-alpha*predicted)/normalizer),
                nonlinear_rank=nonlinear_rank,nonlinear95=nonlinear95,
                internal_status=[x['status'] for x in internal],
                projected_gradient_norms=[x['projected_gradient_norm'] for x in internal],
                surrogate_calls=calls,linear_calls=lin_calls,**acceptance,seconds=time.perf_counter()-tic))
        audit.phase='solver'
        return torch.from_numpy(selected).to(proposal),torch.from_numpy(checked).to(after),changed
    return hook


def make_hook(config, problem, audit, packet):
    return path_hook(config,problem,audit,packet) if config.method.startswith('path_') else quadratic_hook(config,problem,audit,packet)
