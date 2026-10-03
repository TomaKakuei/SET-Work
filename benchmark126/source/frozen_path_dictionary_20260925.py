"""Match a full-input quadratic residual along a planned GN-to-anchor chord."""
import time
import run_dictionary_matching_20260924 as old
import aligned_dictionary_20260924 as aligned
import anchor_dictionary_20260924 as anchor
c,np,r,d,torch=old.c,old.np,old.r,old.d,old.torch

def basis(vectors):
    columns=[]
    for a in vectors:
        norm=np.linalg.norm(a)
        if norm<1e-25:continue
        w=a/norm
        for _ in range(2):
            if columns:Q=np.column_stack(columns);w-=Q@(Q.T@w)
        if np.linalg.norm(w)>1e-11:columns.append(w/np.linalg.norm(w))
    return np.column_stack(columns) if columns else np.zeros((len(vectors[0]),0))

def compress(B,pairs,rr,J,start,end,keep_gradient=True):
    v=end-start;d0=B@c.monomials(start,pairs);d1=B@c.monomial_jac(start,pairs)@v;d2=B@c.monomials(v,pairs)
    Q=basis([d0,d1,d2]+([rr+J@start+d0] if keep_gradient else []));D=Q.T@B;new=Q@D
    return new,Q,D,dict(output_rank=Q.shape[1],scope='Exact identified-quadratic residual on the chosen affine line; no physical or off-line guarantee',preserves_anchor_objective_gradient=keep_gradient)

def line_solve(rr,J,B,pairs,mu,start,end):
    v=end-start;a=rr+J@start+B@c.monomials(start,pairs);b=J@v+B@c.monomial_jac(start,pairs)@v;cc=B@c.monomials(v,pairs)
    coefficients=np.zeros(5)
    for i,x in enumerate([a,b,cc]):
        for j,y in enumerate([a,b,cc]):coefficients[i+j]+=.5*(x@y)
    coefficients[:3]+=.5*mu*np.array([start@start,2*start@v,v@v]);derivative=coefficients[1:]*np.arange(1,5)
    candidates=[0.,1.]
    if np.linalg.norm(derivative)>1e-30:
        roots=np.polynomial.polynomial.polyroots(derivative/max(abs(derivative)))
        candidates += [float(np.clip(x.real,0,1)) for x in roots if abs(x.imag)<1e-8 and -1e-10<=x.real<=1+1e-10]
    values=np.polynomial.polynomial.polyval(candidates,coefficients);idx=int(np.argmin(values));t=candidates[idx]
    return start+t*v,dict(status='quartic_stationary_roots_on_closed_segment',t=t,candidates=candidates,values=values.tolist(),polynomial=coefficients.tolist(),global_certificate='Exact polynomial segment only, floating-point roots')

def algebra():
    rng=np.random.default_rng(92501101);line_errors=[];grad_errors=[];min_errors=[]
    for _ in range(32):
        n,m=13,41;pairs=np.array(c.pairs_for(n));B=rng.normal(size=(m,len(pairs)));rr=rng.normal(size=m);J=rng.normal(size=(m,n));s=rng.normal(size=n);s*=.7/np.linalg.norm(s);end=rng.normal(size=n);end*=.9/np.linalg.norm(end)
        new,Q,D,info=compress(B,pairs,rr,J,s,end)
        for t in [-.4,0,.17,.5,.87,1,1.4]:
            x=s+t*(end-s);line_errors.append(float(np.linalg.norm((new-B)@c.monomials(x,pairs))/max(np.linalg.norm(B@c.monomials(x,pairs)),1e-30)))
        res=rr+J@s+B@c.monomials(s,pairs);grad=(J+B@c.monomial_jac(s,pairs)).T@res;gp=(J+new@c.monomial_jac(s,pairs)).T@res;grad_errors.append(float(np.linalg.norm(grad-gp)/max(np.linalg.norm(grad),1e-30)))
        delta,sol=line_solve(rr,J,B,pairs,.03,s,end);cost=lambda x:.5*(np.linalg.norm(rr+J@x+B@c.monomials(x,pairs))**2+.03*(x@x));grid=min(cost(s+t*(end-s)) for t in np.linspace(0,1,301));min_errors.append(max(0.,cost(delta)-grid));assert info['output_rank']<=4
    assert max(line_errors)<1e-12 and max(grad_errors)<1e-12 and max(min_errors)<1e-10
    return dict(status='passed',models=32,line_points=224,max_line_relative_error=max(line_errors),max_anchor_gradient_relative_error=max(grad_errors),max_excess_over_dense_segment_grid=max(min_errors),scope='Algebra and numerical roots, not physical residual accuracy')

def hook_for(arm,problem,audit,packet,protocol):
    def hook(iteration,origin,proposal,before,after,metric,base,Vnet,step):
        if iteration!=1:return proposal,after,False
        tic=time.perf_counter();audit.phase='intervention';native=problem.native;xp=proposal.numpy();n=len(xp)
        radius=max(float(metric.apply(proposal-origin,inverse=False).norm()),.1*float(metric.apply(step,inverse=False).norm()),1e-8);T=metric.apply(torch.eye(n,dtype=torch.float64)).numpy()*radius
        rp,Jp=native.least_squares(xp,True);norm=max(np.linalg.norm(rp),1e-15);rr=rp/norm;J=np.asarray(Jp@T)/norm;mu=.001*max(float(np.mean(np.sum(J*J,axis=0))),1e-30)
        gn=-np.linalg.solve(J.T@J+mu*np.eye(n),J.T@rr);gn*=min(1.,1/max(np.linalg.norm(gn),1e-30));nodes,mask,probe=aligned.probes(J,gn);values=[];jacs=[]
        for s in nodes:
            v,j=native.least_squares(xp+T@s,True);values.append(v/norm);jacs.append(np.asarray(j@T)/norm)
        pairs,full,groups,recovery=c.recover(J,nodes,np.array(jacs),mask);B2,C2,D2,anchor_info=anchor.compress(full,pairs,rr,J,gn);seed,seedsolve=c.TensorModel(rr,J,pairs,B2,mu).solve(gn)
        if arm=='path_anchor2':B,C,D,comp=B2,C2,D2,anchor_info;delta=seed;solve=seedsolve
        else:
            B,C,D,comp=compress(full,pairs,rr,J,gn,seed,keep_gradient=arm=='path4')
            if arm=='path4':delta,solve=c.TensorModel(rr,J,pairs,B,mu).solve(gn)
            else:delta,solve=line_solve(rr,J,full,pairs,mu,gn,seed)
        steps=np.array([delta,.5*delta,.25*delta,gn,.5*gn,.25*gn,seed,.5*seed,.25*seed]);states=np.array([xp+T@s for s in steps]);costs=[float(after.mean())]+[float(native.objective(x)) for x in states]
        idx=int(np.argmin(costs));changed=costs[idx]<costs[0]-1e-12*max(abs(costs[0]),1e-30);idx=idx if changed else 0;selected=xp if idx==0 else states[idx-1];checked=np.array([float(native.cost(torch.from_numpy(selected),v)) for v in [0,1]]);assert np.isclose(checked.mean(),costs[idx],rtol=1e-9,atol=1e-12)
        packet.update(arrays=dict(origin=origin.numpy(),proposal=xp,selected=selected,selected_cost=checked,T=T,J=J,rr=rr,normalizer=norm,mu=mu,pairs=pairs,full=full,B=B,C=C,D=D,groups=groups,nodes=nodes,mask=mask,delta=delta,gn=gn,seed=seed,white_steps=steps,test_states=states,true_test=costs[1:],query_values=values,query_jacobians=jacs),detail=dict(arm=arm,n=n,radius=radius,probe=probe,recovery=recovery,compression=comp,anchor_seed_solve=seedsolve,solve=solve,changed=bool(changed),original_cost=costs[0],selected_cost=float(checked.mean()),selected_index=idx,selected_source='incumbent' if idx==0 else 'path_candidate' if idx<=3 else 'GN_scaled' if idx<=6 else 'anchor_seed_scaled',seconds=time.perf_counter()-tic))
        audit.phase='solver';return torch.from_numpy(selected).to(proposal),torch.from_numpy(checked).to(after),bool(changed)
    return hook
