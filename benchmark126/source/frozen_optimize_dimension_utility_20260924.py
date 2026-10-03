"""Finite-map dimension screen and residual-defect-gated compiler."""
import json,time,math,traceback
from dataclasses import replace
from functools import lru_cache
from scipy.optimize import minimize
import dimension_utility_20260924 as u
d,r,np,torch,OUT=u.d,u.r,u.np,u.torch,u.OUT
ARMS=['linear_raw4','linear_raw8','cubic_raw1','cubic_raw2','cubic_raw4','cubic_raw8','linear_usable4','linear_usable8','cubic_usable4','cubic_usable8','adaptive_usable8']

@lru_cache(None)
def design(k):
    E=d.poly.powers(k,3);m={1:2,2:4,3:6,4:8,5:11,6:13,7:16,8:20}[k]
    rng=np.random.default_rng(92495200+k);best=None
    for _ in range(16):
        nodes=np.vstack([np.zeros(k),rng.uniform(-1,1,(m-1,k))]);norm=np.linalg.norm(nodes,axis=1)
        nodes=nodes/np.maximum(1,norm[:,None]/2)
        A=d.poly.design(nodes,E);sv=np.linalg.svd(A,compute_uv=False);quality=sv[-1]/sv[0]
        if best is None or quality>best[0]:best=quality,nodes,A
    quality,nodes,A=best;assert np.linalg.matrix_rank(A)==len(E)
    return E,nodes,A,float(quality)

def chart(origin,proposal,metric,base,V,step,kind,k):
    white=metric.apply(proposal-origin,inverse=False);raw=metric.apply(step,inverse=False)
    radius=max(float(white.norm()),.1*float(raw.norm()),1e-8)
    if kind=='usable':
        Z=u.projected(base,V);U,S,sp=u.spectrum(Z,V.norm());directions=U[:,:sp['rank']]
    else:directions=V
    # Q fills only missing ambient rank; no data/GT-dependent task switch.
    candidates=[white]+list(directions.unbind(1))+list(base['Q'].unbind(1));columns=[]
    for v in candidates:
        q=v.clone();ref=float(q.norm())
        if ref<1e-20:continue
        for _ in range(2):
            if columns:
                W=torch.stack(columns,1);q=q-W@(W.T@q)
        if float(q.norm())>1e-9*ref:columns.append(q/q.norm())
        if len(columns)==k:break
    W=torch.stack(columns,1);P=metric.apply(W)*radius
    return P.numpy(),dict(requested=k,actual=W.shape[1],radius=radius,source=kind,orthogonality_error=float((W.T@W-torch.eye(W.shape[1])).abs().max()))

def search(E,R,starts,calls):
    k=E.shape[1];rows=[]
    def fun(z):
        calls['value']+=1;res=R@d.poly.basis(z,E);return float(res@res/2)
    def jac(z):
        calls['jacobian']+=1;res=R@d.poly.basis(z,E);return (R@d.poly.dbasis(z,E)).T@res
    for start in starts:
        z=np.clip(start,-1,1);z=z/max(1,np.linalg.norm(z)/2)
        result=minimize(fun,z,jac=jac,method='SLSQP',bounds=[(-1.,1.)]*k,
            constraints=[dict(type='ineq',fun=lambda z:4-z@z,jac=lambda z:-2*z)],options=dict(maxiter=50,ftol=1e-11))
        z=np.clip(result.x,-1,1);z=z/max(1,np.linalg.norm(z)/2)
        rows.append(dict(z=z,cost=fun(z),status=int(result.status),iterations=int(result.nit)))
    return sorted(rows,key=lambda q:q['cost'])

def make_hook(arm,problem,audit,packet):
    mode,kind=arm.split('_');k=int(kind[-1]);kind=kind[:-1]
    def hook(iteration,origin,proposal,before,after,metric,base,V,step):
        if iteration!=1:return proposal,after,False
        started=time.perf_counter();audit.phase='intervention';P,info=chart(origin,proposal,metric,base,V,step,kind,k)
        n=P.shape[1];x0=origin.numpy();native=problem.native;rr0,J=native.least_squares(x0,True);JP0=J@P
        normalizer=max(np.linalg.norm(rr0),1e-15);linearE=d.poly.powers(n,1)
        linearC=np.array([rr0 if not e.sum() else JP0[:,np.flatnonzero(e)[0]] for e in linearE]);linearR=np.linalg.qr(linearC.T,mode='r')/normalizer
        calls=dict(value=0,jacobian=0);linear=search(linearE,linearR,[np.zeros(n)],calls)[0];zlin=linear['z']
        rlin,_=native.least_squares(x0+P@zlin,False)
        defect=float(np.linalg.norm(rlin-rr0-JP0@zlin)/normalizer)
        gate=mode=='cubic' or (mode=='adaptive' and defect>.01)
        states=[proposal.numpy(),x0,x0+P@zlin];values=[float(after.mean()),float(before.mean()),float(rlin@rlin/2)];labels=['incumbent','origin','linear']
        arrays=dict(origin=x0,proposal=proposal.numpy(),P=P,r0=rr0,JP0=JP0,linear=zlin,linear_residual=rlin)
        nonlinear95=0;nonlinearity=0.;fitdefect=None;internal=[];polytests=[];truepoly=[]
        if gate:
            E,nodes,A,geometry=design(n);rr=[];jj=[]
            for z in nodes:
                residual,J=native.least_squares(x0+P@z,True);rr.append(residual);jj.append(J@P)
            rr=np.array(rr);jj=np.array(jj);B=np.concatenate([np.vstack([q,j.T]) for q,j in zip(rr,jj)])
            C=np.linalg.lstsq(A,B,rcond=None)[0];R=np.linalg.qr(C.T,mode='r');fitdefect=float(np.linalg.norm(A@C-B)/max(np.linalg.norm(B),1e-30))
            changes=(jj-JP0).reshape(-1,n);sv=np.linalg.svd(changes,compute_uv=False);nonlinearity=float(np.linalg.norm(changes)/max(np.linalg.norm(jj),1e-30))
            if nonlinearity>1e-8:nonlinear95=int(np.searchsorted(np.cumsum(sv*sv),.95*np.sum(sv*sv)))+1
            rng=np.random.default_rng(92495300+n);starts=np.vstack([np.zeros(n),zlin,rng.uniform(-1,1,(14,n))])
            internal=search(E,R/normalizer,starts,calls);chosen=[]
            for row in internal:
                if all(np.linalg.norm(row['z']-z)>1e-5 for z in chosen):chosen.append(row['z'])
                if len(chosen)==3:break
            while len(chosen)<3:chosen.append(zlin)
            states += [x0+P@z for z in nodes];values += [float(q@q/2) for q in rr];labels += ['probe']*len(nodes)
            for z in chosen:
                x=x0+P@z;value=float(native.objective(x));states.append(x);values.append(value);labels.append('polynomial');polytests.append(z);truepoly.append(value)
            arrays.update(E=E,nodes=nodes,A=A,B=B,C=C,R=R,rr=rr,JP=jj,polytests=polytests,truepoly=truepoly,nonlinear_singular_values=sv)
        idx=int(np.argmin(values));changed=values[idx]<values[0]-1e-12*max(abs(values[0]),1e-30);idx=idx if changed else 0;x=states[idx]
        checked=np.array([float(native.cost(torch.from_numpy(x),v)) for v in (0,1)])
        assert np.isclose(checked.mean(),values[idx],rtol=1e-9,atol=1e-12)
        assert checked.mean()<=values[0]+1e-10*max(abs(values[0]),1e-12)
        arrays.update(selected=x,selected_cost=checked,candidate_states=states,candidate_values=values)
        packet.update(arrays=arrays,detail=dict(chart=info,gate=gate,linear_defect=defect,nonlinear95=nonlinear95,nonlinearity=nonlinearity,fit_defect=fitdefect,changed=bool(changed),original_cost=values[0],selected_cost=float(checked.mean()),selected_source=labels[idx],selected_index=idx,linear_cost=values[2],best_probe_cost=min([value for value,label in zip(values,labels) if label=='probe'],default=values[0]),surrogate_calls=calls,internal_status=[q['status'] for q in internal],seconds=time.perf_counter()-started))
        audit.phase='solver';return torch.from_numpy(x).to(proposal),torch.from_numpy(checked).to(after),bool(changed)
    return hook

def run():
    torch.set_num_threads(1);r.s.threadpool_limits(1);r.s.cv2.setNumThreads(1)
    path=OUT/'optimization_protocol.json'
    if path.exists():p=r.read(path)
    else:
        p=dict(schema='finite-map-dimension-utility-v1',cases=u.specifications(),arms=ARMS,steps=5,intervention_step=2,training_updates=0,
            registration='After 156 original-architecture ablations; before finite-map optimization outcomes. Development only.',
            chart='Whitened accepted displacement first. raw: original network heads. usable: singular axes of projected valid completion. Fill insufficient rank with measured Q. Use nested prefixes.',
            feasible='Same radius as prior system run. For all k use [-1,1]^k intersect ||z||<=2, so larger k cannot merely increase maximum displacement beyond the old 4D box radius. 1D/2D still have smaller maximum attainable norm due to box; 4D vs8D share radius bound.',
            fit='Full cubic exact-residual Hermite fit. 2/4/8/20 value-J queries for k1/2/4/8; actual rank may be smaller. Center cached from RNC. Dense QR norm-preserving compression.',
            search='Same SLSQP, sixteen starts (zero, linear minimizer, fourteen fixed random), max50 iterations each. Include linear candidate and all probes in physical validation. No global-optimum claim.',
            adaptive='At usable8, solve cached linear chart first. Only fit cubic if exact residual defect at that candidate / ||r0|| exceeds .01. Fixed task-independent threshold before outcomes; uses no truth. One witness can miss curvature elsewhere.',
            accounting='No hard21J cap under new user instruction. cubic8 nominal29 jointJ (ten originalcurve plus nineteen noncenter). Physical J, forwards, internal calls and elapsed all recorded.',
            route_count=12*len(ARMS),solve_seconds_limit=300,protected_hashes=r.read(OUT/'protocol.json')['protected_hashes'],source_sha256=r.sha(__file__),
            designs={str(k):dict(E=design(k)[0],nodes=design(k)[1],singular_ratio=design(k)[3]) for k in [1,2,4,8]})
        u.write('optimization_protocol.json',p)
    r.assert_protected(p);model=r.s.load_model(r.s.CKPT).eval();d.OUT=OUT;solver=d.solver_with_hook()
    path=OUT/'optimization_predictions.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else [];done={(q['key'],q['arm']) for q in rows};elapsed=sum(q['seconds'] for q in rows)
    with d.serial_numerics():
        for spec in p['cases']:
            problem,unused,meta=u.load(spec);del unused
            for arm in ARMS:
                if (spec['key'],arm) in done:continue
                assert elapsed<p['solve_seconds_limit'];native=problem.native;original=native.least_squares;cache={};hits=[0]
                def cached(x,jacobian=True):
                    if jacobian and cache and np.array_equal(x,cache['x']):hits[0]+=1;return cache['value']
                    result=original(x,jacobian)
                    if jacobian:cache.update(x=np.array(x),value=result)
                    return result
                native.least_squares=cached;audit=d.PhysicalAudit(native);packet={};started=time.perf_counter()
                try:result=solver(problem,model,r.CountedEngine(r.engine_for(problem)),curve_mode='norm_mean_envelope',config=replace(r.s.phase.cfg(spec['group']),snapshot_steps=(1,2,3,4,5)),hook=make_hook(arm,problem,audit,packet))
                finally:audit.close();native.least_squares=original
                seconds=time.perf_counter()-started;elapsed+=seconds;name=f"map_{spec['key']}_{arm}.npz";np.savez_compressed(OUT/name,**packet['arrays'])
                row=dict(key=spec['key'],group=spec['group'],arm=arm,theta=np.asarray(result['theta']).tolist(),snapshots=result['snapshots'],trajectory=result['trajectory'],final_cost=result['final_cost'],seconds=seconds,counts=audit.counts(),J_cache_hits=hits[0],intervention=packet['detail'],packet=name,packet_sha256=r.sha(OUT/name))
                with path.open('a',encoding='utf-8') as f:f.write(json.dumps(r.s.safe(row),allow_nan=False)+'\n')
                rows.append(row);done.add((spec['key'],arm))
            u.write('optimization_state.json',dict(status='running',completed=len(rows),expected=p['route_count'],solve_seconds=elapsed))
            print(json.dumps(dict(key=spec['key'],completed=len(rows),solve_seconds=elapsed)),flush=True)
    assert len(rows)==p['route_count'];u.write('optimization_freeze.json',dict(rows=len(rows),sha256=r.sha(path)));scores=[]
    for spec in p['cases']:
        problem,e,meta=u.load(spec)
        for row in [q for q in rows if q['key']==spec['key']]:
            x=torch.tensor(row['theta'],dtype=problem.initial.dtype);scores.append(dict(key=row['key'],arm=row['arm'],metric=float(e(x)['metric']),objective=float(problem.native.objective(x)),J=row['counts']['actual_joint_J'],F=row['counts']['joint_forward_only'],seconds=row['seconds']))
    u.write('optimization_scores.json',scores);r.assert_protected(p);u.write('optimization_state.json',dict(status='computed_unreviewed',completed=len(rows),background=False,solve_seconds=elapsed))
    for spec in p['cases']:print(spec['key'],json.dumps({q['arm']:q['metric'] for q in scores if q['key']==spec['key']}))

if __name__=='__main__':
    try:run()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'optimization_failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
