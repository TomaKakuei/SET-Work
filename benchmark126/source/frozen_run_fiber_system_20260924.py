"""Five-step integration screen: frozen shared SETSUNET + bounded cubic chart search."""
from pathlib import Path
import sys,os,json,time,types,pathlib,hashlib,traceback,itertools
sys.modules.setdefault('pathlib._local',pathlib)
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'tools'))
import run_architecture_strengthening_20260919 as r
import run_inverse_fiber_round2_20260924 as poly
from dataclasses import replace
from scipy.optimize import least_squares
from research_serial_lock_20260919 import serial_numerics
np,torch=r.np,r.torch
OUT=r.ROOT/'results_stage9_dev/runs/fiber_system_20260924'
ARMS=['setsunet_original','setsunet_curve','lm','probe_control','fiber_learned','fiber_measured']

def write(name,x):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/name;t=p.with_suffix(p.suffix+'.pending')
    t.write_text(json.dumps(r.s.safe(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def rows():return [json.loads(x) for x in (OUT/'predictions.jsonl').read_text().splitlines()] if (OUT/'predictions.jsonl').exists() else []

def register():
    if (OUT/'protocol.json').exists():return r.read(OUT/'protocol.json')
    specs=[]
    for i,n in enumerate([32,128]):specs.append(dict(key=f'fiber_broyden_{n}',group='broyden',spec=dict(family='broyden',n=n,seed=92491001+i)))
    for i,condition in enumerate(['noisy_view','biased_view']):specs.append(dict(key='fiber_graph_'+condition,group='view_graph_'+condition,spec=dict(family='view_graph',n=48,condition=condition,seed=92492001+i)))
    for i,nodes in enumerate([4,6]):specs.append(dict(key=f'fiber_se3_{nodes}',group='se3',spec=dict(family='se3',nodes=nodes,seed=92493001+i)))
    registry={s['key']:s for s in r.read(r.ROOT/'results_stage9/task_registry.json')['cases']}
    specs += [dict(key=k,group='tum',registry=True,record=registry[k]) for k in ['case_0737','case_0757']]
    for start in [0,1]:specs.append(dict(key=f'fiber_rat43_start{start}',group='nist_Rat43',spec=dict(family='nist',dataset='Rat43',start=start,seed=92494001,partition='interleave')))
    source=poly.OUT/'protocol.json';nodes=r.read(source)['nodes']['3']
    protected=list(r.PROTECTED)+[r.ROOT/'setsunet_csn/stage5_curve_ablation.py',r.ROOT/'research_writing/iclr_joint_contribution_20260923/main_revision.tex']
    p=dict(schema='frozen-shared-chart-cubic-system-v1',cases=specs,arms=ARMS,outer_steps=5,intervention_outer_step=2,
        max_actual_joint_J=21,expected_J=dict(setsunet_original=5,setsunet_curve=10,lm=5,probe_control=18,fiber_learned=18,fiber_measured=18),
        training_updates=0,shared_checkpoint=str(r.s.CKPT),parameter_count=49096,nodes=nodes,
        chart='Four orthonormal whitened directions: accepted proposal first, then raw existing network V (learned/probe) or measured Q. Reorthogonalize twice; rank<4 falls back without probing. Physical P uses the original block metric.',
        radius='Whitened max(||accepted displacement||,0.1*||raw step||,1e-8); z in [-1,1]^4. Fixed before outcomes. No truth or task-specific radius.',
        model='Full 4D cubic Hermite fit to exact objective-preserving residual and directional Jacobian, eight physical joint J; QR compress residual to <=35 dimensions preserving squared norm.',
        search='16 fixed starts: eight fit nodes and +/- four Hadamard corners; bounded scipy least_squares, max_nfev20 each. Rank/deduplicate candidates and physically validate four, padding origin if needed. This is bounded multistart, not complete fiber enumeration.',
        control='Probe arm uses identical learned chart, same eight value/J probes, and four Hadamard-corner true forwards. Select among probes, validated candidates, origin and original accepted proposal using actual observed mean cost.',
        acceptance='Only override original proposal if actual full objective improves by >1e-12 relative (floor1e-30). This protects current-step objective, not final truth quality. Preserve original damping decision; stored rho pertains to original proposal; update history to selected displacement.',
        data='Six newly seeded synthetic cases; two historical public TUM windows; two published starts on one NIST Rat43 dataset. Mechanism development screen; no universal pass/fail or independent dataset count claim.',
        route_count=60,solve_seconds_limit=480,internal_surrogate_calls_counted=True,truth='Scorers only called after all predictions frozen; no ground-truth values passed to hook.',
        timing='Serial single thread, one solve each. Per-route elapsed is diagnostic, not a repeated performance benchmark.',
        protected_hashes={str(x.relative_to(BASE)):r.sha(x) for x in protected},
        source_hashes={str(Path(__file__).relative_to(BASE)):r.sha(Path(__file__)),str(Path(poly.__file__).relative_to(BASE)):r.sha(Path(poly.__file__))},
        input_hashes={k:r.sha(r.ROOT/'results_stage9/compiled'/f'{k}.pkl.gz') for k in ['case_0737','case_0757']})
    write('protocol.json',p);return p

class PhysicalAudit:
    def __init__(self,native):
        self.native=native;self.events=[];self.phase='solver'
        if hasattr(native,'raw_jac'):
            self.attribute='raw_jac';self.original=native.raw_jac
            def call(x,v,jac):
                out=self.original(x,v,jac);self.events.append(dict(view=v,J=bool(jac),phase=self.phase));return out
        else:
            self.attribute='evaluate';self.original=native.evaluate
            def call(x,v,mode=2,*,jacobian=False):
                out=self.original(x,v,mode,jacobian=jacobian);self.events.append(dict(view=v,J=bool(mode>0 or jacobian),phase=self.phase));return out
        setattr(native,self.attribute,call)
    def close(self):setattr(self.native,self.attribute,self.original)
    def counts(self):
        j=sum(x['J'] for x in self.events);f=len(self.events)-j;assert j%2==f%2==0
        return dict(actual_joint_J=j//2,joint_forward_only=f//2,view_calls=len(self.events),intervention_J=sum(x['J'] and x['phase']=='intervention' for x in self.events)//2,intervention_forward=sum(not x['J'] and x['phase']=='intervention' for x in self.events)//2)

def solver_with_hook():
    source=Path(r.curve.__file__).read_text(encoding='utf-8')
    signature="def solve(problem, model, engine, tangent='csn', curve_mode='straight', config=None):"
    assert source.count(signature)==1;source=source.replace(signature,signature[:-2]+', hook=None):')
    marker='        trajectory.append(dict(\n';assert source.count(marker)==1
    source=source.replace(marker,"""        fiber_changed = False
        if hook is not None:
            x, after, fiber_changed = hook(iteration, origin, x, before, after, metric, base, V, step)
            if fiber_changed:
                history=x-origin
                accepted=True
"""+marker)
    source=source.replace('            iteration=iteration, accepted=accepted, step_scale=alpha,','            iteration=iteration, accepted=accepted, step_scale=alpha, fiber_override=fiber_changed,')
    module=types.ModuleType('setsunet_csn.fiber_system_isolated');module.__package__='setsunet_csn'
    exec(compile(source,'<fiber_system_isolated>','exec'),module.__dict__)
    (OUT/'isolated_solver_source.txt').write_text(source,encoding='utf-8');return module.solve

def whitened_chart(origin,proposal,metric,base,V,step,mode):
    delta=proposal-origin;white=metric.apply(delta,inverse=False);raw=metric.apply(step,inverse=False)
    radius=max(float(white.norm()),.1*float(raw.norm()),1e-8)
    candidates=[white]+list((base['Q'] if mode=='fiber_measured' else V).unbind(1));Q=[]
    for v in candidates:
        u=v.clone();reference=float(u.norm())
        if reference<1e-20:continue
        for _ in range(2):
            if Q:
                q=torch.stack(Q,1);u=u-q@(q.T@u)
        if float(u.norm())>1e-9*reference:Q.append(u/u.norm())
        if len(Q)==4:break
    if len(Q)<4:return None,dict(rank=len(Q),radius=radius)
    W=torch.stack(Q,1);P=metric.apply(W)*radius
    return P.detach().numpy(),dict(rank=4,radius=radius,physical_column_norms=np.linalg.norm(P.numpy(),axis=0).tolist())

HAD=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],float)

def hook_for(arm,problem,audit,p,packet):
    def hook(iteration,origin,proposal,before,after,metric,base,V,step):
        if iteration!=1:return proposal,after,False
        started=time.perf_counter();P,chart=whitened_chart(origin,proposal,metric,base,V,step,arm)
        if P is None:packet.update(detail=dict(chart=chart,skipped='rank below four'));return proposal,after,False
        native=problem.native;audit.phase='intervention';x0=origin.detach().numpy();xp=proposal.detach().numpy();nodes=np.asarray(p['nodes']);E=poly.powers(4,3)
        rr=[];jj=[]
        for z in nodes:
            residual,J=native.least_squares(x0+P@z,True);JP=J@P;rr.append(residual);jj.append(np.asarray(JP))
        rr=np.array(rr);B=np.concatenate([np.vstack([q,j.T]) for q,j in zip(rr,jj)]);A=poly.design(nodes,E)
        C=np.linalg.lstsq(A,B,rcond=None)[0]
        R=np.linalg.qr(C.T,mode='r');normalizer=max(np.linalg.norm(rr[0]),1e-15);Rsmall=R/normalizer
        fit_defect=float(np.linalg.norm(A@C-B)/max(np.linalg.norm(B),1e-30))
        compression_error=float(np.max(abs(np.sum((poly.basis(nodes,E)@C)**2,axis=1)-np.sum((poly.basis(nodes,E)@R.T)**2,axis=1)))/max(np.max(np.sum(rr**2,axis=1)),1e-30))
        assert compression_error<1e-9
        calls=dict(value=0,jacobian=0);internal=[];search_start=time.perf_counter()
        if arm=='probe_control':test=HAD.copy()
        else:
            def fun(z):calls['value']+=1;return Rsmall@poly.basis(z,E)
            def jac(z):calls['jacobian']+=1;return Rsmall@poly.dbasis(z,E)
            for start in np.vstack([nodes,HAD,-HAD]):
                out=least_squares(fun,np.clip(start,-1+1e-10,1-1e-10),jac=jac,bounds=(-1.,1.),max_nfev=20,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                internal.append(dict(z=out.x,cost=float(out.fun@out.fun/2),status=int(out.status),nfev=out.nfev))
            ordered=sorted(internal,key=lambda x:x['cost']);chosen=[]
            for row in ordered:
                if all(np.linalg.norm(row['z']-z)>1e-5 for z in chosen):chosen.append(row['z'])
                if len(chosen)==4:break
            while len(chosen)<4:chosen.append(np.zeros(4))
            test=np.array(chosen)
        surrogate_seconds=time.perf_counter()-search_start
        states=[xp,x0]+[x0+P@z for z in nodes];values=[float(after.mean()),float(before.mean())]+[float(q@q/2) for q in rr]
        candidate_view=[];predicted=[];true=[]
        for z in test:
            x=x0+P@z;view=np.array([float(native.cost(torch.from_numpy(x),v)) for v in [0,1]])
            value=float(view.mean());states.append(x);values.append(value);candidate_view.append(view)
            model=poly.basis(z,E)@C;predicted.append(float(model@model/2));true.append(value)
        # The selected point is explicitly validated even when its probe residual was cached.
        idx=int(np.argmin(values));improved=values[idx]<values[0]-1e-12*max(abs(values[0]),1e-30)
        idx=idx if improved else 0;x=states[idx]
        checked=np.array([float(native.cost(torch.from_numpy(x),v)) for v in [0,1]])
        assert abs(checked.mean()-values[idx])<1e-9*max(abs(values[idx]),1e-12)
        assert checked.mean()<=float(after.mean())+1e-10*max(abs(float(after.mean())),1e-12)
        detail=dict(arm=arm,chart=chart,fit_residual=fit_defect,compression_relative_error=compression_error,
            surrogate_calls=calls,surrogate_seconds=surrogate_seconds,total_seconds=time.perf_counter()-started,
            selected_index=idx,changed=improved,original_cost=float(after.mean()),selected_cost=float(checked.mean()),candidate_costs=values,
            predicted_candidate_cost=predicted,true_candidate_cost=true,internal_terminations=[x['status'] for x in internal],
            objective_contract='0.5 ||exact transformed residual||^2 equals mean original view cost')
        packet.update(detail=detail,arrays=dict(origin=x0,proposal=xp,P=P,nodes=nodes,E=E,B=B,A=A,C=C,R=R,rr=rr,JP=np.array(jj),test=test,selected=x,selected_cost=checked,
            original_cost=after.numpy(),predicted=predicted,truth_cost=true))
        audit.phase='solver';return torch.from_numpy(x).to(proposal),torch.from_numpy(checked).to(after),bool(improved)
    return hook

def load(spec):
    if spec.get('registry'):
        problem,cached,meta=r.s.load_case(spec['record'])
        cells={k:c.cell_contents for k,c in zip(cached.__code__.co_freevars,cached.__closure__)}
        graph=cells['graph'];timestamps=cells['compiled'].node_timestamps
        # No stale pickled scoring function is executed. GT is loaded only when this fresh closure is called.
        def score(x):
            import benchmark_stage2_pose as pose
            truth=pose.tum_truth(BASE/'datasets/tum'/spec['record']['sequence'],timestamps)
            R,t=pose._decode(graph,x.to(problem.initial));a=pose.metrics(dict(rotations=R,translations=t),truth)
            return dict(metric=a['ate_translation_rmse_m'],**a)
        return problem,score,meta
    problem,score=r.s.compile_case(spec['spec']);return problem,score,dict(higher_better=False)

def run():
    p=register();existing=rows();done={(x['key'],x['arm']) for x in existing};elapsed=sum(x['seconds'] for x in existing)
    r.assert_protected(p);torch.set_num_threads(1);r.s.threadpool_limits(1);r.s.cv2.setNumThreads(1)
    model=r.s.load_model(r.s.CKPT).eval()
    for v in model.parameters():v.requires_grad_(False)
    assert sum(v.numel() for v in model.parameters())==49096
    altered=solver_with_hook()
    with serial_numerics():
        for spec in p['cases']:
            problem,unused,meta=load(spec);del unused
            with r.s.prepared_problem(problem) as (problem,preparation):
                assert hasattr(problem,'native')
                for arm in p['arms']:
                    if (spec['key'],arm) in done:continue
                    assert elapsed<p['solve_seconds_limit']
                    write('state.json',dict(status='running',key=spec['key'],arm=arm,pid=os.getpid(),completed=len(existing),expected=p['route_count'],solve_seconds=elapsed))
                    packet={};audit=PhysicalAudit(problem.native);start=time.perf_counter();cpu=time.process_time()
                    try:
                        cfg=replace(r.s.phase.cfg(spec['group']),snapshot_steps=(1,2,3,4,5))
                        if arm=='setsunet_original':result=r.s.stage5.solve(problem,model,'csn',cfg)
                        elif arm=='lm':result=r.s.classical(problem,{},'lm',spec['group'],steps=5)
                        elif arm=='setsunet_curve':result=r.curve.solve(problem,model,r.CountedEngine(r.engine_for(problem)),curve_mode='norm_mean_envelope',config=cfg)
                        else:result=altered(problem,model,r.CountedEngine(r.engine_for(problem)),curve_mode='norm_mean_envelope',config=cfg,hook=hook_for(arm,problem,audit,p,packet))
                    finally:audit.close()
                    seconds=time.perf_counter()-start;elapsed+=seconds;counts=audit.counts()
                    assert counts['actual_joint_J']<=21
                    assert len(result['trajectory'])==5
                    if arm in ['probe_control','fiber_learned','fiber_measured']:
                        ref=next(v for v in existing if v['key']==spec['key'] and v['arm']=='setsunet_curve')
                        assert np.array_equal(result['snapshots']['1']['theta'],ref['snapshots']['1']['theta'])
                    row=dict(key=spec['key'],group=spec['group'],arm=arm,theta=np.asarray(result['theta']).tolist(),snapshots=result.get('snapshots'),trajectory=result['trajectory'],initial_cost=result['initial_cost'],final_cost=result['final_cost'],counts=counts,seconds=seconds,cpu_seconds=time.process_time()-cpu,preparation=preparation)
                    if packet:
                        row['intervention']=packet['detail']
                        if 'arrays' in packet:
                            name=f"packet_{spec['key']}_{arm}.npz";np.savez_compressed(OUT/name,**packet['arrays']);row['packet']=name;row['packet_sha256']=r.sha(OUT/name)
                    with (OUT/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r.s.safe(row),allow_nan=False)+'\n')
                    existing.append(r.s.safe(row));done.add((spec['key'],arm))
                    print(json.dumps(dict(key=spec['key'],arm=arm,completed=len(existing),seconds=seconds,J=counts['actual_joint_J'],F=counts['joint_forward_only'],changed=packet.get('detail',{}).get('changed'))),flush=True)
    assert len(existing)==60;write('predictions_freeze.json',dict(rows=60,sha256=r.sha(OUT/'predictions.jsonl')))
    score(p,existing);r.assert_protected(p);write('state.json',dict(status='computed_unreviewed',background=False,solve_seconds=elapsed,completed=60))

def score(p,predictions):
    scores=[]
    for spec in p['cases']:
        problem,e,meta=load(spec)
        with r.s.prepared_problem(problem) as (problem,prep):
            for row in predictions:
                if row['key']!=spec['key']:continue
                x=torch.tensor(row['theta'],dtype=problem.initial.dtype);value=e(x)
                objective=float(problem.native.objective(x));assert np.isclose(objective,np.mean(row['final_cost']),rtol=1e-9,atol=1e-12)
                scores.append(dict(key=spec['key'],group=spec['group'],arm=row['arm'],metric=float(value['metric']),details=value,objective=objective,J=row['counts']['actual_joint_J'],F=row['counts']['joint_forward_only'],seconds=row['seconds'],higher_better=meta['higher_better']))
    summary=[]
    for spec in p['cases']:
        a={s['arm']:s for s in scores if s['key']==spec['key']};candidate=a['fiber_learned']
        summary.append(dict(key=spec['key'],group=spec['group'],metrics={k:v['metric'] for k,v in a.items()},
            vs_original=r.s.phase.winner(candidate['metric'],a['setsunet_original']['metric']),vs_curve=r.s.phase.winner(candidate['metric'],a['setsunet_curve']['metric']),vs_lm=r.s.phase.winner(candidate['metric'],a['lm']['metric']),
            seconds={k:v['seconds'] for k,v in a.items()}))
    write('scores.json',dict(records=scores,summary=summary));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    try:run()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
