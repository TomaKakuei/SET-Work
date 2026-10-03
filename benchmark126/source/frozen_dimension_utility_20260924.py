"""Bounded dimension attribution; isolated solver, immutable trained checkpoint."""
from pathlib import Path
import json,time,types,gzip,sys,traceback
from dataclasses import replace
import run_fiber_system_20260924 as d
import cloudpickle
r,np,torch=d.r,d.np,d.torch
OLD=d.OUT
OUT=r.ROOT/'results_stage9_dev/runs/dimension_utility_20260924'
ARMS=['original','none','prefix1','prefix2','prefix4','svd1','svd2','svd4','random8a','random8b','analytic8','rotated8','untrained8']

def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/name;t=p.with_suffix(p.suffix+'.pending')
    t.write_text(json.dumps(r.s.safe(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)

def specifications():
    return r.read(OLD/'protocol.json')['cases']+r.read(OLD/'visual_protocol.json')['cases']

def load(spec):
    if spec['key'].startswith('photo_'):
        with gzip.open(OLD/f"input_{spec['key']}.pkl.gz",'rb') as f:return cloudpickle.load(f)
    return d.load(spec)

def register():
    if (OUT/'protocol.json').exists():return r.read(OUT/'protocol.json')
    old=r.read(OLD/'protocol.json')
    p=dict(schema='dimension-utility-v1',cases=specifications(),arms=ARMS,steps=5,
        scope='2026-09-24 user explicitly relaxes the hard compute/J cap. Keep five outer steps for comparison; additional work must be attributed to useful quality. No long-convergence sweep.',
        purpose='Separate raw proposal rank, projected rank, and causal five-step utility. Existing 12 development cases, not new independent confirmation.',
        ablations='Same Q8 and original per-group CG. Prefix trims learned heads. SVD trims leading singular vectors AFTER the exact oblique projection and cleanup. SVD energy is conditioning, not task utility. All arms still evaluate the full network; no speed-saving claim.',
        controls='Two fixed random direction streams; one random-weight same architecture network; deterministic analytic residual/response proposal; orthogonal rotation of all eight heads.',
        seeds=dict(random8a=92495101,random8b=92495102,untrained8=92495103,rotation=92495104),
        numerical_rank='Singular value >1e-9*Frobenius norm of original V, same global scale as completion. Original sequential QR rank retained separately.',
        gain_diagnostic='Correction coefficients expressed in left singular vectors of pre-orthogonalized projected V. Energy dimensions are basis-dependent diagnostics; counterfactual full trajectories decide quality.',
        freeze='All endpoints frozen before scoring. No truth passed to solver or dimension selector.',
        route_count=len(ARMS)*12,training_updates=0,protected_hashes=old['protected_hashes'],
        input_hashes={str(OLD/f'input_{k}.pkl.gz'):r.sha(OLD/f'input_{k}.pkl.gz') for k in ['photo_i_nuts','photo_i_objects']},
        source_sha256=r.sha(__file__),solve_seconds_limit=240)
    write('protocol.json',p);return p

def projected(base,V):
    Z=V.clone()
    if base['Q'].shape[1]:
        Z=Z-base['Q']@torch.linalg.solve(base['A'],base['Y'].T@Z)
        U=torch.linalg.qr(base['Y'],mode='reduced')[0]
        for _ in range(2):Z=Z-U@(U.T@Z)
    return Z

def spectrum(V,reference=None):
    U,S,_=torch.linalg.svd(V,full_matrices=False)
    threshold=1e-9*float(V.norm() if reference is None else reference)
    rank=int((S>threshold).sum());energy=S.square();total=float(energy.sum())
    participation=total**2/max(float(energy.square().sum()),1e-300) if total else 0.
    n95=int(torch.searchsorted(energy.cumsum(0),energy.sum()*.95))+1 if total else 0
    return U,S,dict(rank=rank,participation=participation,energy95=n95,singular_values=S.tolist())

def isolated(arm,seed,diagnostics,arrays):
    source=Path(r.s.stage5.__file__).read_text(encoding='utf-8')
    mod=types.ModuleType('setsunet_csn.dimension_isolated');mod.__package__='setsunet_csn'
    exec(compile(source,'<dimension-isolated>','exec'),mod.__dict__)
    original=mod.complete;rng=np.random.default_rng(seed)
    rotation=torch.from_numpy(np.linalg.qr(np.random.default_rng(92495104).normal(size=(8,8)))[0])
    def complete(base,a,b,V,cfg,counts,consensus=True):
        inputV=V
        if arm=='none':V=V[:,:0]
        elif arm.startswith('prefix'):V=V[:,:int(arm[-1])]
        elif arm.startswith('svd'):
            zz=projected(base,V);U,S,sp=spectrum(zz,V.norm());V=U[:,:min(int(arm[-1]),sp['rank'])]
        elif arm.startswith('random8'):
            V=torch.from_numpy(rng.normal(size=tuple(V.shape))).to(V);V=V/V.norm(dim=0)
        elif arm=='analytic8':
            V=mod.orthogonalize(torch.cat((base['e'][:,None],base['Y'],base['Ya']-base['Yb']),1),cfg.rank_tolerance,backend=cfg.backend)[:,:8]
        elif arm=='rotated8':V=V@rotation
        out=original(base,a,b,V,cfg,counts,consensus)
        if arm=='original':
            zz=projected(base,inputV);U,S,sp=spectrum(zz,inputV.norm());_,_,raw=spectrum(inputV)
            correction=out[0]-base['p0'];coef=U.T@correction;energy=coef.square();total=float(energy.sum())
            relative=float(correction.norm()/out[0].norm().clamp_min(1e-30))
            meaningful=relative>1e-8 and sp['rank']>0
            order=torch.argsort(energy,descending=True)
            correction95=int(torch.searchsorted(energy[order].cumsum(0),.95*energy.sum()))+1 if meaningful else 0
            usedS=energy/energy.sum().clamp_min(1e-300)
            diagnostic=dict(n=V.shape[0],measured_rank=base['Q'].shape[1],raw=raw,projected=sp,
                executed_rank=out[3]['rank'],certificate_fallback=out[3]['certificate_fallback'],completion_norm=float(correction.norm()),
                relative_completion=relative,correction95=correction95,correction_energy_by_singular_axis=usedS.tolist(),
                correction_first_spectral_axis=float(usedS[0]),gain=float(out[2]),observed_step_pending=True)
            i=len(diagnostics);diagnostics.append(diagnostic)
            for key,value in dict(Q=base['Q'],Y=base['Y'],A=base['A'],V=inputV,Zpre=zz,U=U,S=S,Ha=a.H,Hb=b.H,ga=a.g,gb=b.g,p0=base['p0'],step=out[0],action=out[1],correction=correction).items():arrays[f'{i}_{key}']=value.detach().numpy()
        return out
    mod.complete=complete;return mod.solve

def run():
    p=register();r.assert_protected(p);torch.set_num_threads(1);r.s.threadpool_limits(1);r.s.cv2.setNumThreads(1)
    payload=torch.load(r.s.CKPT,map_location='cpu',weights_only=False)
    write('checkpoint_metadata.json',{k:v for k,v in payload.items() if k!='state_dict'})
    model=r.s.load_model(r.s.CKPT).eval();torch.manual_seed(p['seeds']['untrained8']);untrained=type(model)(**payload['architecture']).double().eval()
    for m in [model,untrained]:
        for v in m.parameters():v.requires_grad_(False)
    path=OUT/'predictions.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={(x['key'],x['arm']) for x in rows};elapsed=sum(x['seconds'] for x in rows)
    with d.serial_numerics():
        for spec in p['cases']:
            problem,unused,meta=load(spec);del unused
            for arm in ARMS:
                if (spec['key'],arm) in done:continue
                assert elapsed<p['solve_seconds_limit'];diag=[];arrays={};solver=isolated(arm,p['seeds'].get(arm,92495100),diag,arrays)
                audit=d.PhysicalAudit(problem.native);started=time.perf_counter()
                try:result=solver(problem,untrained if arm=='untrained8' else model,'csn',replace(r.s.phase.cfg(spec['group']),snapshot_steps=(1,2,3,4,5)))
                finally:audit.close()
                seconds=time.perf_counter()-started;elapsed+=seconds
                row=dict(key=spec['key'],group=spec['group'],arm=arm,theta=np.asarray(result['theta']).tolist(),n=problem.initial.numel(),seconds=seconds,counts=audit.counts(),solver_counts=result['counts'],trajectory=result['trajectory'],snapshots=result['snapshots'],final_cost=result['final_cost'])
                if diag:
                    for info,step in zip(diag,result['trajectory']):info.update(accepted=step['accepted'],step_scale=step['step_scale'],observed_step_pending=False)
                    row['diagnostics']=diag;name=f"audit_{spec['key']}.npz";np.savez_compressed(OUT/name,**arrays);row['packet']=name;row['packet_sha256']=r.sha(OUT/name)
                with path.open('a',encoding='utf-8') as f:f.write(json.dumps(r.s.safe(row),allow_nan=False)+'\n')
                rows.append(row);done.add((spec['key'],arm))
            write('state.json',dict(status='running',completed=len(rows),expected=p['route_count'],solve_seconds=elapsed))
            print(json.dumps(dict(key=spec['key'],completed=len(rows),solve_seconds=elapsed)),flush=True)
    assert len(rows)==p['route_count'];write('freeze.json',dict(rows=len(rows),sha256=r.sha(path)));scores=[]
    for spec in p['cases']:
        problem,e,meta=load(spec)
        for row in [q for q in rows if q['key']==spec['key']]:
            theta=torch.tensor(row['theta'],dtype=problem.initial.dtype)
            scores.append(dict(key=row['key'],arm=row['arm'],metric=float(e(theta)['metric']),objective=float(problem.native.objective(theta)),J=row['counts']['actual_joint_J'],F=row['counts']['joint_forward_only'],seconds=row['seconds']))
    write('scores.json',scores);r.assert_protected(p)
    write('state.json',dict(status='audit_computed_unreviewed',background=False,completed=len(rows),solve_seconds=elapsed))
    for spec in p['cases']:
        print(spec['key'],json.dumps({q['arm']:q['metric'] for q in scores if q['key']==spec['key']}))

if __name__=='__main__':
    try:run()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
