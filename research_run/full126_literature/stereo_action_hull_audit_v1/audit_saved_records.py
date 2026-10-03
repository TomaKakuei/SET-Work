"""Independent cached TRAIN array arithmetic only; no optimizer/evaluator imports."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import pathlib,json,hashlib,time,collections,math
import numpy as np
start=time.monotonic();root=pathlib.Path(__file__).resolve().parents[3];out=root/'research_run/full126_training/stereo_action_hull_audit_v1';old=out.parent/'stereo_local_labels_v1';dest=pathlib.Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text());source=read(out/'SOURCE_FREEZE.json');frozen=read(out/'OBSERVED_FREEZE.json');r=read(out/'RESULTS.json');obs=read(out/'OBSERVED_RESULTS.json');acc=read(out/'ACCOUNTING.json')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
for p,h in source['hashes'].items():assert sha(root/p)==h
for p,h in frozen['hashes'].items():assert sha(out/p)==h
assert source['created_unix']<frozen['created_unix']<read(out/'SCORER_STARTED.json')['created_unix']
records=[];trials=0;pairs=0;ends=0;scoreerr=0.;scales=collections.Counter();worse=[]
for row,o in zip(r['records'],obs):
 i,k=row['index'],row['state'];assert (i,k)==(o['index'],o['state'])
 with np.load(old/'observations'/f'{i:02d}.npz',allow_pickle=False) as f:d0=f['initial'][0,0,4:36,24:72].astype(float)
 a=load(old/'ordinary'/f'{i:02d}_{k}.npz');v=load(out/'observed'/f'{i:02d}_{k}.npz');t=load(out/'teachers'/f'{i:02d}_{k}.npz');gt=load(old/'scorer'/f'{i:02d}.npz');truth=gt['disparity'];mask=gt['visibility'].astype(bool)&np.isfinite(truth)
 x=a['x'];d=d0+x.reshape(32,48);b=d+a['p'].reshape(32,48);z=b+a['q'];l=np.minimum(np.minimum(d,b),z);u=np.maximum(np.maximum(d,b),z)
 assert np.array_equal(l,v['l']) and np.array_equal(u,v['u'])
 target=np.where(np.isfinite(truth),np.minimum(np.maximum(truth,l),u),d);assert np.array_equal(target,t['target'])
 assert np.array_equal((d+b+z)/3,v['barycenter_target']);assert np.all(v['numerical_target']>=l) and np.all(v['numerical_target']<=u)
 measures={'original':d0+a['original_endpoint'].reshape(32,48),'pixel_box_bound':target,'finite_teacher':d0+t['endpoint'].reshape(32,48),'numerical':d0+v['numerical_endpoint'].reshape(32,48),'barycenter':d0+v['barycenter_endpoint'].reshape(32,48)}
 scores={n:float(np.mean(abs(value-truth)[mask])) for n,value in measures.items()}
 for n,value in scores.items():scoreerr=max(scoreerr,abs(value-row['mae'][n]));assert math.isclose(value,row['mae'][n],rel_tol=1e-13,abs_tol=1e-13)
 for arm in ('numerical','barycenter'):assert math.isclose(float(np.mean(abs(v[arm+'_target']-truth)[mask])),row['raw_proposal_mae'][arm],rel_tol=1e-13,abs_tol=1e-13)
 assert scores['pixel_box_bound']<=row['mae']['prior_scalar_bound']+1e-12
 note=o['numerical'];trace=note['trace'];n=note['actual_objective_gradient_pairs'];assert n==len(trace)<=12 and note['nit']==5 and note['status']==1 and not note['success'] and not note['cap_hit'];pairs+=n
 for j,item in enumerate(trace):assert item['call']==j+1 and item['incumbent_cost']==min(z['mean_cost'] for z in trace[:j+1])
 assert note['incumbent_cost']==min(z['mean_cost'] for z in trace)
 costs={}
 for arm,info,target,endpoint in [('numerical',o['numerical_guard'],v['numerical_target'],v['numerical_endpoint']),('barycenter',o['barycenter_guard'],v['barycenter_target'],v['barycenter_endpoint']),('teacher',row['teacher_guard'],t['target'],t['endpoint'])]:
  h=(target-d).reshape(-1);directional=float(a['g']@h);assert np.isfinite(h).all() and directional<0 and math.isclose(directional,info['directional'],rel_tol=1e-12,abs_tol=1e-13)
  assert info['accepted'] and not info['fallback'];ts=info['trials'];before=a['before']
  for j,trial in enumerate(ts):
   scale=.5**j;assert trial['scale']==scale;limit=float(before.mean())+1e-4*scale*directional+1e-14*max(float(np.abs(before).mean()),1e-30)
   assert math.isclose(limit,trial['limit'],rel_tol=1e-13,abs_tol=1e-13);accepted=np.isfinite(trial['cost']).all() and np.mean(trial['cost'])<=limit;assert bool(accepted)==trial['accepted'];assert not accepted or j==len(ts)-1;trials+=1
  assert ts[-1]['scale']==info['scale'] and np.array_equal(x+info['scale']*h,endpoint);ends+=1;costs[arm]=float(np.mean(ts[-1]['cost']));scales[arm+'_scale_'+str(info['scale'])]+=1
 bad=costs['numerical']<costs['teacher'] and scores['numerical']>scores['finite_teacher']+1e-10
 if bad:worse.append([i,k])
 qzero=(a['q']==0)&mask;outside=(t['target']<np.minimum(b,z))|(t['target']>np.maximum(b,z));changed=t['target']!=b
 records.append(dict(index=i,state=k,regime=i//4,visible_pixels=int(mask.sum()),visible_q_exact_zero=int(qzero.sum()),visible_teacher_differs_from_b_where_q_zero=int((qzero&changed).sum()),visible_teacher_outside_full_step_bz_interval=int((outside&mask).sum()),numerical_lower_objective_but_higher_mae_than_teacher=bool(bad)))
assert ends==108 and pairs==acc['counts']['production_numerical_pair_attempts']==226 and trials==acc['counts']['production_guard_trials']==132
assert acc['counts']['production_numerical_view_attempts']+acc['counts']['production_guard_view_attempts']==716
assert all(z['num_threads']==1 for z in acc['threadpools']) and acc['wall_seconds']<120
for n in r['summary']['mean_state_mae']:
 assert math.isclose(r['summary']['mean_state_mae'][n],float(np.mean([z['mae'][n] for z in r['records']])),rel_tol=1e-14)
 assert r['summary']['sizable_vs_scalar'][n]==sum(z['mae']['prior_scalar_bound']-z['mae'][n]>=max(.002,.01*z['mae']['original']) for z in r['records'])
report=dict(status='independent_cached_algebra_score_and_guard_audit_passed',source_freeze_sha256=sha(out/'SOURCE_FREEZE.json'),observed_freeze_sha256=sha(out/'OBSERVED_FREEZE.json'),results_sha256=sha(out/'RESULTS.json'),all_source_and_observed_hashes_match=True,observed_freeze_precedes_truth_scoring=True,exact_hull_and_teacher_projection_checks=36,applied_endpoint_checks=ends,guard_trial_checks=trials,max_endpoint_error=0.,max_teacher_projection_error=0.,max_recomputed_mae_error=scoreerr,numerical_pairs=pairs,production_view_evaluations=716,all_numerical_terminations='5-iteration limit; no convergence claim, no hard12-pair caps or execution failures',guard_scale_counts=dict(scales),numerical_lower_objective_but_higher_mae_than_teacher_states=len(worse),such_state_keys=worse,local_retention_descriptives=records,totals={key:sum(z[key] for z in records) for key in ('visible_pixels','visible_q_exact_zero','visible_teacher_differs_from_b_where_q_zero','visible_teacher_outside_full_step_bz_interval')},review_seconds=time.monotonic()-start,review_scope='Cached TRAIN initial-disparity, original thin states, new frozen targets/endpoints, and TRAIN truth/visibility arrays for vector arithmetic only. No image arrays, renderer, original or numerical optimizer, gradients, J/H or evaluators invoked by reviewer. The outside-bz statistic concerns the full-step [b,z] interval only, not the union over old global scales.')
(dest/'INDEPENDENT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('local_retention_descriptives','such_state_keys')},indent=2))
