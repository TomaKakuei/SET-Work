"""Single, capped synthetic TRAIN cached-state action-hull diagnostic.
No original outer solver, new trajectories, model, initializer, label search, renderer, J/H, or benchmark run.
"""
import os,time,sys,json,hashlib,resource,signal,atexit,traceback
from pathlib import Path
from collections import Counter
START=time.monotonic(); CPU=time.process_time()
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]
PRIOR=OUT.parent/'stereo_local_labels_v1'
PLAN=ROOT/'research_run/full126_literature/stereo_action_hull_audit_v1/PLAN.md'
PLAN_SHA='ec67f03e5ad662f76574d89425a6b6cc7aee6737c023b8b7d4b66bae048401ba'
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
 os.environ[name]='1'
resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30))
class BudgetExceeded(RuntimeError):pass
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(BudgetExceeded('120 second cumulative wall cap')))
signal.setitimer(signal.ITIMER_REAL,120)
C=Counter(); SECONDS=Counter(); STATUS={'status':'starting','truth_accessed':False}

def read(p):return json.loads(Path(p).read_text())
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def account():
 write(OUT/'ACCOUNTING.json',dict(**STATUS,counts=dict(C),seconds=dict(SECONDS),wall_seconds=time.monotonic()-START,cpu_seconds=time.process_time()-CPU,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,hard_address_space_bytes=3*2**30,production_objective_gradient_pair_cap=432,production_guard_pair_cap=648,production_original_view_cap=2160,threadpools=threadpool_info() if 'threadpool_info' in globals() else []))
assert not (OUT/'STARTED.json').exists(),'No unaccounted reruns'
write(OUT/'STARTED.json',dict(unix=time.time(),budget_seconds=120))
atexit.register(account)
# Protect prior studies, registered inputs, and TRAIN scoring until output freeze.
def access_guard(event,args):
 if event!='open':return
 path,mode,flags=args
 if not isinstance(path,(str,bytes,os.PathLike)):return
 p=Path(path).resolve()
 writing=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or (isinstance(flags,int) and flags&3!=0)
 if writing:
  if p.is_relative_to(ROOT) and not p.is_relative_to(OUT):raise PermissionError('Prior/source paths are read-only: '+str(p))
  return
 if p.is_relative_to(ROOT/'benchmark126/data') or p.is_relative_to(ROOT/'code/data_stage7') or 'current48' in str(p).lower():raise PermissionError('Protected inputs')
 if p.is_relative_to(ROOT/'research_run') and not p.is_relative_to(OUT):
  allowed=p==PLAN or p.is_relative_to(PRIOR/'observations') or p.is_relative_to(PRIOR/'ordinary') or p in [PRIOR/'INPUT_FREEZE.json',PRIOR/'ORDINARY_FREEZE.json']
  score_allowed=STATUS['truth_accessed'] and (p.is_relative_to(PRIOR/'scorer') or p==PRIOR/'RESULTS.json')
  if not (allowed or score_allowed):raise PermissionError('Outside frozen cached-state scope: '+str(p))
sys.addaudithook(access_guard)
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'benchmark126/source')]
import numpy as np
import torch
import torch.nn.functional as F
import scipy
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits,threadpool_info
torch.set_num_threads(1);torch.set_num_interop_threads(1);threadpool_limits(1)
from neural_grey_v2 import middlebury_stereo as stereo
from setsunet_csn.problems_stage2 import scalar_factor_problem
SCALES=(1.,.5,.25,.125,.0625,.03125); STATES=(0,2,4)
OPTIONS=dict(maxiter=5,maxls=3,maxcor=3,maxfun=12,ftol=2.220446049250313e-9,gtol=1e-5)

def tensor(x):return torch.as_tensor(x,dtype=torch.float64)
def npz(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (keys if keys is not None else z.files)}
def save(p,**arrays):np.savez(p,**{k:v.detach().numpy() if isinstance(v,torch.Tensor) else v for k,v in arrays.items()})
def problem_from(a,scope):
 C[scope+'_problem_constructors']+=1;C[scope+'_constructor_factor_forwards']+=1
 left,right,initial=(tensor(a[k]) for k in ('left','right','initial'))
 obs=stereo.StereoSceneObservation('cached_synthetic_TRAIN',OUT,left,right,initial,((4,24),),(bool(a['valid_crop']),),16)
 f,x,ids,_=stereo.stereo_scene_factor_graph(obs)
 problem=scalar_factor_problem(obs,family='middlebury_stereo',factor_function=f,initial=x[0].double(),block_id=ids)
 assert problem.metadata['factor_count']==63 and problem.initial.numel()==1536
 c=dict(left=left[0,0,4:36,24:72],right=right[:,:,4:36,:],d0=initial[0,0,4:36,24:72],valid=float(a['valid_crop']),x=torch.arange(24,72,dtype=torch.float64)[None].expand(32,-1))
 c['wx']=torch.exp(-5*F.pad(c['left'][:,1:]-c['left'][:,:-1],(0,1)).abs())
 c['wy']=torch.exp(-5*F.pad(c['left'][1:]-c['left'][:-1],(0,0,0,1)).abs())
 return problem,c

def original_pair(problem,x,scope,gradient=False):
 C[scope+'_pair_attempts']+=1;C[scope+'_view_attempts']+=2
 t=time.monotonic()
 try:
  x=x.detach().clone().requires_grad_(gradient)
  with torch.set_grad_enabled(gradient):
   values=torch.stack([problem.cost(x,v) for v in (0,1)])
   if gradient:
    C[scope+'_mean_gradient_attempts']+=1
    g=torch.autograd.grad(values.mean(),x)[0]
   else:g=None
  C[scope+'_pairs_completed']+=1
  return values.detach(),None if g is None else g.detach()
 except Exception:
  C[scope+'_pair_failures']+=1;raise
 finally:SECONDS[scope]+=time.monotonic()-t

def local_energy(d,c):
 C['fixture_local_energy_calls']+=1
 sample=stereo.horizontal_bilinear_sample(c['right'],(c['x']-d)[None])[0,0]
 e=c['left']-sample
 dx=F.pad(d[:,1:]-d[:,:-1],(0,1));dy=F.pad(d[1:]-d[:-1],(0,0,0,1))
 return c['valid']*(.25*(torch.sqrt(e.square()+1e-4)-.01).square()+.001*(d-c['d0']).square()+.04*(c['wx']*torch.sqrt(dx.square()+1e-4)+c['wy']*torch.sqrt(dy.square()+1e-4))).sum()

def state_from(i,k):
 stem=PRIOR/'ordinary'/f'{i:02d}_{k}'
 a=npz(stem.with_suffix('.npz'));m=read(stem.with_suffix('.json'))
 s={key:tensor(a[key]) for key in ('x','p','g','before')}
 s.update(endpoint=tensor(a['original_endpoint']),trajectory=m['trajectory'],trials=[dict(t,theta=tensor(a[f'baseline_trial_{j}'])) for j,t in enumerate(m['baseline_trials'])])
 return a,s

def hull(d,p,q):
 b=d+p;z=b+q;stack=torch.stack((d,b,z))
 return stack.amin(0),stack.amax(0),b,z

def project_truth(truth,l,u,d):
 # No visibility input: all finite truth pixels are projected; nonfinite hold.
 return torch.where(torch.isfinite(truth),torch.minimum(torch.maximum(truth,l),u),d)

class Guard:
 def __init__(self,problem,s,scope):
  self.problem,self.s,self.scope=problem,s,scope
  self.cache={s['x'].numpy().tobytes():s['before']}
  for t in s['trials']:self.cache[t['theta'].numpy().tobytes()]=tensor(t['cost'])
 def costs(self,x):
  C[self.scope+'_guard_pair_requests']+=1;key=x.numpy().tobytes()
  if key in self.cache:C[self.scope+'_guard_cache_hits']+=1;return self.cache[key]
  value,_=original_pair(self.problem,x,self.scope+'_guard');self.cache[key]=value;return value
 def apply(self,h):
  h=h.reshape(-1);s=self.s
  if torch.equal(h,s['p']):
   C[self.scope+'_guard_original_reuse']+=1
   return s['endpoint'].clone(),dict(fallback=True,reason='exact original reuse',accepted=s['trajectory']['accepted'],scale=s['trajectory']['step_scale'],trials=[])
  directional=float(s['g']@h);trials=[]
  if not bool(torch.isfinite(h).all()) or not np.isfinite(directional) or directional>=0:
   C[self.scope+'_guard_nondescent_or_nonfinite']+=1
   return s['endpoint'].clone(),dict(fallback=True,reason='nonfinite or non-descent',accepted=False,scale=0.,directional=directional if np.isfinite(directional) else None,trials=[])
  for scale in SCALES:
   x=s['x']+scale*h;value=self.costs(x);C[self.scope+'_guard_trials']+=1
   limit=float(s['before'].mean())+1e-4*scale*directional+1e-14*max(float(s['before'].abs().mean()),1e-30)
   accepted=bool(torch.isfinite(value).all()) and float(value.mean())<=limit
   trials.append(dict(scale=scale,cost=value.tolist(),limit=limit,accepted=accepted))
   if accepted:return x,dict(fallback=False,reason='first accepted original guard',accepted=True,scale=scale,directional=directional,trials=trials)
  C[self.scope+'_guard_six_trial_fallback']+=1
  return s['endpoint'].clone(),dict(fallback=True,reason='six trials rejected',accepted=False,scale=0.,directional=directional,trials=trials)

class PairCap(RuntimeError):pass

def numerical(problem,c,s,l,u):
 lo,hi=l.numpy().ravel(),u.numpy().ravel();d0=c['d0'].numpy().ravel()
 raw=(c['d0']+s['endpoint'].reshape(32,48)).numpy().ravel()
 start=np.clip(raw,lo,hi);cache={};calls=0;best=None;best_cost=float('inf');trace=[]
 assert np.max(np.abs(raw-start))<=1e-12*max(1.,float(np.max(np.abs(raw)))),'warm-start violation exceeds roundoff'
 def fg(v):
  nonlocal calls,best,best_cost
  key=np.asarray(v,dtype=np.float64).tobytes();C['production_numerical_pair_requests']+=1
  if key in cache:C['production_numerical_exact_cache_hits']+=1;return cache[key]
  if calls>=12:C['production_numerical_cap_hits']+=1;raise PairCap('hard 12 objective+gradient pairs')
  calls+=1
  if not np.isfinite(v).all() or np.any(v<lo) or np.any(v>hi):raise ValueError('infeasible numerical evaluation')
  costs,g=original_pair(problem,tensor(v-d0),'production_numerical',gradient=True)
  f=float(costs.mean());gn=g.numpy().copy()
  if not np.isfinite(f) or not np.isfinite(gn).all():raise FloatingPointError('nonfinite objective or gradient')
  if f<best_cost:best_cost=f;best=np.asarray(v).copy()
  trace.append(dict(call=calls,mean_cost=f,incumbent_cost=best_cost))
  cache[key]=(f,gn);return cache[key]
 status={}
 try:
  fg(start) # Always actually evaluate even when roundoff clipping changed bytes.
  result=minimize(fg,start,method='L-BFGS-B',jac=True,bounds=list(zip(lo,hi)),options=OPTIONS)
  status=dict(status=int(result.status),success=bool(result.success),message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),njev=int(result.njev),cap_hit=False)
 except PairCap as e:status=dict(status='hard_pair_cap',success=False,message=str(e),cap_hit=True)
 except BudgetExceeded:raise
 except Exception as e:
  C['production_numerical_failures']+=1;status=dict(status='failure',success=False,message=repr(e),cap_hit=False)
 if not status['success']:C['production_numerical_non_success_states']+=1
 status.update(actual_objective_gradient_pairs=calls,start_clipped_coordinates=int(np.count_nonzero(raw!=start)),start_max_clip=float(np.max(np.abs(raw-start))),trace=trace,incumbent_cost=best_cost if np.isfinite(best_cost) else None)
 if best is None:
  status['incumbent_missing_exact_original_fallback']=True
  return None,status
 return tensor(best.reshape(32,48)),status

def fixtures():
 result={}
 # Invented clamp and closed convex hull inclusions, including prior scalar family.
 d=tensor([[-2.,1.,4.],[2.,3.,0.]])
 p=tensor([[1.,-4.,3.],[-1.,0.,0.]])
 q=tensor([[-3.,5.,-2.],[6.,0.,0.]])
 l,u,b,z=hull(d,p,q)
 truth=tensor([[-9.,9.,5.],[float('nan'),3.,2.]])
 t=project_truth(truth,l,u,d)
 assert bool(((t>=l)&(t<=u)).all()) and t[1,0]==d[1,0]
 assert torch.equal(t,tensor([[-4.,2.,5.],[2.,3.,0.]]))
 for alpha in (0.,)+SCALES:
  for a in (0.,.137,.5,1.):
   v=d+alpha*p+alpha*a*q
   assert bool(((v>=l-1e-14)&(v<=u+1e-14)).all())
 bary=(d+b+z)/3;assert bool(((bary>=l)&(bary<=u)).all())
 result['clamp_finite_truth_null_coordinates_and_prior_scalar_hull']=True
 # Null, non-descent, all-six reject, first-scale accept with invented guard.
 s=dict(x=torch.zeros(2,dtype=torch.float64),p=tensor([-1.,0.]),g=tensor([1.,0.]),before=tensor([1.,1.]),endpoint=tensor([-.5,0.]),trajectory=dict(accepted=True,step_scale=.5),trials=[])
 class Fake:
  def __init__(self,value):self.value=value
  def cost(self,x,v):return x.new_tensor(self.value)
 guard=Guard(Fake(2.),s,'fixture')
 endpoint,info=guard.apply(s['p']);assert torch.equal(endpoint,s['endpoint']) and info['fallback']
 endpoint,info=guard.apply(torch.zeros(2,dtype=torch.float64));assert torch.equal(endpoint,s['endpoint']) and info['fallback']
 endpoint,info=guard.apply(tensor([float('nan'),0.]));assert torch.equal(endpoint,s['endpoint']) and info['fallback']
 endpoint,info=guard.apply(tensor([-.2,0.]));assert torch.equal(endpoint,s['endpoint']) and len(info['trials'])==6
 endpoint,info=Guard(Fake(.5),s,'fixture').apply(tensor([-.2,0.]));assert torch.equal(endpoint,tensor([-.2,0.])) and info['scale']==1.
 result['original_guard_null_nonfinite_nondescent_six_reject_first_accept']=True
 # Original compiled mean cost/autograd versus exact local energy/autograd.
 yy=torch.arange(40,dtype=torch.float64)[:,None];xx=torch.arange(96,dtype=torch.float64)[None,:]
 a=dict(left=(.5+.2*torch.sin(.17*xx+.11*yy))[None,None],right=(.5+.2*torch.sin(.17*xx+.11*yy+.13))[None,None],initial=torch.full((1,1,40,96),3.,dtype=torch.float64),valid_crop=True)
 problem,c=problem_from(a,'fixture')
 x=(.12+.03*torch.sin(torch.arange(1536,dtype=torch.float64)*.1)).requires_grad_(True)
 values,g=original_pair(problem,x,'fixture_parity',True)
 energy=local_energy(c['d0']+x.reshape(32,48),c);gl=torch.autograd.grad(energy,x)[0]
 error=abs(float(values.mean())-float(energy));grad_error=float((g-gl).abs().max())
 assert error<1e-11 and grad_error<1e-11,(error,grad_error)
 result['original_compiled_local_energy_gradient_parity']=dict(energy_abs_error=error,gradient_max_abs_error=grad_error)
 write(OUT/'FIXTURES.json',dict(passed=True,checks=result,wall_seconds=time.monotonic()-START))


def freeze_source():
 assert sha(PLAN)==PLAN_SHA
 paths=[Path(__file__),OUT/'CONFIG.json',PLAN,ROOT/'benchmark126/source/neural_grey_v2/middlebury_stereo.py',ROOT/'benchmark126/source/neural_grey_v2/factor_sketch.py',ROOT/'code/setsunet_csn/problems_stage2.py',ROOT/'code/setsunet_csn/problems.py']
 write(OUT/'SOURCE_FREEZE.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},scipy=scipy.__version__,numpy=np.__version__,torch=torch.__version__,options=OPTIONS,created_unix=time.time(),truth_opened=False))

def verify_sources():
 for path,h in read(OUT/'SOURCE_FREEZE.json')['hashes'].items():assert sha(ROOT/path)==h

def mae(d,t,m):
 C['scorer_mae_evaluations']+=1;C['scorer_mae_pixel_terms']+=int(m.sum())
 return float((d-t).abs()[m].mean())

def main():
 freeze_source();fixtures();verify_sources()
 inputs=read(PRIOR/'INPUT_FREEZE.json');frozen=read(PRIOR/'ORDINARY_FREEZE.json')
 manifest={'observations':{},'ordinary':{}}
 for i in range(12):
  p=PRIOR/'observations'/f'{i:02d}.npz';h=sha(p);assert h==inputs['rows'][i]['observation_sha256'];manifest['observations'][p.name]=h
 for r in frozen['rows']:
  for suffix in ('npz','json'):
   p=PRIOR/'ordinary'/f"{r['index']:02d}_{r['state']}.{suffix}";h=sha(p);assert h==r[suffix+'_sha256'];manifest['ordinary'][p.name]=h
 assert len(frozen['rows'])==36
 write(OUT/'INPUT_FREEZE.json',dict(prior_input_freeze_sha256=sha(PRIOR/'INPUT_FREEZE.json'),prior_ordinary_freeze_sha256=sha(PRIOR/'ORDINARY_FREEZE.json'),hashes=manifest))
 observed=[];problems={}
 (OUT/'observed').mkdir();(OUT/'teachers').mkdir()
 STATUS.update(status='observed_controls',truth_accessed=False)
 for i in range(12):
  a=npz(PRIOR/'observations'/f'{i:02d}.npz',('left','right','initial','valid_crop'))
  problem,c=problem_from(a,'production');problems[i]=(problem,c)
  for k in STATES:
   a,s=state_from(i,k);d=c['d0']+s['x'].reshape(32,48);l,u,b,z=hull(d,s['p'].reshape(32,48),tensor(a['q']))
   target,note=numerical(problem,c,s,l,u);guard=Guard(problem,s,'production')
   if target is None:endpoint=s['endpoint'].clone();info=dict(fallback=True,reason='no finite numerical incumbent',accepted=False,scale=0.,trials=[])
   else:endpoint,info=guard.apply(target-d)
   bary=(d+b+z)/3;be,bi=guard.apply(bary-d)
   path=OUT/'observed'/f'{i:02d}_{k}.npz'
   save(path,numerical_endpoint=endpoint,barycenter_endpoint=be,numerical_target=target if target is not None else c['d0']+s['endpoint'].reshape(32,48),barycenter_target=bary,l=l,u=u)
   observed.append(dict(index=i,state=k,regime=i//4,numerical=note,numerical_guard=info,barycenter_guard=bi))
  print('OBSERVED',i+1,'/12',round(time.monotonic()-START,3),flush=True)
 write(OUT/'OBSERVED_RESULTS.json',observed)
 files=sorted((OUT/'observed').glob('*.npz'))+[OUT/'OBSERVED_RESULTS.json']
 write(OUT/'OBSERVED_FREEZE.json',dict(all_36_observed_endpoints_frozen=True,truth_opened=False,created_unix=time.time(),hashes={str(p.relative_to(OUT)):sha(p) for p in files}))
 verify_sources()
 for p,h in read(OUT/'OBSERVED_FREEZE.json')['hashes'].items():assert sha(OUT/p)==h
 STATUS.update(status='train_teacher_and_score',truth_accessed=True)
 write(OUT/'SCORER_STARTED.json',dict(created_unix=time.time(),observed_freeze_sha256=sha(OUT/'OBSERVED_FREEZE.json')))
 prior_scores={(r['index'],r['state']):r for r in read(PRIOR/'RESULTS.json')['records']}
 records=[]
 for i in range(12):
  path=PRIOR/'scorer'/f'{i:02d}.npz';assert sha(path)==inputs['rows'][i]['scorer_sha256']
  score=npz(path);truth=tensor(score['disparity']);mask=torch.as_tensor(score['visibility']).bool()&torch.isfinite(truth)
  assert bool(mask.any());problem,c=problems[i]
  for k in STATES:
   a,s=state_from(i,k);f=npz(OUT/'observed'/f'{i:02d}_{k}.npz');d=c['d0']+s['x'].reshape(32,48);l,u=tensor(f['l']),tensor(f['u'])
   target=project_truth(truth,l,u,d);C['teacher_finite_pixel_projections']+=int(torch.isfinite(truth).sum())
   endpoint,info=Guard(problem,s,'production').apply(target-d)
   save(OUT/'teachers'/f'{i:02d}_{k}.npz',target=target,endpoint=endpoint)
   original=mae(c['d0']+s['endpoint'].reshape(32,48),truth,mask)
   assert abs(original-prior_scores[i,k]['ordinary_mae']['original'])<1e-12
   scalar=float(prior_scores[i,k]['global_scalar_bound']['mae']);threshold=max(.002,.01*original)
   scores=dict(original=original,prior_scalar_bound=scalar,pixel_box_bound=mae(target,truth,mask),finite_teacher=mae(c['d0']+endpoint.reshape(32,48),truth,mask),numerical=mae(c['d0']+tensor(f['numerical_endpoint']).reshape(32,48),truth,mask),barycenter=mae(c['d0']+tensor(f['barycenter_endpoint']).reshape(32,48),truth,mask))
   assert scores['pixel_box_bound']<=scalar+1e-12
   row=dict(index=i,state=k,regime=i//4,finite_visible_pixels=int(mask.sum()),finite_truth_pixels=int(torch.isfinite(truth).sum()),mae=scores,threshold=threshold,teacher_guard=info,gains_vs_original={name:original-value for name,value in scores.items()},gains_vs_scalar={name:scalar-value for name,value in scores.items()},sizable_vs_original={name:original-value>=threshold for name,value in scores.items()},sizable_vs_scalar={name:scalar-value>=threshold for name,value in scores.items()})
   row['raw_proposal_mae']=dict(finite_teacher=scores['pixel_box_bound'],numerical=mae(tensor(f['numerical_target']),truth,mask) if not observed[len(records)]['numerical'].get('incumbent_missing_exact_original_fallback',False) else None,barycenter=mae(tensor(f['barycenter_target']),truth,mask))
   row['numerical_raw']=row['raw_proposal_mae']['numerical'];row['barycenter_raw']=row['raw_proposal_mae']['barycenter']
   records.append(row)
  print('SCORED',i+1,'/12',round(time.monotonic()-START,3),flush=True)
 names=('original','prior_scalar_bound','pixel_box_bound','finite_teacher','numerical','barycenter')
 def aggregate(rows):
  return dict(states=len(rows),raw_mean_state_mae={n:float(np.mean([r[n] for r in rows])) if all(r[n] is not None for r in rows) else None for n in ('numerical_raw','barycenter_raw')},raw_available_states={n:sum(r[n] is not None for r in rows) for n in ('numerical_raw','barycenter_raw')},mean_state_mae={n:float(np.mean([r['mae'][n] for r in rows])) for n in names},sizable_vs_original={n:sum(r['sizable_vs_original'][n] for r in rows) for n in names},sizable_vs_scalar={n:sum(r['sizable_vs_scalar'][n] for r in rows) for n in names},scene_any_sizable_vs_scalar={n:sum(any(r['sizable_vs_scalar'][n] for r in rows if r['index']==i) for i in sorted({r['index'] for r in rows})) for n in names})
 result=dict(status='completed',scope='synthetic TRAIN cached-state diagnostic only; original failed branch unchanged',states=36,scenes=12,records=records,summary=aggregate(records),by_regime={str(r):aggregate([x for x in records if x['regime']==r]) for r in range(3)},by_state={str(k):aggregate([x for x in records if x['state']==k]) for k in STATES},by_scene={str(i):aggregate([x for x in records if x['index']==i]) for i in range(12)},no_gate_reinterpretation=True,no_promotion=True,no_learnability_claim=True)
 result['numerical_status_counts']=dict(Counter(str(r['numerical']['status'])+': '+r['numerical']['message'] for r in observed))
 result['guard_summary']={name:dict(fallbacks=sum(r[name+'_guard']['fallback'] for r in observed),accepted_scales=dict(Counter(str(r[name+'_guard']['scale']) for r in observed if not r[name+'_guard']['fallback']))) for name in ('numerical','barycenter')}
 result['guard_summary']['finite_teacher']=dict(fallbacks=sum(r['teacher_guard']['fallback'] for r in records),accepted_scales=dict(Counter(str(r['teacher_guard']['scale']) for r in records if not r['teacher_guard']['fallback'])))
 write(OUT/'RESULTS.json',result)
 import csv
 with (OUT/'STATE_RESULTS.csv').open('w',newline='') as file:
  writer=csv.DictWriter(file,fieldnames=['index','regime','state','threshold']+list(names));writer.writeheader()
  for r in records:writer.writerow({**{key:r[key] for key in ('index','regime','state','threshold')},**r['mae']})
 # Final verification accesses only the exact already-authorized cached inputs.
 for name,h in manifest['observations'].items():assert sha(PRIOR/'observations'/name)==h
 for name,h in manifest['ordinary'].items():assert sha(PRIOR/'ordinary'/name)==h
 verify_sources()
 assert C['production_numerical_pair_attempts']<=432
 assert C['production_guard_pair_attempts']<=648
 assert C['production_numerical_view_attempts']+C['production_guard_view_attempts']<=2160
 STATUS.update(status='completed',states=36,observed_freeze_sha256=sha(OUT/'OBSERVED_FREEZE.json'))
 account()
 write(OUT/'FINAL_MANIFEST.json',dict(hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('FINAL_MANIFEST.json','RUN.log','ACCOUNTING.json')},prior_observation_and_ordinary_hashes_unchanged=True))
 print(json.dumps(result['summary']),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as error:
  STATUS.update(status='failed',exception=repr(error));traceback.print_exc();raise
