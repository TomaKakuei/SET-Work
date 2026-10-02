"""Independent regression, sparse nonlinear equations, and view-noise controls.

All solvers share analytic derivatives; the engine contains no certified
solution or geometric truth. NIST references and synthetic truth are only
captured by the separate evaluator returned by compile_case.
"""
from pathlib import Path
import json
import numpy as np
import scipy.sparse as sp
import torch
from setsunet_csn.problems import ResidualProblem
ROOT=Path(__file__).resolve().parents[1]


class AnalyticFactors:
    evaluator_backend = 'project_NumPy_SciPy_analytic_factors'
    evaluates_in_c = False
    def __init__(self,n,raw_jac):
        self.n,self.raw_jac,self.calls=n,raw_jac,[0,0,0]
    def evaluate(self,x,view,mode=2,*,jacobian=False):
        z=x.detach().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
        r,J=self.raw_jac(z,view,mode>0 or jacobian)
        self.calls[mode]+=1
        g=np.asarray(J.T@r).reshape(-1) if mode else None
        H=J.T@J if mode==2 else None
        if sp.issparse(H):H=H.toarray()
        return {'r':np.asarray(r),'J':J,'g':g,'H':H,'w':np.ones_like(r),'cost':float(.5*r@r),'group_width':1}
    def cost(self,x,view):return torch.tensor(self.evaluate(x,view,0)['cost'],dtype=torch.float64)
    def objective(self,x,gradient=False):
        a,b=[self.evaluate(x,v,int(gradient)) for v in (0,1)]
        f=.5*(a['cost']+b['cost'])
        return (f,.5*(a['g']+b['g'])) if gradient else f
    def least_squares(self,x,jacobian=True):
        a,b=[self.raw_jac(np.asarray(x),v,jacobian) for v in (0,1)]
        r=np.concatenate((a[0],b[0]))/np.sqrt(2)
        if not jacobian:return r,None
        J=sp.vstack((sp.csr_matrix(a[1]),sp.csr_matrix(b[1])),format='csr')/np.sqrt(2)
        return r,J


def pack(initial,ids,edges,engine,metadata):
    # Inference-only adapter: the native analytic interface supplies J/g/H.
    # This NumPy-backed raw callback is not an autograd training interface.
    def raw(x,v):return torch.from_numpy(engine.evaluate(x,v,0)['r'])
    p=ResidualProblem(torch.tensor(initial,dtype=torch.float64),torch.tensor(ids,dtype=torch.long),raw,
        engine.cost,lambda x,v:torch.ones_like(raw(x,v)),(None,None),
        metadata={'block_edges':torch.from_numpy(np.array(edges,dtype=np.int64)).reshape(2,-1),'derivatives':'shared analytic NumPy/SciPy sparse factors',**metadata})
    p.native=engine
    return p


def nist_raw(name,b,x):
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        if name=='Chwirut2':
            e=np.exp(-b[0]*x);den=b[1]+b[2]*x;y=e/den
            J=np.column_stack((-x*y,-y/den,-x*y/den))
        elif name=='Rat43':
            from scipy.special import expit
            t=b[1]-b[2]*x;u=np.logaddexp(0,t);z=np.exp(-u/b[3]);y=b[0]*z
            common=-y*expit(t)/b[3]
            J=np.column_stack((z,common,-x*common,y*u/(b[3]*b[3])))
        elif name=='Thurber':
            X=np.column_stack((np.ones_like(x),x,x*x,x*x*x))
            den=1+X[:,1:]@b[4:];y=(X@b[:4])/den
            J=np.column_stack((X/den[:,None],-y[:,None]*X[:,1:]/den[:,None]))
        else:raise ValueError(name)
    return y,J


def nist(spec):
    name=spec['dataset'];data=json.loads((ROOT/'data_stage7/nist_tables.json').read_text())['datasets'][name]
    starts=np.array(data['starts']);scale=np.max(np.abs(starts),axis=0)
    table=np.array(data['yx']);y,x=table.T;yscale=np.sqrt(np.mean(y*y))
    initial=starts[spec['start']]/scale
    if spec.get('jitter'):
        initial=initial*(1+.08*np.random.default_rng(spec['seed']).normal(size=len(initial)))
    partition=spec.get('partition','interleave')
    if partition=='blocked':selected=[np.arange(len(y))<len(y)//2,np.arange(len(y))>=len(y)//2]
    elif partition=='duplicate':selected=[np.ones(len(y),bool)]*2
    else:selected=[np.arange(len(y))%2==v for v in (0,1)]
    def raw_jac(z,v,jac):
        predicted,J=nist_raw(name,z*scale,x[selected[v]])
        multiplier=(1 if partition=='duplicate' else np.sqrt(2))/yscale
        return (predicted-y[selected[v]])*multiplier,(J*scale*multiplier if jac else None)
    engine=AnalyticFactors(len(initial),raw_jac)
    ids=np.arange(len(initial));edges=np.triu_indices(len(initial),1)
    p=pack(initial,ids,edges,engine,{'family':f'nist_{name}','dataset':name,'partition':partition,
        'data_status':'public observed NIST data; starting points are not independent datasets','parameter_scale_source':'maximum absolute value of the two published starts','source':data['url']})
    f0=engine.objective(initial);ref=.5*data['certified_sse']/yscale**2
    def evaluate(z):
        z=np.asarray(z);f=engine.objective(z)
        return {'metric':float(np.sqrt(max(2*f,0)/len(y))),
            'objective_gap_ratio':float((f-ref)/max(f0-ref,1e-30)),
            'sse':float(2*f*yscale**2),'certified_sse':data['certified_sse'],
            'certified_sse_relative_error':float(abs(2*f*yscale**2/data['certified_sse']-1))}
    return p,evaluate


def broyden(spec):
    n=spec['n'];seed=spec['seed'];rng=np.random.default_rng(seed)
    initial=-np.ones(n)+(0 if spec.get('canonical') else .2*rng.normal(size=n))
    selected=[np.arange(n)%2==v for v in (0,1)]
    def raw_jac(z,v,jac):
        r=(3-z)*z+1;r[1:]-=z[:-1];r[:-1]-=2*z[1:]
        J=sp.diags((-np.ones(n-1),3-2*z,-2*np.ones(n-1)),(-1,0,1),format='csr') if jac else None
        if spec.get('partition')=='duplicate':return r,J
        return np.sqrt(2)*r[selected[v]],np.sqrt(2)*J[selected[v]] if jac else None
    engine=AnalyticFactors(n,raw_jac)
    edges=np.array([(i,j) for i in range(n) for j in range(i+1,min(i+3,n))]).T
    p=pack(initial,np.arange(n),edges,engine,{'family':'broyden','n':n,'seed':seed,
        'data_status':'standard synthetic Broyden tridiagonal residual; declared initial perturbations',
        'source':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html'})
    f0=engine.objective(initial)
    def evaluate(z):
        f=engine.objective(np.asarray(z))
        return {'metric':float(np.sqrt(max(2*f,0)/n)),'objective_gap_ratio':float(f/f0)}
    return p,evaluate


def view_graph(spec):
    n=spec['n'];rng=np.random.default_rng(spec['seed'])
    pairs=sorted({(i,j) for stride in (1,3) for i in range(n-stride) for j in (i+stride,)})
    rr=[];cc=[];dd=[]
    for k,(i,j) in enumerate(pairs):rr.extend((k,k));cc.extend((i,j));dd.extend((1.,-1.))
    for i in range(n):rr.append(len(pairs)+i);cc.append(i);dd.append(.1)
    A=sp.csr_matrix((dd,(rr,cc)),shape=(len(pairs)+n,n))
    truth=np.sin(np.linspace(0,2*np.pi,n))+.2*rng.normal(size=n)
    clean=A@truth
    noise=[rng.normal(size=len(clean)) for _ in (0,1)]
    mode=spec['condition'];sigma0=.01;sigma1=.1 if mode=='noisy_view' else .01
    bias=.12*np.sin(np.linspace(0,4*np.pi,len(clean))) if mode=='biased_view' else 0
    observations=[clean+sigma0*noise[0],clean+sigma1*noise[1]+bias]
    def raw_jac(z,v,jac):return A@z-observations[v],A if jac else None
    engine=AnalyticFactors(n,raw_jac)
    p=pack(np.zeros(n),np.arange(n),np.array(pairs).T,engine,{'family':'view_graph','n':n,'seed':spec['seed'],
        'condition':mode,'data_status':'synthetic graph calibration with two views; true parameters withheld from all solvers'})
    f0=engine.objective(np.zeros(n))
    # Scoring reference only: the engine never receives this optimum or truth.
    import scipy.sparse.linalg as spla
    optimum=spla.spsolve(A.T@A,A.T@(.5*(observations[0]+observations[1])))
    fstar=engine.objective(optimum)
    def evaluate(z):
        z=np.asarray(z);f=engine.objective(z)
        return {'metric':float(np.sqrt(np.mean((z-truth)**2))),
            'objective_gap_ratio':float((f-fstar)/max(f0-fstar,1e-30)),
            'irreducible_objective':float(fstar)}
    return p,evaluate


def compile_case(spec):
    f=spec['family']
    if f=='nist':return nist(spec)
    if f=='broyden':return broyden(spec)
    if f=='view_graph':return view_graph(spec)
    from stage5_tasks import compile_case as original
    return original(spec)


def group(spec):return 'nist_'+spec['dataset'] if spec['family']=='nist' else spec['family']
