"""Exact nine-color Hessian recovery for the fixed 32x48 stereo stencil."""
import torch,inspect
import sidd_colored_hessian as sidd
from setsunet_csn import curvature_stage4 as original,curvature_stage5 as stage5
from setsunet_csn.core import sym
REFERENCE=sidd.REFERENCE
REFERENCE_LINEARIZE=sidd.REFERENCE_LINEARIZE
CACHE={}

def hessian(problem,x,view,counts,chunk):
    if (problem.metadata or {}).get('family')!='middlebury_stereo' or x.numel()!=1536:
        return sidd.scalar_hessian(problem,x,view,counts,chunk)
    key=(x.dtype,str(x.device))
    if key not in CACHE:
        ids=torch.arange(1536,device=x.device);colors=((ids//48)%3)*3+ids%48%3
        vectors=x.new_zeros(9,1536);vectors[colors,ids]=1;rows=[];cols=[]
        for y in range(32):
            for q in range(48):
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if 0<=y+dy<32 and 0<=q+dx<48:rows.append(y*48+q);cols.append((y+dy)*48+q+dx)
        rows=torch.tensor(rows,device=x.device);cols=torch.tensor(cols,device=x.device)
        assert len(torch.unique(rows*9+colors[cols]))==len(rows)
        CACHE[key]=(vectors,rows,cols,colors[cols])
    vectors,rows,cols,colors=CACHE[key]
    z=x.detach().requires_grad_(True);cost=problem.cost(z,view);g=torch.autograd.grad(cost,z,create_graph=True)[0]
    packed=torch.autograd.grad(g,z,grad_outputs=vectors,is_grads_batched=True)[0].detach()
    H=x.new_zeros(1536,1536);H[rows,cols]=packed[colors,rows]
    counts['hessian_vector_products']+=9;counts['objective_evaluations']+=1;counts['residual_evaluations']+=1;counts['linearizations']+=1
    return sym(H),g.detach()

source=inspect.getsource(REFERENCE_LINEARIZE)
old='        min_eig=min(float(torch.linalg.eigvalsh(H).min()) for H,g in pairs)'
assert source.count(old)==1
source=source.replace(old,'''        positive=all(int(torch.linalg.cholesky_ex(H,check_errors=False).info)==0 for H,g in pairs)
        min_eig=0. if positive else min(float(torch.linalg.eigvalsh(H).min()) for H,g in pairs)''')
source=source.replace("'unshifted_min_eigenvalue':min_eig}","'unshifted_min_eigenvalue':None if positive else min_eig,'positive_cholesky_shortcut':positive}")
namespace=original.__dict__.copy();namespace['scalar_hessian']=hessian
exec(compile(source,'<stereo_exact_linearization>','exec'),namespace)
FAST=namespace['linearize']

def linearize(problem,*args,**kwargs):
    if (problem.metadata or {}).get('family')=='middlebury_stereo' and problem.initial.numel()==1536:return FAST(problem,*args,**kwargs)
    return sidd.linearize(problem,*args,**kwargs)

def install():
    original.scalar_hessian=hessian;stage5.reference_linearize=linearize

def restore():sidd.install()
