"""Exact stencil-color Hessian recovery for the frozen SIDD scalar objective.

The image Hessian couples channels at the same pixel and pixels in the same
channel at Chebyshev distance at most one. Two noise latents are dense columns.
27 image colors and two latent directions recover the full 770x770 Hessian.
No curvature approximation, objective change or learned parameter is used.
"""
import torch,inspect
from setsunet_csn.core import sym
from setsunet_csn import curvature_stage4 as original
from setsunet_csn import curvature_stage5 as stage5_curvature

REFERENCE=original.scalar_hessian
REFERENCE_LINEARIZE=original.linearize
CACHE={}

def structure(x):
    key=(x.dtype,str(x.device))
    if key in CACHE:return CACHE[key]
    n=770;side=16;count=768
    color=torch.arange(count,device=x.device)
    channels=color//256;y=(color%256)//16;xx=color%16
    colors=channels*9+(y%3)*3+xx%3
    vectors=x.new_zeros(29,n)
    vectors[colors,torch.arange(count,device=x.device)]=1
    vectors[27,768]=1;vectors[28,769]=1
    rows=[];cols=[]
    for c in range(3):
        for y in range(16):
            for q in range(16):
                i=c*256+y*16+q
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if 0<=y+dy<16 and 0<=q+dx<16:
                            rows.append(i);cols.append(c*256+(y+dy)*16+q+dx)
                for other in range(3):
                    if other!=c:rows.append(i);cols.append(other*256+y*16+q)
    rows=torch.tensor(rows,device=x.device);cols=torch.tensor(cols,device=x.device)
    # Within each row, every potentially nonzero image column has unique color.
    assert len(torch.unique(rows*27+colors[cols]))==len(rows)
    CACHE[key]=(vectors,rows,cols,colors[cols]);return CACHE[key]

def scalar_hessian(problem,x,view,counts,chunk):
    if (problem.metadata or {}).get('family')!='sidd_noise' or x.numel()!=770:
        return REFERENCE(problem,x,view,counts,chunk)
    z=x.detach().requires_grad_(True)
    objective=problem.cost(z,view)
    g=torch.autograd.grad(objective,z,create_graph=True)[0]
    vectors,rows,cols,colors=structure(x)
    packed=torch.autograd.grad(g,z,grad_outputs=vectors,is_grads_batched=True)[0].detach()
    H=x.new_zeros(770,770)
    H[rows,cols]=packed[colors,rows]
    H[:,768:]=packed[27:,:].T
    H[768:,:]=packed[27:,:]
    counts['hessian_vector_products']+=29
    counts['objective_evaluations']+=1
    counts['residual_evaluations']+=1
    counts['linearizations']+=1
    return sym(H),g.detach()

_source=inspect.getsource(REFERENCE_LINEARIZE)
_old='        min_eig=min(float(torch.linalg.eigvalsh(H).min()) for H,g in pairs)'
assert _source.count(_old)==1
_source=_source.replace(_old,'''        positive=all(int(torch.linalg.cholesky_ex(H,check_errors=False).info)==0 for H,g in pairs)
        min_eig=0. if positive else min(float(torch.linalg.eigvalsh(H).min()) for H,g in pairs)''')
_source=_source.replace("'unshifted_min_eigenvalue':min_eig}","'unshifted_min_eigenvalue':None if positive else min_eig,'positive_cholesky_shortcut':positive}")
_namespace=original.__dict__.copy();_namespace['scalar_hessian']=scalar_hessian
exec(compile(_source,'<sidd_exact_linearization>','exec'),_namespace)
_FAST_LINEARIZE=_namespace['linearize']

def linearize(problem,*args,**kwargs):
    if (problem.metadata or {}).get('family')=='sidd_noise' and problem.initial.numel()==770:
        return _FAST_LINEARIZE(problem,*args,**kwargs)
    return REFERENCE_LINEARIZE(problem,*args,**kwargs)

def install():
    original.scalar_hessian=scalar_hessian
    stage5_curvature.reference_linearize=linearize

def restore():
    original.scalar_hessian=REFERENCE
    stage5_curvature.reference_linearize=REFERENCE_LINEARIZE
