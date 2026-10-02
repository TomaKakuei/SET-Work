"""Native residual adapters that retain the exact legacy support objectives."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import torch
from torch import Tensor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"legacy"))

@dataclass
class ResidualProblem:
    initial: Tensor
    block_id: Tensor
    raw_function: object
    cost_function: object
    weight_function: object
    legacy_views: tuple
    expand_function: object = lambda x: x
    metadata: dict | None = None

    def raw(self,x,view): return self.raw_function(x,view)
    def cost(self,x,view): return self.cost_function(x,view)
    def weights(self,x,view): return self.weight_function(x,view)
    def expand(self,x): return self.expand_function(x)

def bilinear(image, xy):
    """Border-padded align_corners=True sampling with forward-AD support."""
    h,w = image.shape
    x,y = xy[:,0].clamp(0,w-1),xy[:,1].clamp(0,h-1)
    x0,y0 = torch.floor(x).long(),torch.floor(y).long()
    x1,y1 = (x0+1).clamp_max(w-1),(y0+1).clamp_max(h-1)
    a,b = x-x0,y-y0
    flat = image.reshape(-1)
    return ((1-a)*(1-b)*flat[y0*w+x0]+a*(1-b)*flat[y0*w+x1]
            +(1-a)*b*flat[y1*w+x0]+a*b*flat[y1*w+x1])

def hpatches_problem(observation, reference_image, target_image):
    from neural_grey_v2.hpatches_homography import (
        _local_contrast_normalize,_as_image_tensor,_safe_project,
        homography_from_theta,homography_photometric_factor_graph,HOMOGRAPHY_BLOCK_ID)
    dtype,device = torch.float64,torch.device("cpu")
    kwargs=dict(device=device,dtype=dtype)
    reference = _local_contrast_normalize(_as_image_tensor(reference_image,**kwargs))
    target = _local_contrast_normalize(_as_image_tensor(target_image,**kwargs))
    reference_shape,target_shape = reference_image.shape,target_image.shape
    H0 = torch.as_tensor(observation.initial_homography,**kwargs)
    def decode(x): return homography_from_theta(x[None],H0,reference_shape,target_shape)[0]
    axis = torch.linspace(.08,.92,24,**kwargs)
    yy,xx = torch.meshgrid(axis,axis,indexing="ij")
    source = torch.stack((xx.flatten()*(reference_shape[1]-1),
                          yy.flatten()*(reference_shape[0]-1)),1)
    def project(x): return _safe_project(source,decode(x)[None])[0]
    initial = torch.zeros(8,**kwargs)
    projected = project(initial)
    valid = ((2*projected[:,0]/(target_shape[1]-1)-1).abs()<=.98)&(
             (2*projected[:,1]/(target_shape[0]-1)-1).abs()<=.98)
    ref_levels,target_levels = [],[]
    for kernel in (15,7,1):
        ref = reference if kernel==1 else torch.nn.functional.avg_pool2d(reference,kernel,1,kernel//2)
        tgt = target if kernel==1 else torch.nn.functional.avg_pool2d(target,kernel,1,kernel//2)
        ref_levels.append(bilinear(ref[0,0],source).detach())
        target_levels.append(tgt[0,0].detach())
    view_sqrt_weights=[]
    for view in (0,1):
        per_level=[]
        for level,weight in enumerate((.75,.20,.05)):
            coverage = initial.new_zeros(24,24)
            offset=21*level
            axis_i = torch.arange(24)
            for cells in (1,2,4):
                region = (axis_i[:,None]*cells//24)*cells+(axis_i[None,:]*cells//24)
                coverage += ((region+offset)%2==view).to(dtype)/3.
                offset += cells*cells
            per_level.append((weight*coverage.flatten()*valid).sqrt())
        view_sqrt_weights.append(torch.stack(per_level))
    def raw(x,view):
        xy=project(x)
        errors=torch.stack([bilinear(im,xy)-ref for im,ref in zip(target_levels,ref_levels)])
        return (errors*view_sqrt_weights[view]).reshape(-1,1)
    def cost(x,view): return .5*raw(x,view).square().sum()
    def weights(x,view): return x.new_ones(1728,1)
    old,_,_,_ = homography_photometric_factor_graph(
        observation,reference_image,target_image,**kwargs)
    old_views=(lambda x: old(x)[:,0::2],lambda x: old(x)[:,1::2])
    problem=ResidualProblem(initial,HOMOGRAPHY_BLOCK_ID.clone(),raw,cost,weights,old_views,
                            metadata={"family":"hpatches","gauge":"H33 normalized",
                                      "residuals_per_view":1728,"factor_views":"legacy even/odd multiscale regions"})
    return problem,decode

def ba_problem(graph):
    # Camera zero removes SE3 gauge; one INITIAL landmark depth fixes monocular scale.
    full_initial=graph.initial_theta[0].double()
    anchor_index=6*(graph.camera_count-1)+2
    free=torch.arange(full_initial.numel())!=anchor_index
    indices=free.nonzero().flatten()
    unique,ids=torch.unique(graph.block_id[free],sorted=True,return_inverse=True)
    def expand(x):
        if x.ndim==1:
            return full_initial.index_copy(0,indices,x)
        return full_initial[None].expand(x.shape[0],-1).index_copy(1,indices,x)
    cameras=graph.observation_camera
    landmarks=graph.observation_landmark
    selected=[(cameras%2==v).nonzero().flatten() for v in (0,1)]
    def raw(x,view):
        rotations,translations,points=graph.decode(expand(x)[None])
        ci,li=cameras[selected[view]],landmarks[selected[view]]
        cp=torch.einsum("oij,oj->oi",rotations[0,ci].transpose(-1,-2),points[0,li]-translations[0,ci])
        uv=cp[:,:2]/cp[:,2:].clamp_min(.25)
        return uv-graph.observed_uv[selected[view]].to(x)
    delta=graph.reprojection_delta
    def cost(x,view):
        s=raw(x,view).square().sum(-1)
        return (s/(torch.sqrt(1+s/(delta*delta))+1)).sum()
    def weights(x,view):
        return torch.rsqrt(1+raw(x,view).square().sum(-1,keepdim=True)/(delta*delta))
    old=[graph.factor_function(view=v,granularity="landmark") for v in ("even","odd")]
    legacy=tuple((lambda x,fn=fn: fn(expand(x))) for fn in old)
    return ResidualProblem(full_initial[free],ids,raw,cost,weights,legacy,expand,
                           {"family":"synthetic_ba","gauge":"camera0 fixed; landmark0 initial depth fixed",
                            "fixed_coordinate":anchor_index,"robust":"frozen IRLS pseudo-Huber"})

def se3_problem(graph):
    initial=graph.initial_theta[0].double()
    edges=graph.edges
    selected=[torch.arange(graph.source_points.shape[1])%2==v for v in (0,1)]
    def raw(x,view):
        R,t=graph.decode(x[None])
        source=graph.source_points[:,selected[view]].to(x)
        target=graph.target_points[:,selected[view]].to(x)
        anchor=torch.einsum("eij,ekj->eki",R[0,edges[:,0]],source)+t[0,edges[:,0],None]
        pred=torch.einsum("eji,ekj->eki",R[0,edges[:,1]],anchor-t[0,edges[:,1],None])
        return (pred-target).reshape(-1,3)
    masks=[graph.correspondence_mask[:,s].reshape(-1,1).double() for s in selected]
    delta=graph.huber_delta_m
    def cost(x,view):
        s=raw(x,view).square().sum(-1,keepdim=True)
        return (masks[view]*s/(torch.sqrt(1+s/(delta*delta))+1)).sum()
    def weights(x,view):
        return masks[view]*torch.rsqrt(1+raw(x,view).square().sum(-1,keepdim=True)/(delta*delta))
    legacy=tuple(graph.factor_function(view=v,granularity="node") for v in ("even","odd"))
    return ResidualProblem(initial,graph.block_id.clone(),raw,cost,weights,legacy,
                           metadata={"family":"se3","gauge":"first pose fixed","robust":"frozen IRLS pseudo-Huber"})

def contract_check(problem,perturbation=0.001):
    generator=torch.Generator().manual_seed(351)
    worst_cost=worst_gradient=worst_hvp=0.
    from .core import GNView,initial_counters
    for k in range(2):
        x=problem.initial+torch.randn(problem.initial.shape,generator=generator,dtype=torch.float64)*perturbation*k
        for view in (0,1):
            old=problem.legacy_views[view](x[None]).sum()
            native=problem.cost(x,view)
            worst_cost=max(worst_cost,float(abs(old-native)/old.abs().clamp_min(1e-10)))
            expected=torch.func.grad(lambda z: problem.cost(z,view))(x)
            op=GNView(problem,x,view,1e-3,initial_counters())
            worst_gradient=max(worst_gradient,float((expected-op.g).norm()/expected.norm().clamp_min(1e-12)))
            direction=torch.randn(x.shape,generator=generator,dtype=x.dtype)
            J=op.full_jacobian()
            reference=J.T@(J@direction)+1e-3*direction
            worst_hvp=max(worst_hvp,float((op(direction)-reference).norm()/reference.norm().clamp_min(1e-12)))
    return dict(objective_relative_error=worst_cost,gradient_relative_error=worst_gradient,
                gn_action_relative_error=worst_hvp)


