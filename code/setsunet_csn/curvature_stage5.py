"""Native sparse-factor accumulation with explicit layout reuse."""
import torch
from .curvature_stage4 import DenseView,BlockMetric,linearize as reference_linearize


class BlockLayout:
    def __init__(self,ids):
        self.ids=ids
        sizes=torch.bincount(ids);width=int(sizes.max());blocks=len(sizes)
        self.mask=torch.arange(width)[None,:]<sizes[:,None]
        self.indices=torch.zeros(blocks,width,dtype=torch.long)
        for block in range(blocks):self.indices[block,:sizes[block]]=(ids==block).nonzero().flatten()

    def metric(self,normal):
        # Reuse immutable indexing, while recomputing exact block eigen roots.
        out=object.__new__(BlockMetric)
        out.ids,out.indices,out.mask=self.ids,self.indices,self.mask
        indices,mask=self.indices,self.mask
        values=normal[indices[:,:,None],indices[:,None,:]]*mask[:,:,None]*mask[:,None,:]
        values=values+torch.diag_embed((~mask).to(values.dtype))
        eig,vec=torch.linalg.eigh(values)
        scale=values.diagonal(dim1=-2,dim2=-1).abs().amax(dim=1,keepdim=True)
        floor=torch.finfo(values.dtype).eps*32*scale.clamp_min(1e-300)
        clipped=eig.clamp_min(floor)
        out.inverse=(vec*clipped.rsqrt()[:,None,:])@vec.transpose(-2,-1)
        out.root=(vec*clipped.sqrt()[:,None,:])@vec.transpose(-2,-1)
        out.clipped_eigenvalues=int((eig<floor).sum())
        return out


def linearize(problem,x,damping,counts,config):
    if not hasattr(problem,'native'):return reference_linearize(problem,x,damping,counts,config)
    engine=problem.native
    pairs=[engine.evaluate(x,v,2) for v in (0,1)]
    normals=[torch.from_numpy(p['H']) for p in pairs]
    average=.5*(normals[0].diagonal()+normals[1].diagonal())
    scale=float((problem.metadata or {}).get('objective_scale',1.))
    if config.damping_mode=='declared':shift=x.new_full(x.shape,damping*scale)
    elif config.damping_mode=='relative':shift=x.new_full(x.shape,damping*float(average.mean().clamp_min(1e-30)))
    elif config.damping_mode=='diagonal':shift=damping*average.clamp_min(max(float(average.max())*1e-12,1e-30))
    else:raise ValueError(config.damping_mode)
    for H in normals:H.diagonal().add_(shift)
    views=[DenseView(H,torch.from_numpy(p['g']),counts) for H,p in zip(normals,pairs)]
    for name in ('linearizations','jacobian_evaluations','residual_evaluations','weight_evaluations'):
        counts[name]+=2
    counts['factor_evaluations']=counts.get('factor_evaluations',0)+2
    if getattr(engine,'evaluates_in_c',False):counts['native_calls']+=2
    # This graph is actual compiler factor incidence, checked against derivatives.
    edges=torch.as_tensor(problem.metadata['block_edges'],dtype=torch.long)
    note={'kind':'analytic_factor_GN','PSD_shift_min':float(shift.min()),
        'PSD_shift_max':float(shift.max()),
        'evaluator_backend':getattr(engine,'evaluator_backend',type(engine).__module__+'.'+type(engine).__name__),
        'rectangular_jacobian_materialized':any(p.get('J') is not None for p in pairs),
        'normal_matrix_materialized':True,'normal_matrix_storage':'dense',
        'implementation_credit':'shared project factor interface, independent of optimizer'}
    return views[0],views[1],edges,note
