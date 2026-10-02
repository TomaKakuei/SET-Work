"""Identified curvature with bounded AD batches and reusable frozen operators."""
from dataclasses import dataclass
import torch
from .core import sym


@dataclass
class CurvatureConfig:
    ad_chunk: int = 128
    dense_limit: int = 2048
    jacobian_cache_elements: int = 8_000_000
    scalar_curvature: str = "exact_hessian"
    damping_mode: str = "declared"


def graph_from_derivatives(jacobians, ids, normals=None):
    """Observed incidence; never fabricate adjacency from numeric block labels."""
    blocks=int(ids.max())+1
    adjacency=torch.zeros(blocks,blocks,dtype=torch.bool,device=ids.device)
    if jacobians:
        for J in jacobians:
            threshold=torch.finfo(J.dtype).eps*32*J.abs().amax(dim=1,keepdim=True).clamp_min(1e-300)
            active=J.new_zeros(J.shape[0],blocks).index_add(1,ids,(J.abs()>threshold).to(J.dtype))>0
            adjacency |= active.to(J.dtype).T @ active.to(J.dtype)>0
    else:
        for H in normals:
            active=H.abs()>torch.finfo(H.dtype).eps*32*H.abs().max().clamp_min(1e-300)
            reduced=H.new_zeros(H.shape[0],blocks).index_add(1,ids,active.to(H.dtype))
            adjacency |= H.new_zeros(blocks,blocks).index_add(0,ids,reduced)>0
    return torch.triu(adjacency,diagonal=1).nonzero().T


class DenseView:
    def __init__(self,H,g,counts):
        self.H,self.g,self.counts=H,g,counts
    def columns(self,V):
        self.counts['curvature_products']+=V.shape[1]
        self.counts['cached_matrix_products']+=V.shape[1]
        return self.H@V
    def __call__(self,v): return self.columns(v[:,None])[:,0]


class NativeView:
    def __init__(self,problem,x,view,damping,ids,counts,config):
        self.counts,self.damping,self.x=counts,damping,x
        weights=problem.weights(x,view).detach().sqrt()
        counts['weight_evaluations']+=1
        if (problem.metadata or {}).get('robust'): counts['residual_evaluations']+=1
        self.residual=lambda z:(problem.raw(z,view)*weights).reshape(-1)
        counts['residual_evaluations']+=1
        self.compatibility=False
        try:
            self.r,self.pullback=torch.func.vjp(self.residual,x)
            self.g=self.pullback(self.r)[0].detach()
            counts['vjp_directions']+=1
        except RuntimeError as exc:
            if 'functorch' not in str(exc) and 'setup_context' not in str(exc): raise
            self.compatibility=True
            z=x.detach().requires_grad_(True)
            r=self.residual(z)
            self.r=r.detach()
            self.g=torch.autograd.grad(.5*r.square().sum(),z)[0].detach()
            counts['vjp_directions']+=1
        self.J=None
        n=x.numel()
        cache=self.r.numel()*n<=config.jacobian_cache_elements
        # A dense normal is useful for small/medium systems, independently of
        # whether retaining the rectangular Jacobian is worthwhile.
        self.H=None
        if cache:
            self.J=self._jacobian(config.ad_chunk)
            if n<=config.dense_limit:
                self.H=sym(self.J.T@self.J)+damping*torch.eye(n,dtype=x.dtype,device=x.device)
        elif n<=config.dense_limit:
            self.H=x.new_zeros(n,n)
            # Streaming residual rows prevent an m*n retained Jacobian when
            # a dense n*n normal is affordable. Every reverse direction counts.
            for first in range(0,self.r.numel(),config.ad_chunk):
                last=min(first+config.ad_chunk,self.r.numel())
                fn=lambda z,lo=first,hi=last:self.residual(z)[lo:hi]
                if self.compatibility:
                    block=torch.autograd.functional.jacobian(fn,x,vectorize=False).detach()
                else:
                    block=torch.func.jacrev(fn,chunk_size=config.ad_chunk)(x).detach()
                self.H+=block.T@block
                counts['vjp_directions']+=last-first
                counts['residual_evaluations']+=1
            counts['jacobian_evaluations']+=1
            self.H=sym(self.H)+damping*torch.eye(n,dtype=x.dtype,device=x.device)
        self.local_blocks=None
        if self.H is None:
            local=[]
            for block in range(int(ids.max())+1):
                indices=(ids==block).nonzero().flatten()
                if self.J is not None: Jlocal=self.J[:,indices]
                else:
                    parts=[]
                    for selected in indices.split(config.ad_chunk):
                        V=x.new_zeros(n,len(selected))
                        V[selected,torch.arange(len(selected),device=x.device)]=1
                        parts.append(self.jvp(V))
                    Jlocal=torch.cat(parts,dim=1)
                local.append(sym(Jlocal.T@Jlocal)+damping*torch.eye(len(indices),dtype=x.dtype,device=x.device))
            self.local_blocks=local
        counts['linearizations']+=1

    def jvp(self,V):
        if self.J is not None: return self.J@V
        try:
            result=torch.vmap(lambda v:torch.func.jvp(self.residual,(self.x,),(v,))[1])(V.T).T
            self.counts['jvp_directions']+=V.shape[1]
            self.counts['residual_evaluations']+=1
            return result.detach()
        except (NotImplementedError,RuntimeError) as exc:
            text=str(exc)
            if not any(s in text for s in ('forward','functorch','setup_context','vmap','not implemented')): raise
            # Reverse fallback is charged and bounded by output chunks.
            if self.J is None:
                self.J=self._reverse_jacobian(16)
            return self.J@V

    def _reverse_jacobian(self,chunk):
        self.counts['jacobian_evaluations']+=1
        self.counts['vjp_directions']+=self.r.numel()
        if self.compatibility:
            return torch.autograd.functional.jacobian(self.residual,self.x,vectorize=False).detach()
        return torch.func.jacrev(self.residual,chunk_size=chunk)(self.x).detach()

    def _jacobian(self,chunk):
        n=self.x.numel()
        if self.compatibility or self.r.numel()<n:
            return self._reverse_jacobian(chunk)
        self.counts['jacobian_evaluations']+=1
        pieces=[]
        for first in range(0,n,chunk):
            width=min(chunk,n-first)
            V=self.x.new_zeros(n,width)
            V[first+torch.arange(width),torch.arange(width)]=1
            pieces.append(self.jvp(V))
        return torch.cat(pieces,dim=1)

    def columns(self,V):
        self.counts['curvature_products']+=V.shape[1]
        if self.H is not None:
            self.counts['cached_matrix_products']+=V.shape[1]
            return self.H@V
        if self.J is not None:
            self.counts['cached_matrix_products']+=V.shape[1]
            return self.J.T@(self.J@V)+self.damping*V
        JV=self.jvp(V)
        out=torch.vmap(lambda v:self.pullback(v)[0])(JV.T).T
        self.counts['vjp_directions']+=V.shape[1]
        return out+self.damping*V
    def __call__(self,v): return self.columns(v[:,None])[:,0]


def scalar_hessian(problem,x,view,counts,chunk):
    """Exact objective Hessian, not GN of sqrt(aggregated scalar costs)."""
    z=x.detach().requires_grad_(True)
    objective=problem.cost(z,view)
    g=torch.autograd.grad(objective,z,create_graph=True)[0]
    rows=[]
    n=x.numel()
    for first in range(0,n,chunk):
        width=min(chunk,n-first)
        vectors=x.new_zeros(width,n)
        vectors[torch.arange(width),first+torch.arange(width)]=1
        try:
            part=torch.autograd.grad(g,z,grad_outputs=vectors,is_grads_batched=True,retain_graph=True)[0]
        except (RuntimeError,NotImplementedError):
            part=torch.stack([torch.autograd.grad(g,z,grad_outputs=v,retain_graph=True)[0] for v in vectors])
        rows.append(part.detach())
    counts['hessian_vector_products']+=n
    counts['objective_evaluations']+=1
    counts['residual_evaluations']+=1
    counts['linearizations']+=1
    return sym(torch.cat(rows)),g.detach()


class BlockMetric:
    """Batched symmetric block roots: no numbering- or basis-dependent Cholesky map."""
    def __init__(self,ids,normal=None,local=None):
        self.ids=ids
        blocks=int(ids.max())+1
        sizes=torch.bincount(ids,minlength=blocks)
        width=int(sizes.max())
        indices=torch.zeros(blocks,width,dtype=torch.long,device=ids.device)
        mask=torch.arange(width,device=ids.device)[None,:]<sizes[:,None]
        for block in range(blocks): indices[block,:sizes[block]]=(ids==block).nonzero().flatten()
        self.indices,self.mask=indices,mask
        if normal is not None:
            values=normal[indices[:,:,None],indices[:,None,:]]*mask[:,:,None]*mask[:,None,:]
        else:
            values=local[0].new_zeros(blocks,width,width)
            for block,H in enumerate(local): values[block,:H.shape[0],:H.shape[0]]=H
        # Padded coordinates are independent identity blocks and never scatter back.
        values=values+torch.diag_embed((~mask).to(values.dtype))
        eig,vec=torch.linalg.eigh(values)
        scale=values.diagonal(dim1=-2,dim2=-1).abs().amax(dim=1,keepdim=True)
        floor=torch.finfo(values.dtype).eps*32*scale.clamp_min(1e-300)
        clipped=eig.clamp_min(floor)
        self.inverse=(vec*clipped.rsqrt()[:,None,:])@vec.transpose(-2,-1)
        self.root=(vec*clipped.sqrt()[:,None,:])@vec.transpose(-2,-1)
        self.clipped_eigenvalues=int((eig<floor).sum())
    def apply(self,value,inverse=True):
        vector=value.ndim==1
        V=value[:,None] if vector else value
        gathered=V[self.indices]*self.mask[:,:,None]
        output=(self.inverse if inverse else self.root)@gathered
        output=output*self.mask[:,:,None]
        result=torch.zeros_like(V).index_add(0,self.indices.flatten(),output.reshape(-1,V.shape[1]))
        return result[:,0] if vector else result
    def normal(self,H): return self.apply(self.apply(H).T).T


def linearize(problem,x,damping,counts,config=None):
    config=config or CurvatureConfig()
    ids=problem.block_id.to(x.device)
    scalar='scalarized' in str((problem.metadata or {}).get('geometry',''))
    objective_scale=float((problem.metadata or {}).get('objective_scale',1.))
    if not objective_scale>0: raise ValueError('objective_scale must be positive')
    if config.damping_mode not in ('declared','relative'): raise ValueError(config.damping_mode)
    if scalar:
        if config.scalar_curvature!='exact_hessian': raise ValueError('Scalar factor costs require explicit objective-Hessian curvature')
        pairs=[scalar_hessian(problem,x,v,counts,config.ad_chunk) for v in (0,1)]
        min_eig=min(float(torch.linalg.eigvalsh(H).min()) for H,g in pairs)
        scale=sum(float(H.diagonal().abs().mean()) for H,g in pairs)*.5
        regularizer=damping*(max(scale,1e-30) if config.damping_mode=='relative' else objective_scale)
        shift=max(regularizer,regularizer-min_eig)
        views=[DenseView(H+shift*torch.eye(x.numel(),dtype=x.dtype,device=x.device),g,counts) for H,g in pairs]
        edges=graph_from_derivatives([],ids,[v.H for v in views])
        note={'kind':'exact_objective_hessian','PSD_shift':shift,'unshifted_min_eigenvalue':min_eig}
    else:
        views=[NativeView(problem,x,v,0.,ids,counts,config) for v in (0,1)]
        traces=[float(v.H.diagonal().sum()) if v.H is not None else sum(float(H.diagonal().sum()) for H in v.local_blocks) for v in views]
        regularizer=damping*(max(sum(traces)/(2*x.numel()),1e-30) if config.damping_mode=='relative' else objective_scale)
        for v in views:
            v.damping=regularizer
            if v.H is not None: v.H=v.H+regularizer*torch.eye(x.numel(),dtype=x.dtype,device=x.device)
            if v.local_blocks is not None:
                v.local_blocks=[H+regularizer*torch.eye(H.shape[0],dtype=x.dtype,device=x.device) for H in v.local_blocks]
        jacobians=[v.J for v in views if v.J is not None]
        if jacobians:
            edges=graph_from_derivatives(jacobians,ids)
        elif all(v.H is not None for v in views):
            edges=graph_from_derivatives([],ids,[v.H for v in views])
        else:
            declared=(problem.metadata or {}).get('block_edges')
            if declared is None: raise ValueError('Matrix-free problems must supply actual factor incidence; no chain fallback')
            edges=torch.as_tensor(declared,dtype=torch.long,device=x.device)
        note={'kind':'native_vector_GN','PSD_shift':regularizer,'jacobians_cached':sum(v.J is not None for v in views)}
    return views[0],views[1],edges,note
