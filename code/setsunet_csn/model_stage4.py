"""Dimensionless graph proposals carrying every measured view response."""
import torch
from torch import nn
from .model_stage2 import _segment_sum, _neighbor_mean


class NormalizedSchurNet(nn.Module):
    def __init__(self,hidden=48,max_directions=8,max_probes=8,message_rounds=2):
        super().__init__()
        self.hidden,self.max_directions,self.max_probes=hidden,max_directions,max_probes
        self.message_rounds=message_rounds
        self.carriers=4+2*max_probes
        self.feature_dim=self.carriers**2+4+4*max_probes
        self.embed=nn.Sequential(nn.Linear(self.feature_dim,hidden),nn.LayerNorm(hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.SiLU())
        self.message=nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.LayerNorm(hidden),nn.SiLU()) for _ in range(message_rounds))
        self.head=nn.Linear(hidden,max_directions*(1+2*self.carriers))

    @staticmethod
    def normalized(V):
        return V / torch.linalg.vector_norm(V,dim=0,keepdim=True).clamp_min(torch.finfo(V.dtype).tiny**.25)

    def carriers_and_features(self,base,ga,gb,ids,edges):
        if base['Q'].shape[1]>self.max_probes:
            raise ValueError('Measured probes exceed trained feature capacity; use an explicit probe-window adapter')
        n,blocks=ga.numel(),int(ids.max())+1
        zero=torch.zeros_like(ga)
        vectors=[base['e'],ga-gb,.5*(ga+gb),base.get('history',zero)]
        for Y in (base['Ya'],base['Yb']):
            vectors += [Y[:,j] if j<Y.shape[1] else zero for j in range(self.max_probes)]
        carriers=self.normalized(torch.stack(vectors,dim=1))
        gram=_segment_sum((carriers[:,:,None]*carriers[:,None,:]).flatten(1),ids,blocks)
        sizes=torch.bincount(ids,minlength=blocks).to(ga)
        energy=_segment_sum(carriers[:,0].square()[:,None],ids,blocks)[:,0]
        degrees=ga.new_zeros(blocks)
        if edges.numel(): degrees.index_add_(0,edges.flatten(),torch.ones_like(edges.flatten(),dtype=ga.dtype))
        scalars=torch.stack((sizes/n,energy,degrees/max(blocks-1,1),(degrees>0).to(ga.dtype)),dim=1)
        summaries=ga.new_zeros(blocks,4*self.max_probes)
        Ya,Yb=base['Ya'],base['Yb']
        if Ya.shape[1]:
            # One shared scale per probe preserves view disagreement while
            # removing common objective-unit scaling from all four summaries.
            scales=(.5*(Ya.square()+Yb.square()).sum(dim=0)).sqrt().clamp_min(torch.finfo(ga.dtype).tiny**.25)
            ya,yb=Ya/scales,Yb/scales
            q=base['Q']
            per=torch.stack((q*.5*(ya+yb),ya.square(),yb.square(),ya*yb),dim=2).flatten(1)
            summaries[:,:per.shape[1]]=_segment_sum(per,ids,blocks)
        return carriers,torch.cat((gram,scalars,summaries),dim=1)

    def forward(self,base,ga,gb,ids,edges,*,directions=4):
        if directions>self.max_directions: raise ValueError('Requested directions exceed trained head capacity')
        carriers,features=self.carriers_and_features(base,ga,gb,ids,edges)
        hidden=self.embed(features)
        for layer in self.message:
            hidden=hidden+layer(torch.cat((hidden,_neighbor_mean(hidden,edges)),dim=1))
        raw=self.head(hidden).reshape(-1,self.max_directions,1+2*self.carriers)[:,:directions]
        local=raw[:,:,1:1+self.carriers]
        global_coeff=raw[:,:,1+self.carriers:].mean(dim=0,keepdim=True)
        # The solver measures and optimizes coefficients. An SPD map of e is
        # unnecessary and restricts expressivity, so propose the full carrier mix.
        V=(carriers[:,None,:]*(local[ids]+global_coeff)).sum(dim=2)
        V=V+torch.nn.functional.softplus(raw[:,:,0])[ids]*carriers[:,0:1]
        return self.normalized(V)

    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

    @property
    def architecture(self):
        return dict(hidden=self.hidden,max_directions=self.max_directions,max_probes=self.max_probes,message_rounds=self.message_rounds)
