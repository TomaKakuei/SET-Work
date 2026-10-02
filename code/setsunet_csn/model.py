"""A shared block rule proposes SPD residual transformations, never a step gain."""
from __future__ import annotations
import torch
from torch import nn

class SchurProposalNet(nn.Module):
    def __init__(self, hidden=32, directions=2, carriers=4):
        super().__init__()
        self.directions, self.carriers, self.hidden = directions, carriers, hidden
        self.net = nn.Sequential(nn.Linear(carriers*carriers+2,hidden),nn.Tanh(),
                                 nn.Linear(hidden,hidden),nn.Tanh(),
                                 nn.Linear(hidden,directions*(1+2*carriers)))

    def forward(self, base, ga, gb, block_id):
        e = base["e"]
        Ya, Yb = base["Ya"], base["Yb"]
        ya = Ya[:,0] if Ya.shape[1] else ga
        yb = Yb[:,0] if Yb.shape[1] else gb
        carriers = torch.stack((e,ga-gb,(ya+yb)*.5,ya-yb),1)
        carriers = carriers/carriers.norm(dim=0,keepdim=True).clamp_min(1e-15)
        ids = block_id.to(e.device)
        nblocks = int(ids.max())+1
        gram_elements = (carriers[:,:,None]*carriers[:,None,:]).reshape(e.numel(),-1)
        grams = e.new_zeros(nblocks,self.carriers*self.carriers).index_add(0,ids,gram_elements)
        sizes = torch.bincount(ids,minlength=nblocks).to(e)
        energy = e.new_zeros(nblocks).index_add(0,ids,e.square())/e.square().sum().clamp_min(1e-30)
        features = torch.cat((grams,torch.log1p(sizes)[:,None],
                              torch.log1p(energy*sizes.sum())[:,None]),1)
        parameters = self.net(features).reshape(nblocks,self.directions,1+2*self.carriers)
        a = torch.nn.functional.softplus(parameters[:,:,0])+1e-3
        coefficients_t = parameters[:,:,1:1+self.carriers]
        coefficients_u = parameters[:,:,1+self.carriers:]
        T = (carriers[:,None,:]*coefficients_t[ids]).sum(-1)
        U = (carriers[:,None,:]*coefficients_u[ids]).sum(-1)
        local_inner = e.new_zeros(nblocks,self.directions).index_add(0,ids,T*e[:,None])
        global_inner = (U*e[:,None]).sum(0,keepdim=True)
        return a[ids]*e[:,None]+T*local_inner[ids]+U*global_inner

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

