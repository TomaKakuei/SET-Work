"""Checkpoint-compatible inference with fused C feature/proposal construction."""
import torch
from .model_stage4 import NormalizedSchurNet as ReferenceNet
from .model_stage2 import _neighbor_mean
from .native_stage5 import features,proposal


class NormalizedSchurNet(ReferenceNet):
    def carriers_and_features(self,base,ga,gb,ids,edges):
        if torch.is_grad_enabled() or ga.device.type!='cpu':
            return super().carriers_and_features(base,ga,gb,ids,edges)
        if self.max_probes!=8 or self.carriers!=20:raise ValueError('C feature schema requires the frozen Stage 4 architecture')
        return features(base,ga,gb,ids,edges)

    def forward(self,base,ga,gb,ids,edges,*,directions=8):
        if torch.is_grad_enabled() or ga.device.type!='cpu':
            return super().forward(base,ga,gb,ids,edges,directions=directions)
        if not 1<=directions<=8:raise ValueError('Direction budget outside trained capacity')
        carriers,feature=self.carriers_and_features(base,ga,gb,ids,edges)
        hidden=self.embed(feature)
        for layer in self.message:
            hidden=hidden+layer(torch.cat((hidden,_neighbor_mean(hidden,edges)),dim=1))
        raw=self.head(hidden).reshape(-1,8,41)
        return proposal(carriers,raw,ids,directions)
