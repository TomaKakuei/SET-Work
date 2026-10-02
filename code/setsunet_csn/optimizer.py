"""Public model: learned proposals and the five-step CSN optimizer together."""
from pathlib import Path
import torch
from torch import nn
from .core import CSNConfig, solve
from .model import SchurProposalNet
from .model_stage2 import GraphSchurProposalNet
from .stage2 import Stage2Config, solve as solve_stage2

class SETSUNETCSN(nn.Module):
    def __init__(self, config=None, proposal=None):
        super().__init__()
        self.config=config or CSNConfig()
        self.proposal=proposal if proposal is not None else SchurProposalNet().double()

    def forward(self, problem):
        """Return the optimized free coordinates plus per-step proof and work logs."""
        return solve(problem,self.proposal,method="csn",config=self.config)

    @classmethod
    def from_checkpoint(cls,path,config=None):
        payload=torch.load(Path(path),map_location="cpu",weights_only=True)
        proposal=SchurProposalNet().double()
        proposal.load_state_dict(payload["state_dict"],strict=True)
        return cls(config,proposal).eval()


class SETSUNETCSNStage2(nn.Module):
    """Graph-aware CSN with measured block whitening and adaptive damping."""

    def __init__(self, config=None, proposal=None):
        super().__init__()
        self.config = config or Stage2Config()
        self.proposal = proposal if proposal is not None else GraphSchurProposalNet().double()

    def forward(self, problem):
        return solve_stage2(problem, self.proposal, method="csn", config=self.config)

    @classmethod
    def from_checkpoint(cls, path, config=None):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        proposal = GraphSchurProposalNet(**payload["architecture"]).double()
        proposal.load_state_dict(payload["state_dict"], strict=True)
        return cls(config, proposal).eval()

