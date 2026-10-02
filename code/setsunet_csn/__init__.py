"""SETSUNET CSN: learned completion of measured residual geometry."""
from .core import CSNConfig, solve
from .model import SchurProposalNet
from .stage2 import Stage2Config

from .optimizer import SETSUNETCSN, SETSUNETCSNStage2

__all__ = [
    "CSNConfig", "GraphSchurProposalNet", "SchurProposalNet", "SETSUNETCSN",
    "SETSUNETCSNStage2", "Stage2Config", "solve",
]

from .model_stage2 import GraphSchurProposalNet

from .stage4 import SETSUNETCSNStage4, Stage4Config
from .model_stage4 import NormalizedSchurNet
__all__ += ["SETSUNETCSNStage4", "Stage4Config", "NormalizedSchurNet"]

from .stage5 import SETSUNETCSNStage5, Stage5Config
__all__ += ["SETSUNETCSNStage5", "Stage5Config"]
