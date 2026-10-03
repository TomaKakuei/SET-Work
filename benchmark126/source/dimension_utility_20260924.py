"""Portable projection/spectrum surface used by the frozen chart search."""
import numpy as np
import torch
import run_fiber_system_20260924 as d
r = None
OUT = None


def projected(base, V):
    Z = V.clone()
    if base['Q'].shape[1]:
        Z = Z-base['Q']@torch.linalg.solve(base['A'], base['Y'].T@Z)
        U = torch.linalg.qr(base['Y'], mode='reduced')[0]
        for _ in range(2):
            Z = Z-U@(U.T@Z)
    return Z


def spectrum(V, reference=None):
    U, S, _ = torch.linalg.svd(V, full_matrices=False)
    threshold = 1e-9*float(V.norm() if reference is None else reference)
    rank = int((S > threshold).sum())
    energy = S.square()
    total = float(energy.sum())
    participation = total**2/max(float(energy.square().sum()), 1e-300) if total else 0.
    n95 = int(torch.searchsorted(energy.cumsum(0), energy.sum()*.95))+1 if total else 0
    return U, S, dict(rank=rank, participation=participation, energy95=n95,
                      singular_values=S.tolist())
