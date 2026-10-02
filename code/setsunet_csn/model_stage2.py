"""Graph-aware, curvature-observable proposal network for SETSUNET CSN stage 2.

The network only selects a completion subspace.  Step coefficients and all
curvature values are still measured and solved analytically by ``stage2.py``.
All scalar features are invariant to orthogonal changes of coordinates inside
a declared variable block; output directions are carried by equivariant
vectors, so the rule preserves the block-coordinate symmetry of the pilot.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _segment_sum(values: Tensor, ids: Tensor, blocks: int) -> Tensor:
    result = values.new_zeros((blocks,) + values.shape[1:])
    return result.index_add(0, ids, values)


def _neighbor_mean(values: Tensor, edges: Tensor) -> Tensor:
    if edges.numel() == 0:
        return values.new_zeros(values.shape)
    source, target = edges
    both_source = torch.cat((source, target))
    both_target = torch.cat((target, source))
    result = values.new_zeros(values.shape).index_add(0, both_target, values[both_source])
    counts = values.new_zeros(values.shape[0]).index_add(
        0, both_target, torch.ones_like(both_target, dtype=values.dtype)
    )
    return result / counts.clamp_min(1.0)[:, None]


class GraphSchurProposalNet(nn.Module):
    """Small shared block rule using every measured response and graph edges."""

    carriers = 8

    def __init__(
        self,
        *,
        hidden: int = 64,
        max_directions: int = 8,
        max_probes: int = 8,
        information: str = "full",
        message_rounds: int = 2,
    ) -> None:
        super().__init__()
        if information not in ("limited", "full"):
            raise ValueError("information must be limited or full")
        if hidden < 8 or max_directions < 1 or max_probes < 1:
            raise ValueError("invalid network size")
        self.hidden = int(hidden)
        self.max_directions = int(max_directions)
        self.max_probes = int(max_probes)
        self.information = information
        self.message_rounds = int(message_rounds)
        # Carrier Gram, block size/energy/degree, and four summaries per probe.
        self.feature_dim = self.carriers**2 + 4 + 4 * self.max_probes
        self.embed = nn.Sequential(
            nn.Linear(self.feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.message = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh())
            for _ in range(message_rounds)
        )
        self.head = nn.Linear(hidden, max_directions * (1 + 2 * self.carriers))

    @staticmethod
    def _normalized(vector: Tensor) -> Tensor:
        return vector / vector.norm().clamp_min(1.0e-15)

    def _response_carriers(
        self,
        base: dict[str, Tensor],
        ga: Tensor,
        gb: Tensor,
        ids: Tensor,
        edges: Tensor,
    ) -> Tensor:
        e = base["e"]
        zero = torch.zeros_like(e)
        history = base.get("history")
        if history is None:
            history = zero
        Q, Ya, Yb = base["Q"], base["Ya"], base["Yb"]
        if Q.shape[1]:
            mean_y = 0.5 * (Ya + Yb)
            diff_y = Ya - Yb
            mean_response = mean_y @ (mean_y.T @ e)
            diff_response = diff_y @ (diff_y.T @ e)
            cross_response = Ya @ (Yb.T @ e) + Yb @ (Ya.T @ e)
        else:
            mean_response = diff_response = cross_response = zero
        blocks = int(ids.max()) + 1
        energy = _segment_sum(e.square()[:, None], ids, blocks)[:, 0]
        if self.information == "full":
            neighbor_energy = _neighbor_mean(energy[:, None], edges)[:, 0]
            graph_carrier = e * neighbor_energy[ids] / energy.mean().clamp_min(1.0e-15)
            vectors = (
                e, ga - gb, 0.5 * (ga + gb), history,
                mean_response, diff_response, cross_response, graph_carrier,
            )
        else:
            vectors = (e, ga - gb, 0.5 * (ga + gb), history, zero, zero, zero, zero)
        return torch.stack(tuple(self._normalized(item) for item in vectors), dim=1)

    def _features(
        self,
        carriers: Tensor,
        base: dict[str, Tensor],
        ids: Tensor,
        edges: Tensor,
    ) -> Tensor:
        blocks = int(ids.max()) + 1
        gram_values = (carriers[:, :, None] * carriers[:, None, :]).reshape(
            carriers.shape[0], -1
        )
        grams = _segment_sum(gram_values, ids, blocks)
        sizes = torch.bincount(ids, minlength=blocks).to(carriers)
        e = base["e"]
        energy = _segment_sum(e.square()[:, None], ids, blocks)[:, 0]
        degrees = carriers.new_zeros(blocks)
        if edges.numel():
            degrees.index_add_(0, edges.reshape(-1), torch.ones_like(edges.reshape(-1), dtype=carriers.dtype))
        summaries = carriers.new_zeros(blocks, 4 * self.max_probes)
        if self.information == "full" and base["Q"].shape[1]:
            for probe in range(min(base["Q"].shape[1], self.max_probes)):
                q, ya, yb = base["Q"][:, probe], base["Ya"][:, probe], base["Yb"][:, probe]
                values = torch.stack((q * (ya + yb) * 0.5, ya.square(), yb.square(), ya * yb), dim=1)
                summaries[:, 4 * probe : 4 * (probe + 1)] = _segment_sum(values, ids, blocks)
        scalars = torch.stack(
            (
                torch.log1p(sizes),
                torch.log1p(energy * sizes.sum() / energy.sum().clamp_min(1.0e-15)),
                torch.log1p(degrees),
                torch.log1p((degrees > 0).to(carriers.dtype)),
            ),
            dim=1,
        )
        if self.information == "limited":
            scalars[:, 2:] = 0.0
        return torch.cat((grams, scalars, summaries), dim=1)

    def forward(
        self,
        base: dict[str, Tensor],
        ga: Tensor,
        gb: Tensor,
        block_id: Tensor,
        block_edges: Tensor | None = None,
        *,
        directions: int | None = None,
    ) -> Tensor:
        ids = block_id.to(device=ga.device, dtype=torch.long)
        blocks = int(ids.max()) + 1
        if block_edges is None:
            block_edges = torch.empty(2, 0, dtype=torch.long, device=ga.device)
        else:
            block_edges = block_edges.to(device=ga.device, dtype=torch.long)
        carriers = self._response_carriers(base, ga, gb, ids, block_edges)
        features = self._features(carriers, base, ids, block_edges)
        hidden = self.embed(features)
        if self.information == "full":
            for layer in self.message:
                hidden = hidden + layer(torch.cat((hidden, _neighbor_mean(hidden, block_edges)), dim=1))
        raw = self.head(hidden).reshape(blocks, self.max_directions, 1 + 2 * self.carriers)
        count = self.max_directions if directions is None else min(int(directions), self.max_directions)
        raw = raw[:, :count]
        positive = torch.nn.functional.softplus(raw[:, :, 0]) + 1.0e-3
        local_coeff = raw[:, :, 1 : 1 + self.carriers]
        global_coeff = raw[:, :, 1 + self.carriers :]
        local_vector = (carriers[:, None, :] * local_coeff[ids]).sum(dim=-1)
        global_vector = (carriers[:, None, :] * global_coeff[ids]).sum(dim=-1)
        local_inner = _segment_sum(local_vector * base["e"][:, None], ids, blocks)
        global_inner = (global_vector * base["e"][:, None]).sum(dim=0, keepdim=True)
        return (
            positive[ids] * base["e"][:, None]
            + local_vector * local_inner[ids]
            + global_vector * global_inner
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = ["GraphSchurProposalNet"]
