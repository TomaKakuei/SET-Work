"""A topology-agnostic, five-step learned optimizer.

The solver consumes scalar differentiable factors and their parameter gradients.
It does not assume a task identity, a fixed number of factors, or fixed parameter
blocks.  Coordinates are only used to reconstruct updates from normalized input
directions; learned processing operates on invariant scalar geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FactorFunction = Callable[[Tensor], Tensor]


@dataclass
class DifferentialFactorState:
    """Recurrent state for a fixed topology across optimization steps."""

    global_hidden: Tensor
    factor_hidden: Tensor
    block_hidden: Tensor
    previous_update: Tensor


@dataclass
class DifferentialFactorStep:
    """Outputs and diagnostics from one shared solver application."""

    update: Tensor
    state: DifferentialFactorState
    factor_coefficients: Tensor
    history_coefficients: Tensor
    block_gains: Tensor
    edge_mask: Tensor


@dataclass
class DifferentialFactorOptimizationResult:
    """Trace returned by :class:`FiveStepDifferentialFactorOptimizer`."""

    initial_theta: Tensor
    final_theta: Tensor
    final_factor_values: Tensor
    theta_trace: Tensor
    factor_value_trace: Tensor
    acceptance_trace: Tensor
    update_trace: Tensor
    state: DifferentialFactorState


class GeometryAttention(nn.Module):
    """Self-attention with a scalar pairwise-geometry bias and safe masking."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_scale = nn.Parameter(torch.zeros(heads))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: Tensor, mask: Tensor, geometry: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [N, L, H]")
        n, length, hidden = tokens.shape
        if hidden != self.hidden_dim:
            raise ValueError("unexpected token hidden dimension")
        if mask.shape != (n, length):
            raise ValueError("mask must have shape [N, L]")
        if geometry.shape != (n, length, length):
            raise ValueError("geometry must have shape [N, L, L]")

        qkv = self.qkv(tokens).reshape(n, length, 3, self.heads, self.head_dim)
        q, k, value = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        value = value.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        logits = logits + self.geometry_scale[None, :, None, None] * geometry[:, None]

        key_mask = mask[:, None, None, :]
        logits = logits.masked_fill(~key_mask, -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * key_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(n, length, hidden)
        attended = attended * mask[..., None].to(attended.dtype)
        x = self.norm1(tokens + self.out(attended))
        x = self.norm2(x + self.ffn(x))
        return x * mask[..., None].to(x.dtype)


class DifferentialFactorSolver(nn.Module):
    """One coordinate-equivariant update step on a differential factor graph.

    A call may use any positive ``P``, ``F``, and ``K``.  Recurrent state can be
    reused only while that topology remains unchanged, as it does inside the
    five-step optimizer.
    """

    BASE_EDGE_FEATURES = 8

    def __init__(
        self,
        hidden_dim: int = 96,
        heads: int = 4,
        factor_feature_dim: int = 0,
        block_feature_dim: int = 0,
        global_feature_dim: int = 0,
        max_step_norm: float = 0.25,
        eps: float = 1.0e-8,
        inference_edge_attention_chunk_size: Optional[int] = 64,
        inference_decoder_block_chunk_size: Optional[int] = 8,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if max_step_norm <= 0:
            raise ValueError("max_step_norm must be positive")
        if (
            inference_edge_attention_chunk_size is not None
            and inference_edge_attention_chunk_size < 1
        ):
            raise ValueError(
                "inference edge-attention chunk size must be positive when provided"
            )
        if (
            inference_decoder_block_chunk_size is not None
            and inference_decoder_block_chunk_size < 1
        ):
            raise ValueError(
                "inference decoder chunk size must be positive when provided"
            )
        self.hidden_dim = hidden_dim
        self.factor_feature_dim = factor_feature_dim
        self.block_feature_dim = block_feature_dim
        self.global_feature_dim = global_feature_dim
        self.max_step_norm = max_step_norm
        self.eps = eps
        self.inference_edge_attention_chunk_size = (
            inference_edge_attention_chunk_size
        )
        self.inference_decoder_block_chunk_size = (
            inference_decoder_block_chunk_size
        )

        edge_dim = (
            self.BASE_EDGE_FEATURES
            + factor_feature_dim
            + block_feature_dim
            + global_feature_dim
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_attention = GeometryAttention(hidden_dim, heads)
        self.factor_attention = GeometryAttention(hidden_dim, heads)
        self.block_attention = GeometryAttention(hidden_dim, heads)

        self.factor_gru = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.block_gru = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.global_gru = nn.GRUCell(
            2 * hidden_dim + global_feature_dim, hidden_dim
        )

        self.initial_global = nn.Parameter(torch.zeros(hidden_dim))
        self.initial_factor = nn.Parameter(torch.zeros(hidden_dim))
        self.initial_block = nn.Parameter(torch.zeros(hidden_dim))

        decoder_dim = 4 * hidden_dim
        self.coefficient_head = nn.Sequential(
            nn.Linear(decoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.history_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # At initialization the method is a conservative normalized-gradient
        # step. Learning begins from a functioning five-step optimizer.
        nn.init.zeros_(self.coefficient_head[-1].weight)
        nn.init.zeros_(self.coefficient_head[-1].bias)
        nn.init.zeros_(self.history_head[-1].weight)
        nn.init.zeros_(self.history_head[-1].bias)
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.constant_(self.gain_head[-1].bias, -1.4)

    def _apply_edge_attention(
        self,
        tokens: Tensor,
        mask: Tensor,
        geometry: Tensor,
    ) -> Tensor:
        """Evaluate independent edge-attention rows with bounded eval memory.

        The first dimension contains independent ``(batch, block)`` rows. In
        evaluation mode it can therefore be tiled without changing the
        attention domain, recurrent state, or five-call particle contract.
        Training retains the original fused path for throughput.
        """

        chunk_size = self.inference_edge_attention_chunk_size
        if self.training or chunk_size is None or tokens.shape[0] <= chunk_size:
            return self.edge_attention(tokens, mask, geometry)
        chunks = []
        for start in range(0, tokens.shape[0], chunk_size):
            stop = min(start + chunk_size, tokens.shape[0])
            chunks.append(
                self.edge_attention(
                    tokens[start:stop],
                    mask[start:stop],
                    geometry[start:stop],
                )
            )
        return torch.cat(chunks, dim=0)

    def _decode_coefficient_residual(
        self,
        edge_tokens: Tensor,
        factor_hidden: Tensor,
        block_hidden: Tensor,
        global_hidden: Tensor,
    ) -> Tensor:
        """Decode independent block rows without a full ``[B,K,F,4H]`` copy."""

        batch, blocks, factors, _ = edge_tokens.shape
        chunk_size = self.inference_decoder_block_chunk_size
        if self.training or chunk_size is None or blocks <= chunk_size:
            decoder_context = torch.cat(
                (
                    edge_tokens,
                    factor_hidden[:, None].expand(-1, blocks, -1, -1),
                    block_hidden[:, :, None].expand(-1, -1, factors, -1),
                    global_hidden[:, None, None].expand(
                        -1, blocks, factors, -1
                    ),
                ),
                dim=-1,
            )
            return 2.0 * torch.tanh(
                self.coefficient_head(decoder_context).squeeze(-1)
            )

        decoded = []
        for start in range(0, blocks, chunk_size):
            stop = min(start + chunk_size, blocks)
            width = stop - start
            decoder_context = torch.cat(
                (
                    edge_tokens[:, start:stop],
                    factor_hidden[:, None].expand(-1, width, -1, -1),
                    block_hidden[:, start:stop, None].expand(
                        -1, -1, factors, -1
                    ),
                    global_hidden[:, None, None].expand(
                        -1, width, factors, -1
                    ),
                ),
                dim=-1,
            )
            decoded.append(
                2.0 * torch.tanh(
                    self.coefficient_head(decoder_context).squeeze(-1)
                )
            )
        return torch.cat(decoded, dim=1).reshape(batch, blocks, factors)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
        weights = mask.to(values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        numerator = (values * weights).sum(dim=dim)
        denominator = weights.sum(dim=dim).clamp_min(1.0)
        return numerator / denominator

    def _optional_features(
        self,
        batch: int,
        factors: int,
        blocks: int,
        device: torch.device,
        dtype: torch.dtype,
        factor_features: Optional[Tensor],
        block_features: Optional[Tensor],
        global_features: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if factor_features is None:
            factor_features = torch.zeros(
                batch, factors, self.factor_feature_dim, device=device, dtype=dtype
            )
        if block_features is None:
            block_features = torch.zeros(
                batch, blocks, self.block_feature_dim, device=device, dtype=dtype
            )
        if global_features is None:
            global_features = torch.zeros(
                batch, self.global_feature_dim, device=device, dtype=dtype
            )
        if factor_features.shape != (batch, factors, self.factor_feature_dim):
            raise ValueError("factor_features has an incompatible shape")
        if block_features.shape != (batch, blocks, self.block_feature_dim):
            raise ValueError("block_features has an incompatible shape")
        if global_features.shape != (batch, self.global_feature_dim):
            raise ValueError("global_features has an incompatible shape")
        return (
            factor_features.to(device=device, dtype=dtype),
            block_features.to(device=device, dtype=dtype),
            global_features.to(device=device, dtype=dtype),
        )

    def _initial_state(
        self,
        batch: int,
        factors: int,
        blocks: int,
        parameters: int,
        reference: Tensor,
    ) -> DifferentialFactorState:
        return DifferentialFactorState(
            global_hidden=self.initial_global.to(reference).expand(batch, -1),
            factor_hidden=self.initial_factor.to(reference).expand(
                batch, factors, -1
            ),
            block_hidden=self.initial_block.to(reference).expand(batch, blocks, -1),
            previous_update=reference.new_zeros(batch, parameters),
        )

    def _validate_state(
        self,
        state: DifferentialFactorState,
        batch: int,
        factors: int,
        blocks: int,
        parameters: int,
    ) -> None:
        expected = {
            "global_hidden": (batch, self.hidden_dim),
            "factor_hidden": (batch, factors, self.hidden_dim),
            "block_hidden": (batch, blocks, self.hidden_dim),
            "previous_update": (batch, parameters),
        }
        for name, shape in expected.items():
            if getattr(state, name).shape != shape:
                raise ValueError(f"state.{name} must have shape {shape}")

    @staticmethod
    def _sum_coordinates_by_block(
        values: Tensor,
        block_id: Tensor,
        blocks: int,
    ) -> Tensor:
        """Sum the last coordinate axis without materializing ``[K, ..., P]``.

        ``values`` may be ``[B, P]`` or ``[B, F, P]``.  The returned tensor
        replaces the final ``P`` dimension with ``K``.  This segmented form is
        the key deployment path for image-sized parameter vectors: the earlier
        one-hot implementation allocated a dense ``[B, K, F, P]`` tensor.
        """

        index = block_id.reshape((1,) * (values.ndim - 1) + (-1,)).expand_as(
            values
        )
        output = values.new_zeros(*values.shape[:-1], blocks)
        return output.scatter_add(2 if values.ndim == 3 else 1, index, values)

    def _segmented_gram(
        self,
        unit_directions: Tensor,
        block_id: Tensor,
        blocks: int,
    ) -> Tensor:
        """Exact per-block factor Gram matrices with ``O(BFP + BKF^2)`` memory."""

        matrices = []
        for block_index in range(blocks):
            local = unit_directions[..., block_id == block_index]
            matrices.append(torch.matmul(local, local.transpose(-2, -1)))
        return torch.stack(matrices, dim=1)

    def forward(
        self,
        theta: Tensor,
        directions: Tensor,
        factor_values: Tensor,
        block_id: Tensor,
        factor_mask: Optional[Tensor] = None,
        factor_features: Optional[Tensor] = None,
        block_features: Optional[Tensor] = None,
        global_features: Optional[Tensor] = None,
        state: Optional[DifferentialFactorState] = None,
    ) -> DifferentialFactorStep:
        if theta.ndim != 2:
            raise ValueError("theta must have shape [B, P]")
        if directions.ndim != 3:
            raise ValueError("directions must have shape [B, F, P]")
        batch, parameters = theta.shape
        if directions.shape[0] != batch or directions.shape[2] != parameters:
            raise ValueError("directions is incompatible with theta")
        factors = directions.shape[1]
        if factor_values.shape != (batch, factors):
            raise ValueError("factor_values must have shape [B, F]")
        if block_id.shape != (parameters,):
            raise ValueError("block_id must have shape [P]")
        if factors < 1 or parameters < 1:
            raise ValueError("at least one factor and one parameter are required")
        if not theta.is_floating_point():
            raise ValueError("theta must use a floating-point dtype")

        block_id = block_id.to(device=theta.device, dtype=torch.long)
        if torch.any(block_id < 0):
            raise ValueError("block_id entries must be non-negative")
        blocks = int(block_id.max().item()) + 1
        expected_ids = torch.arange(blocks, device=theta.device)
        if not torch.equal(torch.unique(block_id, sorted=True), expected_ids):
            raise ValueError("block_id entries must be contiguous from zero")

        directions = directions.to(device=theta.device, dtype=theta.dtype)
        factor_values = factor_values.to(device=theta.device, dtype=theta.dtype)
        if factor_mask is None:
            factor_mask = torch.ones(
                batch, factors, device=theta.device, dtype=torch.bool
            )
        elif factor_mask.shape != (batch, factors):
            raise ValueError("factor_mask must have shape [B, F]")
        else:
            factor_mask = factor_mask.to(device=theta.device, dtype=torch.bool)

        factor_features, block_features, global_features = self._optional_features(
            batch,
            factors,
            blocks,
            theta.device,
            theta.dtype,
            factor_features,
            block_features,
            global_features,
        )
        if state is None:
            state = self._initial_state(
                batch, factors, blocks, parameters, theta
            )
        else:
            self._validate_state(state, batch, factors, blocks, parameters)

        # Keep coordinate-sized tensors at [B, F, P] or [B, P].  In particular,
        # never create the old [B, K, F, P] block-direction tensor, whose memory
        # grew catastrophically for image-valued unknowns.
        squared_norm = self._sum_coordinates_by_block(
            directions.square(), block_id, blocks
        ).transpose(1, 2)
        direction_norm = squared_norm.clamp_min(self.eps**2).sqrt()
        edge_mask = factor_mask[:, None, :] & (squared_norm > self.eps**2)
        norm_by_coordinate = direction_norm.transpose(1, 2)[..., block_id]
        unit_directions = directions / norm_by_coordinate.clamp_min(self.eps)
        edge_by_coordinate = edge_mask.transpose(1, 2)[..., block_id]
        unit_directions = unit_directions * edge_by_coordinate.to(theta.dtype)

        theta_squared_norm = self._sum_coordinates_by_block(
            theta.square(), block_id, blocks
        )
        theta_norm = theta_squared_norm.clamp_min(self.eps**2).sqrt()
        theta_inner = self._sum_coordinates_by_block(
            unit_directions * theta[:, None, :], block_id, blocks
        ).transpose(1, 2)
        theta_cosine = theta_inner / theta_norm.clamp_min(self.eps)[..., None]

        active_edge_count = edge_mask.sum(dim=-1).clamp_min(1).to(theta.dtype)
        consensus = unit_directions.sum(dim=1) / active_edge_count[..., block_id]
        consensus_squared_norm = self._sum_coordinates_by_block(
            consensus.square(), block_id, blocks
        )
        consensus_norm = consensus_squared_norm.clamp_min(self.eps**2).sqrt()
        consensus = consensus / consensus_norm[..., block_id].clamp_min(self.eps)
        agreement = self._sum_coordinates_by_block(
            unit_directions * consensus[:, None, :], block_id, blocks
        ).transpose(1, 2)
        gram = self._segmented_gram(unit_directions, block_id, blocks)

        block_dimensions = torch.bincount(block_id, minlength=blocks).to(theta.dtype)
        base_features = torch.stack(
            (
                torch.tanh(factor_values)[:, None, :].expand(-1, blocks, -1),
                torch.log1p(factor_values.abs())[:, None, :].expand(
                    -1, blocks, -1
                ),
                torch.log1p(direction_norm),
                edge_mask.to(theta.dtype),
                agreement,
                theta_cosine,
                torch.log1p(block_dimensions)[None, :, None].expand(
                    batch, -1, factors
                ),
                torch.log1p(theta_norm)[:, :, None].expand(-1, -1, factors),
            ),
            dim=-1,
        )
        expanded_factor_features = factor_features[:, None].expand(
            -1, blocks, -1, -1
        )
        expanded_block_features = block_features[:, :, None].expand(
            -1, -1, factors, -1
        )
        expanded_global_features = global_features[:, None, None].expand(
            -1, blocks, factors, -1
        )
        edge_features = torch.cat(
            (
                base_features,
                expanded_factor_features,
                expanded_block_features,
                expanded_global_features,
            ),
            dim=-1,
        )
        edge_tokens = self.edge_encoder(edge_features)
        edge_tokens = self._apply_edge_attention(
            edge_tokens.reshape(batch * blocks, factors, self.hidden_dim),
            edge_mask.reshape(batch * blocks, factors),
            gram.reshape(batch * blocks, factors, factors),
        ).reshape(batch, blocks, factors, self.hidden_dim)

        factor_node_mask = edge_mask.any(dim=1)
        pair_mask = edge_mask[..., :, None] & edge_mask[..., None, :]
        factor_geometry = (
            gram * pair_mask.to(theta.dtype)
        ).sum(dim=1) / pair_mask.to(theta.dtype).sum(dim=1).clamp_min(1.0)
        factor_seed = self._masked_mean(edge_tokens, edge_mask, dim=1)
        factor_seed = factor_seed + state.factor_hidden
        factor_context = self.factor_attention(
            factor_seed, factor_node_mask, factor_geometry
        )
        global_for_factors = state.global_hidden[:, None].expand(
            -1, factors, -1
        )
        factor_hidden = self.factor_gru(
            torch.cat((factor_context, global_for_factors), dim=-1).reshape(
                batch * factors, 2 * self.hidden_dim
            ),
            state.factor_hidden.reshape(batch * factors, self.hidden_dim),
        ).reshape(batch, factors, self.hidden_dim)
        factor_hidden = factor_hidden * factor_node_mask[..., None].to(theta.dtype)

        block_node_mask = edge_mask.any(dim=2)
        signature = torch.log1p(direction_norm) * edge_mask.to(theta.dtype)
        signature_norm = signature.square().sum(dim=-1).clamp_min(
            self.eps**2
        ).sqrt()
        block_geometry = torch.einsum("bkf,blf->bkl", signature, signature)
        block_geometry = block_geometry / (
            signature_norm[:, :, None] * signature_norm[:, None, :]
        ).clamp_min(self.eps)
        block_seed = self._masked_mean(edge_tokens, edge_mask, dim=2)
        block_seed = block_seed + state.block_hidden
        block_context = self.block_attention(
            block_seed, block_node_mask, block_geometry
        )
        global_for_blocks = state.global_hidden[:, None].expand(-1, blocks, -1)
        block_hidden = self.block_gru(
            torch.cat((block_context, global_for_blocks), dim=-1).reshape(
                batch * blocks, 2 * self.hidden_dim
            ),
            state.block_hidden.reshape(batch * blocks, self.hidden_dim),
        ).reshape(batch, blocks, self.hidden_dim)
        block_hidden = block_hidden * block_node_mask[..., None].to(theta.dtype)

        pooled_factors = self._masked_mean(
            factor_hidden, factor_node_mask, dim=1
        )
        pooled_blocks = self._masked_mean(block_hidden, block_node_mask, dim=1)
        global_hidden = self.global_gru(
            torch.cat((pooled_factors, pooled_blocks, global_features), dim=-1),
            state.global_hidden,
        )

        coefficient_residual = self._decode_coefficient_residual(
            edge_tokens,
            factor_hidden,
            block_hidden,
            global_hidden,
        )
        raw_factor_coefficients = (
            1.0 + coefficient_residual
        ) * edge_mask.to(theta.dtype)

        previous_squared_norm = self._sum_coordinates_by_block(
            state.previous_update.square(), block_id, blocks
        )
        previous_norm = previous_squared_norm.clamp_min(self.eps**2).sqrt()
        previous_unit = state.previous_update / previous_norm[..., block_id].clamp_min(
            self.eps
        )
        history_available = (previous_squared_norm > self.eps**2) & block_node_mask
        block_global_context = torch.cat(
            (block_hidden, global_hidden[:, None].expand(-1, blocks, -1)), dim=-1
        )
        raw_history_coefficients = 2.0 * torch.tanh(
            self.history_head(block_global_context).squeeze(-1)
        ) * history_available.to(theta.dtype)

        coefficient_scale = (
            raw_factor_coefficients.abs().sum(dim=-1)
            + raw_history_coefficients.abs()
        ).clamp_min(1.0)
        factor_coefficients = raw_factor_coefficients / coefficient_scale[..., None]
        history_coefficients = raw_history_coefficients / coefficient_scale
        coefficient_by_coordinate = factor_coefficients.transpose(1, 2)[
            ..., block_id
        ]
        mixed_direction = (
            coefficient_by_coordinate * unit_directions
        ).sum(dim=1) + history_coefficients[..., block_id] * previous_unit

        block_gains = self.max_step_norm * torch.sigmoid(
            self.gain_head(block_global_context).squeeze(-1)
        )
        block_gains = block_gains * block_node_mask.to(theta.dtype)
        update = -block_gains[..., block_id] * mixed_direction

        new_state = DifferentialFactorState(
            global_hidden=global_hidden,
            factor_hidden=factor_hidden,
            block_hidden=block_hidden,
            previous_update=update,
        )
        return DifferentialFactorStep(
            update=update,
            state=new_state,
            factor_coefficients=factor_coefficients,
            history_coefficients=history_coefficients,
            block_gains=block_gains,
            edge_mask=edge_mask,
        )


def factor_value_and_gradients(
    theta: Tensor,
    factor_fn: FactorFunction,
    *,
    create_graph: bool = False,
    vjp_chunk_size: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate factors and all factor gradients using one batched VJP.

    ``factor_fn`` must return one scalar cost per batch item and factor, with
    shape ``[B, F]``. Factors are assumed not to couple different batch items.
    """

    if theta.ndim != 2:
        raise ValueError("theta must have shape [B, P]")
    if not theta.requires_grad:
        raise ValueError("theta must require gradients")
    values = factor_fn(theta)
    if values.ndim != 2 or values.shape[0] != theta.shape[0]:
        raise ValueError("factor_fn(theta) must have shape [B, F]")
    factors = values.shape[1]
    if factors < 1:
        raise ValueError("factor_fn must return at least one factor")
    if not values.requires_grad:
        return values, theta.new_zeros(theta.shape[0], factors, theta.shape[1])
    if vjp_chunk_size is not None and vjp_chunk_size < 1:
        raise ValueError("vjp_chunk_size must be positive when provided")
    chunk_size = factors if vjp_chunk_size is None else min(vjp_chunk_size, factors)
    gradient_chunks = []
    for start in range(0, factors, chunk_size):
        stop = min(start + chunk_size, factors)
        basis = F.one_hot(
            torch.arange(start, stop, device=values.device), num_classes=factors
        ).to(values)
        grad_outputs = basis[:, None, :].expand(
            stop - start, values.shape[0], factors
        )
        chunk = torch.autograd.grad(
            values,
            theta,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            create_graph=create_graph,
            retain_graph=create_graph or stop < factors,
            allow_unused=True,
        )[0]
        if chunk is None:
            chunk = theta.new_zeros(stop - start, theta.shape[0], theta.shape[1])
        gradient_chunks.append(chunk)
    gradients = torch.cat(gradient_chunks, dim=0)
    return values, gradients.permute(1, 0, 2)


class FiveStepDifferentialFactorOptimizer(nn.Module):
    """Apply one shared DifferentialFactorSolver for a fixed small step budget."""

    def __init__(
        self,
        solver: Optional[DifferentialFactorSolver] = None,
        steps: int = 5,
        acceptance_tolerance: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        self.solver = solver or DifferentialFactorSolver()
        self.steps = steps
        self.acceptance_tolerance = acceptance_tolerance

    @staticmethod
    def _validate_values(values: Tensor, batch: int, factors: int) -> None:
        if values.shape != (batch, factors):
            raise ValueError("factor_fn changed its output shape during optimization")

    def forward(
        self,
        initial_theta: Tensor,
        factor_fn: FactorFunction,
        block_id: Tensor,
        factor_mask: Optional[Tensor] = None,
        factor_features: Optional[Tensor] = None,
        block_features: Optional[Tensor] = None,
        global_features: Optional[Tensor] = None,
        *,
        steps: Optional[int] = None,
        create_graph: bool = False,
        enforce_descent: bool = True,
        straight_through_acceptance: bool = False,
    ) -> DifferentialFactorOptimizationResult:
        run_steps = self.steps if steps is None else steps
        if run_steps < 1:
            raise ValueError("steps must be positive")
        if initial_theta.ndim != 2:
            raise ValueError("initial_theta must have shape [B, P]")
        theta = initial_theta
        if not theta.requires_grad:
            theta = theta.detach().clone().requires_grad_(True)
        batch = theta.shape[0]

        state: Optional[DifferentialFactorState] = None
        # Traces are diagnostics. Detaching them keeps five-step training memory
        # bounded; gradients remain available through final_theta/final values.
        theta_trace = [theta.detach()]
        value_trace = []
        acceptances = []
        accepted_updates = []

        for step_index in range(run_steps):
            current_values, directions = factor_value_and_gradients(
                theta, factor_fn, create_graph=create_graph
            )
            factors = current_values.shape[1]
            if step_index == 0:
                value_trace.append(current_values.detach())
                if factor_mask is None:
                    factor_mask = torch.ones(
                        batch, factors, device=theta.device, dtype=torch.bool
                    )
            self._validate_values(current_values, batch, factors)
            assert factor_mask is not None
            if factor_mask.shape != (batch, factors):
                raise ValueError("factor_mask must have shape [B, F]")

            solver_values = current_values if create_graph else current_values.detach()
            solver_directions = directions if create_graph else directions.detach()
            proposal = self.solver(
                theta,
                solver_directions,
                solver_values,
                block_id,
                factor_mask=factor_mask,
                factor_features=factor_features,
                block_features=block_features,
                global_features=global_features,
                state=state,
            )
            candidate = theta + proposal.update
            candidate_values = factor_fn(candidate)
            self._validate_values(candidate_values, batch, factors)

            if enforce_descent:
                mask_float = factor_mask.to(current_values.dtype)
                current_score = (current_values.detach() * mask_float).sum(dim=-1)
                candidate_score = (candidate_values.detach() * mask_float).sum(dim=-1)
                accepted = candidate_score <= (
                    current_score + self.acceptance_tolerance
                )
            else:
                accepted = torch.ones(batch, device=theta.device, dtype=torch.bool)
            gate = accepted.to(theta.dtype)[:, None]
            accepted_update = gate * proposal.update
            if straight_through_acceptance:
                # Forward execution remains the exact hard trust gate. A rejected
                # proposal receives the identity update derivative during
                # training, avoiding a zero-gradient rejection dead zone.
                rejected = 1.0 - gate
                accepted_update = accepted_update + rejected * (
                    proposal.update - proposal.update.detach()
                )
            theta = theta + accepted_update
            selected_values = torch.where(
                accepted[:, None], candidate_values, current_values
            )
            state = replace(proposal.state, previous_update=accepted_update)

            theta_trace.append(theta.detach())
            value_trace.append(selected_values.detach())
            acceptances.append(accepted)
            accepted_updates.append(accepted_update.detach())

        final_values = factor_fn(theta)
        assert state is not None
        return DifferentialFactorOptimizationResult(
            initial_theta=initial_theta,
            final_theta=theta,
            final_factor_values=final_values,
            theta_trace=torch.stack(theta_trace, dim=1),
            factor_value_trace=torch.stack(value_trace, dim=1),
            acceptance_trace=torch.stack(acceptances, dim=1),
            update_trace=torch.stack(accepted_updates, dim=1),
            state=state,
        )


__all__ = [
    "DifferentialFactorOptimizationResult",
    "DifferentialFactorSolver",
    "DifferentialFactorState",
    "DifferentialFactorStep",
    "FiveStepDifferentialFactorOptimizer",
    "factor_value_and_gradients",
]
