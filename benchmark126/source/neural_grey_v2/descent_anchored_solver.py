"""Descent-Anchored Differential Factor Solver (DA-DFS)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from .differential_factor_solver import (
    DifferentialFactorSolver,
    DifferentialFactorState,
    DifferentialFactorStep,
    factor_value_and_gradients,
)
from .robust_spatial_ops import (
    analytic_noise_risk_features,
    finite_difference_xy,
    separable_binomial_blur,
    smoothstep_risk_confidence,
)
from .typed_observables import (
    bounded_log_ratio,
    metric_dual_cosine,
    metric_dual_energy,
    metric_gradient_update_alignment,
    metric_tangent_energy,
)


FactorFunction = Callable[[Tensor], Tensor]
CoordinateMetricFunction = Callable[[Tensor], Tensor]


def spatial_observation_summary_features(
    observation: Tensor,
    field_shape: tuple[int, int, int],
) -> Tensor:
    """Twelve initializer-visible, task-free spatial distribution features."""

    channels, height, width = map(int, field_shape)
    expected = channels * height * width
    if observation.ndim != 2 or observation.shape[1] != expected:
        raise ValueError("observation summary requires the declared spatial field")
    image = observation.reshape(observation.shape[0], channels, height, width)
    scalar = image.mean(1, keepdim=True)
    smooth = torch.nn.functional.avg_pool2d(scalar, 5, stride=1, padding=2)
    high = scalar - smooth
    flat = scalar.flatten(1)
    channel_max = image.max(1).values
    channel_min = image.min(1).values
    channel_range = channel_max - channel_min
    return torch.stack(
        (
            flat.mean(1),
            flat.std(1, unbiased=False),
            torch.quantile(flat, 0.1, dim=1),
            torch.quantile(flat, 0.5, dim=1),
            torch.quantile(flat, 0.9, dim=1),
            (flat < 0.05).to(image.dtype).mean(1),
            (flat > 0.95).to(image.dtype).mean(1),
            high.square().mean((1, 2, 3)).sqrt(),
            high.abs().mean((1, 2, 3)),
            channel_range.mean((1, 2)),
            channel_range.std((1, 2), unbiased=False),
            image.flatten(1).std(1, unbiased=False),
        ),
        dim=1,
    )


def spatial_operator_probe_features(
    *,
    theta: Tensor,
    observation: Tensor,
    current_values: Tensor,
    compact_candidate_values: Tensor,
    total_gradient: Tensor,
    previous_update: Tensor,
    compact_update: Tensor,
    factor_mask: Tensor,
    block_id: Tensor,
    field_shape: tuple[int, int, int],
) -> Tensor:
    """Build a task-ID-free response signature for spatial routing.

    The first nine entries exactly reproduce the legacy M16--M20 router
    descriptor.  The remaining entries describe only observable image
    statistics and the local response of the supplied factor graph to the
    compact solver proposal; no dataset, degradation, or family identifier is
    included.
    """

    channels, height, width = map(int, field_shape)
    expected = channels * height * width
    batch = theta.shape[0]
    if any(
        tensor.ndim != 2 or tensor.shape != theta.shape
        for tensor in (observation, total_gradient, previous_update, compact_update)
    ):
        raise ValueError("spatial probe field tensors must share shape [B,P]")
    if theta.shape[1] != expected:
        raise ValueError("spatial probe field_shape product must match P")
    if current_values.shape != compact_candidate_values.shape:
        raise ValueError("spatial probe factor values must share shape [B,F]")
    if factor_mask.shape != current_values.shape:
        raise ValueError("spatial probe factor mask has the wrong shape")

    mask = factor_mask.to(theta.dtype)
    active = mask.sum(dim=-1).clamp_min(1.0)
    factors = current_values.shape[1]
    blocks = int(torch.unique(block_id).numel())
    observation = observation.to(theta)
    gradient_rms = total_gradient.square().mean(dim=-1).sqrt()
    previous_rms = previous_update.square().mean(dim=-1).sqrt()
    compact_rms = compact_update.square().mean(dim=-1).sqrt()
    current_score = (current_values.detach() * mask).sum(dim=-1)
    candidate_score = (compact_candidate_values.detach() * mask).sum(dim=-1)

    def cosine(first: Tensor, second: Tensor) -> Tensor:
        numerator = (first * second).sum(dim=-1)
        denominator = (
            first.square().sum(dim=-1).sqrt()
            * second.square().sum(dim=-1).sqrt()
        ).clamp_min(1.0e-8)
        return (numerator / denominator).clamp(-1.0, 1.0)

    def field_statistics(field: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        image = field.reshape(batch, channels, height, width)
        scale = image.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1.0e-6)
        dx = image[..., 1:] - image[..., :-1]
        dy = image[..., 1:, :] - image[..., :-1, :]
        tv = 0.5 * (
            dx.abs().mean(dim=(1, 2, 3)) + dy.abs().mean(dim=(1, 2, 3))
        ) / scale
        smooth = torch.nn.functional.avg_pool2d(
            image, 3, stride=1, padding=1, count_include_pad=False
        )
        high_frequency = (image - smooth).square().mean(dim=(1, 2, 3)).sqrt() / scale
        padded = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="replicate")
        laplacian = (
            4.0 * padded[..., 1:-1, 1:-1]
            - padded[..., :-2, 1:-1]
            - padded[..., 2:, 1:-1]
            - padded[..., 1:-1, :-2]
            - padded[..., 1:-1, 2:]
        )
        laplacian_rms = laplacian.square().mean(dim=(1, 2, 3)).sqrt() / scale
        return tuple(torch.tanh(torch.log1p(value)) for value in (tv, high_frequency, laplacian_rms))

    observation_tv, observation_hf, observation_lap = field_statistics(observation)
    gradient_tv, gradient_hf, gradient_lap = field_statistics(total_gradient)
    compact_tv, compact_hf, compact_lap = field_statistics(compact_update)
    relative_factor_change = torch.tanh(
        (compact_candidate_values.detach() - current_values.detach())
        / current_values.detach().abs().clamp_min(1.0e-6)
    )
    relative_factor_change = relative_factor_change * mask
    relative_mean = relative_factor_change.sum(dim=-1) / active
    relative_variance = (
        (relative_factor_change - relative_mean[:, None]).square() * mask
    ).sum(dim=-1) / active
    masked_min = relative_factor_change.masked_fill(~factor_mask, float("inf")).min(dim=-1).values
    masked_max = relative_factor_change.masked_fill(~factor_mask, float("-inf")).max(dim=-1).values
    improved_fraction = (
        ((compact_candidate_values.detach() < current_values.detach()) & factor_mask)
        .to(theta.dtype)
        .sum(dim=-1)
        / active
    )
    absolute_values = current_values.detach().abs() * mask
    factor_mean = absolute_values.sum(dim=-1) / active
    factor_variance = (
        (absolute_values - factor_mean[:, None]).square() * mask
    ).sum(dim=-1) / active
    factor_cv = torch.tanh(torch.log1p(factor_variance.sqrt() / factor_mean.clamp_min(1.0e-8)))
    factor_concentration = absolute_values.max(dim=-1).values / absolute_values.sum(dim=-1).clamp_min(1.0e-8)
    score_response = torch.tanh(
        (candidate_score - current_score) / current_score.abs().clamp_min(1.0e-6)
    )
    observation_flat = observation.reshape(batch, -1)
    gradient_flat = total_gradient.reshape(batch, -1)
    features = torch.stack(
        (
            torch.log1p(current_score.abs() / max(factors, 1)) / 8.0,
            torch.log10(gradient_rms.clamp_min(1.0e-12)) / 8.0,
            observation.mean(dim=-1),
            observation.std(dim=-1, unbiased=False),
            (theta.detach() - observation).square().mean(dim=-1).sqrt(),
            previous_rms,
            theta.new_full((batch,), math.log1p(factors) / 8.0),
            theta.new_full((batch,), math.log1p(theta.shape[1]) / 10.0),
            theta.new_full((batch,), math.log1p(blocks) / 5.0),
            torch.log10(compact_rms.clamp_min(1.0e-12)) / 8.0,
            cosine(-total_gradient, compact_update),
            score_response,
            relative_mean,
            relative_variance.sqrt(),
            masked_min,
            masked_max,
            improved_fraction,
            factor_cv,
            factor_concentration,
            observation_tv,
            observation_hf,
            observation_lap,
            (observation_flat <= 0.01).to(theta.dtype).mean(dim=-1),
            (observation_flat >= 0.99).to(theta.dtype).mean(dim=-1),
            gradient_tv,
            gradient_hf,
            gradient_lap,
            total_gradient.mean(dim=-1) / gradient_rms.clamp_min(1.0e-8),
            compact_tv,
            compact_hf,
            compact_lap,
            cosine(observation_flat, gradient_flat),
        ),
        dim=-1,
    )
    if features.shape != (batch, 32):
        raise RuntimeError("spatial operator probe feature dimension changed")
    return features.detach()


@dataclass(frozen=True)
class SpatialFieldContext:
    """Task-free structural declaration for an image-like variable block.

    ``observation`` is the measured field flattened in channel-first order.
    ``field_shape`` describes only coordinate topology; it is not a task,
    dataset, degradation, or camera identifier.
    """

    observation: Tensor
    field_shape: tuple[int, int, int]
    parameter_indices: Optional[Tensor] = None
    update_mask_fn: Optional[Callable[[Tensor], Tensor]] = None


@dataclass
class DescentAnchoredState:
    solver_state: DifferentialFactorState
    trust_scale: Tensor
    previous_acceptance: Tensor
    previous_reduction: Tensor


@dataclass
class DescentAnchoredStep:
    update: Tensor
    parent_update: Tensor
    proposed_solver_state: DifferentialFactorState
    basis_weights: Tensor
    block_gains: Tensor
    base_step: DifferentialFactorStep
    topology_controls: Tensor
    block_context_controls: Tensor


@dataclass
class DescentAnchoredOptimizationResult:
    initial_theta: Tensor
    final_theta: Tensor
    final_factor_values: Tensor
    proposal_factor_value_trace: Tensor
    proposal_theta_trace: Tensor
    factor_value_trace: Tensor
    theta_trace: Tensor
    acceptance_trace: Tensor
    step_scale_trace: Tensor
    update_trace: Tensor
    trust_trace: Tensor
    basis_weight_trace: Tensor
    topology_control_trace: Tensor
    block_context_control_trace: Tensor
    support_candidate_theta_trace: Tensor
    support_candidate_factor_value_trace: Tensor
    support_acceptance_trace: Tensor
    transfer_logit_trace: Tensor
    transfer_probability_trace: Tensor
    transfer_delta_prediction_trace: Tensor
    expanded_support_admission_logit_trace: Tensor
    expanded_support_admission_probability_trace: Tensor
    expanded_support_override_trace: Tensor
    terminal_keep_logit_trace: Tensor
    terminal_keep_probability_trace: Tensor
    terminal_rollback: Tensor
    pre_terminal_final_theta: Tensor
    joint_latent_pacing_logit_trace: Tensor
    joint_latent_pacing_scale_trace: Tensor
    joint_latent_pacing_gate_logit_trace: Tensor
    joint_latent_pacing_gate_probability_trace: Tensor
    joint_latent_pacing_gate_active_trace: Tensor
    joint_latent_continue_logit_trace: Tensor
    joint_latent_continue_probability_trace: Tensor
    joint_latent_continue_active_trace: Tensor
    joint_latent_unpaced_update_trace: Tensor
    joint_latent_pacing_feature_trace: Tensor
    joint_latent_typed_feature_trace: Tensor
    spatial_update_trace: Tensor
    spatial_trust_offset_trace: Tensor
    support_override_trace: Tensor
    schema_expert_admission_trace: Tensor
    schema_topology_pacing_trace: Tensor
    state: DescentAnchoredState


class GeneralizationTrustHead(nn.Module):
    """Predict support-to-query transfer from pooled task-free geometry."""

    FEATURE_DIM = 23

    def __init__(
        self,
        hidden_dim: int = 64,
        initial_logit: float = 6.0,
        normalize_features: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("generalization trust hidden_dim must be at least 8")
        self.input_norm: nn.Module = (
            nn.LayerNorm(self.FEATURE_DIM) if normalize_features else nn.Identity()
        )
        self.network = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor([initial_logit, 0.0]))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.FEATURE_DIM:
            raise ValueError(
                f"trust features must have shape [B,{self.FEATURE_DIM}]"
            )
        output = self.network(self.input_norm(features))
        return output[:, 0], output[:, 1]


class SpatialExpandedSupportAdmissionHead(nn.Module):
    """Fail-closed admission for spatial overrides beyond the legacy cap.

    The input is the same coordinate-invariant mathematical trajectory summary
    used by the generalization trust head.  A zero final layer and a strongly
    negative bias make a newly attached head reject every expanded override,
    exactly recovering the legacy deployment before calibration.
    """

    BASE_FEATURE_DIM = GeneralizationTrustHead.FEATURE_DIM

    def __init__(
        self,
        hidden_dim: int = 32,
        initial_logit: float = -8.0,
        spatial_probe_dim: int = 0,
        normalize_features: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("expanded support admission hidden_dim must be at least 8")
        if not 0 <= spatial_probe_dim <= 64:
            raise ValueError("expanded support spatial probe dim must lie in [0,64]")
        self.spatial_probe_dim = int(spatial_probe_dim)
        self.feature_dim = self.BASE_FEATURE_DIM + self.spatial_probe_dim
        self.normalize_features = bool(normalize_features)
        if self.normalize_features:
            self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
            self.register_buffer("feature_scale", torch.ones(self.feature_dim))
        self.network = nn.Sequential(
            nn.Identity() if self.normalize_features else nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, float(initial_logit))

    def set_feature_normalization(self, mean: Tensor, scale: Tensor) -> None:
        if not self.normalize_features:
            raise RuntimeError("feature normalization is disabled for this admission head")
        if mean.shape != (self.feature_dim,) or scale.shape != (self.feature_dim,):
            raise ValueError("feature normalization tensors have the wrong shape")
        with torch.no_grad():
            self.feature_mean.copy_(mean.to(self.feature_mean))
            self.feature_scale.copy_(scale.to(self.feature_scale).clamp_min(1.0e-6))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"admission features must have shape [B,{self.feature_dim}]"
            )
        if self.normalize_features:
            features = (features - self.feature_mean) / self.feature_scale
        return self.network(features).squeeze(-1)


class SpatialTerminalRollbackHead(SpatialExpandedSupportAdmissionHead):
    """Keep-versus-initial rollback decision computed inside the five calls."""

    TRAJECTORY_FEATURE_DIM = 32

    def __init__(
        self,
        hidden_dim: int = 32,
        initial_logit: float = 8.0,
        spatial_probe_dim: int = 0,
        trajectory_dim: int = 0,
        normalize_features: bool = False,
    ) -> None:
        if not 0 <= trajectory_dim <= self.TRAJECTORY_FEATURE_DIM:
            raise ValueError(
                "terminal rollback trajectory dim must lie in "
                f"[0,{self.TRAJECTORY_FEATURE_DIM}]"
            )
        # The parent builds a normalized MLP over one flat feature vector.  Keep
        # the externally visible probe dimension separate so the optimizer can
        # append the trajectory summary after the spatial probes.
        super().__init__(
            hidden_dim=hidden_dim,
            initial_logit=initial_logit,
            spatial_probe_dim=spatial_probe_dim + trajectory_dim,
            normalize_features=normalize_features,
        )
        self.spatial_probe_dim = int(spatial_probe_dim)
        self.trajectory_dim = int(trajectory_dim)


class SpatialJointLatentPacingHead(nn.Module):
    """Predict an image-block scale from support/query trajectory geometry.

    The head is evaluated inside each existing optimizer call.  A zero final
    layer and large positive bias make a newly attached head exactly preserve
    the inherited full image update before training.
    """

    TRUST_FEATURE_DIM = GeneralizationTrustHead.FEATURE_DIM
    LATENT_FEATURE_DIM = 13
    TYPED_CONTINUE_FEATURE_DIM = 17

    def __init__(
        self,
        hidden_dim: int = 64,
        initial_logit: float = 8.0,
        spatial_probe_dim: int = 32,
        query_probe_dim: int = 8,
        history_feature_dim: int = 0,
        exact_null_gate: bool = False,
        initial_gate_logit: float = 2.0,
        separate_gate_network: bool = False,
        lock_exact_null_gate: bool = False,
        gate_observation_feature_dim: int = 0,
        continue_gate_feature_mode: str = "none",
        continue_gate_initial_logit: float = 2.0,
        continue_gate_absorbing: bool = False,
        typed_continue_feature_dim: int = 0,
        typed_continue_threshold: float = 0.5,
        typed_continue_support_radius: Optional[float] = None,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("joint-latent pacing hidden_dim must be at least 8")
        if not 0 <= spatial_probe_dim <= 32:
            raise ValueError("joint-latent spatial probe dim must lie in [0,32]")
        if query_probe_dim != 8:
            raise ValueError("joint-latent query probe dim must equal eight")
        if history_feature_dim not in (0, 6):
            raise ValueError("joint-latent history feature dim must be zero or six")
        self.spatial_probe_dim = int(spatial_probe_dim)
        self.query_probe_dim = int(query_probe_dim)
        self.history_feature_dim = int(history_feature_dim)
        self.exact_null_gate = bool(exact_null_gate)
        self.initial_gate_logit = float(initial_gate_logit)
        self.separate_gate_network = bool(separate_gate_network)
        self.lock_exact_null_gate = bool(lock_exact_null_gate)
        if gate_observation_feature_dim not in (0, 12):
            raise ValueError("gate observation feature dim must be zero or twelve")
        self.gate_observation_feature_dim = int(gate_observation_feature_dim)
        self.continue_gate_feature_mode = str(continue_gate_feature_mode)
        self.continue_gate_initial_logit = float(continue_gate_initial_logit)
        self.continue_gate_absorbing = bool(continue_gate_absorbing)
        if typed_continue_feature_dim not in (0, self.TYPED_CONTINUE_FEATURE_DIM):
            raise ValueError(
                "typed continue feature dim must be zero or "
                f"{self.TYPED_CONTINUE_FEATURE_DIM}"
            )
        if not 0.0 < typed_continue_threshold < 1.0:
            raise ValueError("typed continue threshold must lie in (0,1)")
        self.typed_continue_feature_dim = int(typed_continue_feature_dim)
        self.typed_continue_threshold = float(typed_continue_threshold)
        if typed_continue_support_radius is not None and (
            not math.isfinite(float(typed_continue_support_radius))
            or float(typed_continue_support_radius) <= 0.0
        ):
            raise ValueError("typed continue support radius must be positive and finite")
        self.typed_continue_support_radius = (
            None
            if typed_continue_support_radius is None
            else float(typed_continue_support_radius)
        )
        if (self.separate_gate_network or self.lock_exact_null_gate) and not self.exact_null_gate:
            raise ValueError("separate/locked pacing gates require exact_null_gate")
        if self.gate_observation_feature_dim > 0 and not self.separate_gate_network:
            raise ValueError("observation-only gating requires a separate gate network")
        self.feature_dim = (
            self.TRUST_FEATURE_DIM
            + self.spatial_probe_dim
            + self.query_probe_dim
            + self.LATENT_FEATURE_DIM
            + self.history_feature_dim
        )
        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                2
                if self.exact_null_gate and not self.separate_gate_network
                else 1,
            ),
        )
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            self.network[-1].bias[0] = float(initial_logit)
            if self.exact_null_gate and not self.separate_gate_network:
                self.network[-1].bias[1] = self.initial_gate_logit
        if self.separate_gate_network:
            gate_input_dim = (
                self.gate_observation_feature_dim
                if self.gate_observation_feature_dim > 0
                else self.feature_dim
            )
            self.gate_network: Optional[nn.Module] = nn.Sequential(
                nn.Identity()
                if self.gate_observation_feature_dim > 0
                else nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.gate_network[-1].weight)
            nn.init.constant_(
                self.gate_network[-1].bias, self.initial_gate_logit
            )
        else:
            self.gate_network = None
        self.last_gate_logit: Optional[Tensor] = None
        self.continue_gate_network: Optional[nn.Module] = None
        self.typed_continue_gate_network: Optional[nn.Module] = None
        self.last_continue_gate_logit: Optional[Tensor] = None
        self.last_typed_support_active: Optional[Tensor] = None
        if self.continue_gate_feature_mode != "none":
            self.configure_continue_gate(
                self.continue_gate_feature_mode,
                initial_logit=self.continue_gate_initial_logit,
            )
        if self.typed_continue_feature_dim > 0:
            self.configure_typed_continue_gate(
                threshold=self.typed_continue_threshold
            )

    @staticmethod
    def _continue_feature_dim(feature_mode: str) -> int:
        dimensions = {
            "query_history": 14,
            "latent_query_history": 27,
            "full_math": 82,
        }
        if feature_mode not in dimensions:
            raise ValueError(
                "continue gate feature mode must be one of query_history, "
                "latent_query_history, or full_math"
            )
        return dimensions[feature_mode]

    def configure_continue_gate(
        self,
        feature_mode: str,
        *,
        initial_logit: float = 2.0,
    ) -> None:
        if self.continue_gate_network is not None:
            raise RuntimeError("continue gate is already configured")
        if self.feature_dim != 82:
            raise ValueError("continue gate requires the complete 82-feature M29 contract")
        dimension = self._continue_feature_dim(feature_mode)
        self.continue_gate_feature_mode = feature_mode
        self.continue_gate_initial_logit = float(initial_logit)
        self.register_buffer("continue_feature_mean", torch.zeros(dimension))
        self.register_buffer("continue_feature_scale", torch.ones(dimension))
        self.continue_gate_network = nn.Linear(dimension, 1)
        nn.init.zeros_(self.continue_gate_network.weight)
        nn.init.constant_(self.continue_gate_network.bias, initial_logit)

    def configure_typed_continue_gate(
        self,
        *,
        threshold: float = 0.5,
        initial_logit: float = 0.0,
    ) -> None:
        if self.typed_continue_gate_network is not None:
            raise RuntimeError("typed continue gate is already configured")
        if not 0.0 < threshold < 1.0:
            raise ValueError("typed continue threshold must lie in (0,1)")
        self.typed_continue_feature_dim = self.TYPED_CONTINUE_FEATURE_DIM
        self.typed_continue_threshold = float(threshold)
        self.register_buffer(
            "typed_continue_feature_mean",
            torch.zeros(self.TYPED_CONTINUE_FEATURE_DIM),
        )
        self.register_buffer(
            "typed_continue_feature_scale",
            torch.ones(self.TYPED_CONTINUE_FEATURE_DIM),
        )
        self.typed_continue_gate_network = nn.Linear(
            self.TYPED_CONTINUE_FEATURE_DIM, 1
        )
        nn.init.zeros_(self.typed_continue_gate_network.weight)
        nn.init.constant_(self.typed_continue_gate_network.bias, initial_logit)

    def _continue_features(self, features: Tensor) -> Tensor:
        if self.continue_gate_feature_mode == "query_history":
            return torch.cat((features[:, 55:63], features[:, 76:82]), dim=1)
        if self.continue_gate_feature_mode == "latent_query_history":
            return features[:, 55:82]
        if self.continue_gate_feature_mode == "full_math":
            return features
        raise RuntimeError("continue features requested while the gate is disabled")

    def forward(
        self,
        features: Tensor,
        gate_observation_features: Optional[Tensor] = None,
        typed_continue_features: Optional[Tensor] = None,
    ) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"joint-latent pacing features must have shape [B,{self.feature_dim}]"
            )
        output = self.network(features)
        if self.exact_null_gate:
            if self.gate_observation_feature_dim > 0:
                if gate_observation_features is None or gate_observation_features.shape != (
                    features.shape[0],
                    self.gate_observation_feature_dim,
                ):
                    raise ValueError("observation-only gate features have the wrong shape")
                gate_input = gate_observation_features
            else:
                gate_input = features
            self.last_gate_logit = (
                self.gate_network(gate_input).squeeze(-1)
                if self.gate_network is not None
                else output[:, 1]
            )
        else:
            self.last_gate_logit = None
        if typed_continue_features is not None:
            if self.typed_continue_gate_network is None:
                raise RuntimeError(
                    "typed continue features require a configured typed gate"
                )
            if typed_continue_features.shape != (
                features.shape[0],
                self.TYPED_CONTINUE_FEATURE_DIM,
            ):
                raise ValueError("typed continue features have the wrong shape")
            normalized_typed = (
                typed_continue_features
                - self.typed_continue_feature_mean.to(typed_continue_features)
            ) / self.typed_continue_feature_scale.to(
                typed_continue_features
            ).clamp_min(1.0e-5)
            raw_typed_logit = self.typed_continue_gate_network(
                normalized_typed
            ).squeeze(-1)
            threshold_logit = math.log(
                self.typed_continue_threshold
                / (1.0 - self.typed_continue_threshold)
            )
            self.last_continue_gate_logit = raw_typed_logit - threshold_logit
            if self.typed_continue_support_radius is None:
                self.last_typed_support_active = torch.ones(
                    features.shape[0],
                    device=features.device,
                    dtype=torch.bool,
                )
            else:
                self.last_typed_support_active = (
                    normalized_typed.abs().amax(dim=1)
                    <= self.typed_continue_support_radius
                )
                self.last_continue_gate_logit = torch.where(
                    self.last_typed_support_active,
                    self.last_continue_gate_logit,
                    torch.full_like(self.last_continue_gate_logit, -100.0),
                )
        elif self.continue_gate_network is None:
            self.last_continue_gate_logit = None
            self.last_typed_support_active = None
        else:
            self.last_typed_support_active = None
            continue_features = self._continue_features(features)
            normalized = (
                continue_features - self.continue_feature_mean.to(continue_features)
            ) / self.continue_feature_scale.to(continue_features).clamp_min(1.0e-5)
            self.last_continue_gate_logit = self.continue_gate_network(
                normalized
            ).squeeze(-1)
        return output[:, 0]


class SpatialFieldUpdateHead(nn.Module):
    """A channel-agnostic local update basis for declared spatial fields.

    The head is fully convolutional and shares its weights across field
    channels and resolutions.  It consumes only the current field, measured
    field, factor-gradient geometry, and recurrent update state.  Its final
    layer is zero initialized so adding the head to an existing checkpoint is
    an exact functional no-op before training.
    """

    FEATURE_CHANNELS = 8

    def __init__(
        self,
        hidden_dim: int = 32,
        max_update: float = 0.1,
        max_trust_offset: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("spatial hidden_dim must be at least 8")
        if max_update <= 0.0:
            raise ValueError("spatial max_update must be positive")
        if max_trust_offset <= 0.0:
            raise ValueError("spatial max_trust_offset must be positive")
        self.max_update = float(max_update)
        self.max_trust_offset = float(max_trust_offset)
        self.network = nn.Sequential(
            nn.Conv2d(self.FEATURE_CHANNELS, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _unit_rms(field: Tensor) -> Tensor:
        # Adding epsilon before sqrt avoids the undefined derivative of
        # sqrt(x) at an all-zero recurrent update.  The scale is a feature
        # normalization constant, not a quantity the head should optimize.
        rms = (
            field.square()
            .mean(dim=(-3, -2, -1), keepdim=True)
            .add(1.0e-12)
            .sqrt()
            .detach()
        )
        return field / rms

    def forward(
        self,
        theta: Tensor,
        observation: Tensor,
        total_gradient: Tensor,
        previous_update: Tensor,
        field_shape: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor]:
        if len(field_shape) != 3 or any(int(size) < 1 for size in field_shape):
            raise ValueError("field_shape must be a positive (C,H,W) tuple")
        channels, height, width = map(int, field_shape)
        expected = channels * height * width
        tensors = (theta, observation, total_gradient, previous_update)
        if any(tensor.ndim != 2 or tensor.shape != theta.shape for tensor in tensors):
            raise ValueError("spatial field tensors must share shape [B,P]")
        if theta.shape[1] != expected:
            raise ValueError("field_shape product must equal the parameter count")

        batch = theta.shape[0]
        current = theta.reshape(batch, channels, height, width)
        measured = observation.to(theta).reshape(batch, channels, height, width)
        gradient = total_gradient.reshape(batch, channels, height, width)
        previous = previous_update.reshape(batch, channels, height, width)
        current_mean = current.mean(dim=1, keepdim=True).expand_as(current)
        measured_mean = measured.mean(dim=1, keepdim=True).expand_as(measured)
        gradient_mean = gradient.mean(dim=1, keepdim=True).expand_as(gradient)
        features = torch.stack(
            (
                current,
                measured,
                self._unit_rms(gradient),
                self._unit_rms(previous),
                current - measured,
                current_mean,
                measured_mean,
                self._unit_rms(gradient_mean),
            ),
            dim=2,
        ).reshape(batch * channels, self.FEATURE_CHANNELS, height, width)
        output = self.network(features)
        update = self.max_update * torch.tanh(output[:, :1])
        trust_offset = self.max_trust_offset * torch.tanh(output[:, 1:2])
        trust_offset = trust_offset.reshape(batch, channels, -1).mean(dim=(1, 2))
        return update.reshape(batch, expected), trust_offset


class SpatialFieldMixtureUpdateHead(nn.Module):
    """Task-ID-free mixture of spatial update modes.

    Routing is inferred from globally pooled local field/gradient features.
    All experts are evaluated inside the same spatial-head call, so increasing
    the number of modes does not increase the learned recurrent-call budget.
    The trust output remains shared across modes.
    """

    FEATURE_CHANNELS = SpatialFieldUpdateHead.FEATURE_CHANNELS

    def __init__(
        self,
        hidden_dim: int = 32,
        experts: int = 2,
        router_hidden_dim: int = 16,
        global_feature_dim: int = 0,
        max_update: float = 0.1,
        max_trust_offset: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("spatial hidden_dim must be at least 8")
        if experts < 2:
            raise ValueError("spatial mixture requires at least two experts")
        if router_hidden_dim < 4:
            raise ValueError("router_hidden_dim must be at least four")
        if global_feature_dim < 0:
            raise ValueError("global_feature_dim cannot be negative")
        if max_update <= 0.0 or max_trust_offset <= 0.0:
            raise ValueError("spatial update and trust bounds must be positive")
        self.hidden_dim = int(hidden_dim)
        self.experts = int(experts)
        self.router_hidden_dim = int(router_hidden_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.max_update = float(max_update)
        self.max_trust_offset = float(max_trust_offset)
        self.trunk = nn.Sequential(
            nn.Conv2d(self.FEATURE_CHANNELS, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.expert_output = nn.Conv2d(hidden_dim, experts, 3, padding=1)
        self.trust_output = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.router = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_feature_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, experts),
        )
        nn.init.zeros_(self.expert_output.weight)
        nn.init.zeros_(self.expert_output.bias)
        nn.init.zeros_(self.trust_output.weight)
        nn.init.zeros_(self.trust_output.bias)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.last_routing_weights: Optional[Tensor] = None
        self.last_predicted_routing_weights: Optional[Tensor] = None

    @classmethod
    def from_single_head(
        cls,
        source: SpatialFieldUpdateHead,
        *,
        experts: int = 2,
        router_hidden_dim: int = 16,
        global_feature_dim: int = 0,
        perturbation: float = 1.0e-3,
        seed: int = 0,
    ) -> "SpatialFieldMixtureUpdateHead":
        """Migrate a single head while preserving its initial function.

        Expert perturbations have exactly zero mean.  The zero-initialized
        router therefore averages back to the source pre-tanh update, while
        the antisymmetric modes break the otherwise permanent expert symmetry.
        """

        if not isinstance(source.network[0], nn.Conv2d):
            raise TypeError("source spatial trunk layout is unsupported")
        hidden_dim = int(source.network[0].out_channels)
        migrated = cls(
            hidden_dim=hidden_dim,
            experts=experts,
            router_hidden_dim=router_hidden_dim,
            global_feature_dim=global_feature_dim,
            max_update=source.max_update,
            max_trust_offset=source.max_trust_offset,
        ).to(source.network[0].weight)
        with torch.no_grad():
            migrated.trunk[0].weight.copy_(source.network[0].weight)
            migrated.trunk[0].bias.copy_(source.network[0].bias)
            migrated.trunk[2].weight.copy_(source.network[2].weight)
            migrated.trunk[2].bias.copy_(source.network[2].bias)
            update_weight = source.network[4].weight[0]
            update_bias = source.network[4].bias[0]
            migrated.expert_output.weight.copy_(
                update_weight[None].expand(experts, -1, -1, -1)
            )
            migrated.expert_output.bias.fill_(float(update_bias))
            generator = torch.Generator(device=update_weight.device).manual_seed(seed)
            noise = perturbation * torch.randn(
                migrated.expert_output.weight.shape,
                generator=generator,
                device=update_weight.device,
                dtype=update_weight.dtype,
            )
            noise = noise - noise.mean(dim=0, keepdim=True)
            migrated.expert_output.weight.add_(noise)
            migrated.trust_output.weight.copy_(source.network[4].weight[1:2])
            migrated.trust_output.bias.copy_(source.network[4].bias[1:2])
        return migrated

    @staticmethod
    def _unit_rms(field: Tensor) -> Tensor:
        rms = (
            field.square()
            .mean(dim=(-3, -2, -1), keepdim=True)
            .add(1.0e-12)
            .sqrt()
            .detach()
        )
        return field / rms

    def forward(
        self,
        theta: Tensor,
        observation: Tensor,
        total_gradient: Tensor,
        previous_update: Tensor,
        field_shape: tuple[int, int, int],
        *,
        forced_expert: Optional[int] = None,
        forced_routing: Optional[Tensor] = None,
        global_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if len(field_shape) != 3 or any(int(size) < 1 for size in field_shape):
            raise ValueError("field_shape must be a positive (C,H,W) tuple")
        channels, height, width = map(int, field_shape)
        expected = channels * height * width
        tensors = (theta, observation, total_gradient, previous_update)
        if any(tensor.ndim != 2 or tensor.shape != theta.shape for tensor in tensors):
            raise ValueError("spatial field tensors must share shape [B,P]")
        if theta.shape[1] != expected:
            raise ValueError("field_shape product must equal the parameter count")
        batch = theta.shape[0]
        current = theta.reshape(batch, channels, height, width)
        measured = observation.to(theta).reshape(batch, channels, height, width)
        gradient = total_gradient.reshape(batch, channels, height, width)
        previous = previous_update.reshape(batch, channels, height, width)
        current_mean = current.mean(dim=1, keepdim=True).expand_as(current)
        measured_mean = measured.mean(dim=1, keepdim=True).expand_as(measured)
        gradient_mean = gradient.mean(dim=1, keepdim=True).expand_as(gradient)
        features = torch.stack(
            (
                current,
                measured,
                self._unit_rms(gradient),
                self._unit_rms(previous),
                current - measured,
                current_mean,
                measured_mean,
                self._unit_rms(gradient_mean),
            ),
            dim=2,
        ).reshape(batch * channels, self.FEATURE_CHANNELS, height, width)
        hidden = self.trunk(features)
        pooled_mean = hidden.mean(dim=(-2, -1))
        pooled_rms = hidden.square().mean(dim=(-2, -1)).add(1.0e-12).sqrt()
        descriptor = torch.cat((pooled_mean, pooled_rms), dim=-1)
        descriptor = descriptor.reshape(batch, channels, -1).mean(dim=1)
        if self.global_feature_dim:
            if global_features is None or global_features.shape != (
                batch,
                self.global_feature_dim,
            ):
                raise ValueError(
                    "global spatial router features have the wrong shape"
                )
            descriptor = torch.cat(
                (descriptor, global_features.to(descriptor)), dim=-1
            )
        elif global_features is not None:
            raise ValueError("this spatial router does not use global features")
        predicted_routing = torch.softmax(self.router(descriptor), dim=-1)
        if forced_expert is not None and forced_routing is not None:
            raise ValueError("forced_expert and forced_routing are mutually exclusive")
        if forced_routing is not None:
            if forced_routing.shape != predicted_routing.shape:
                raise ValueError("forced routing must have shape [B,experts]")
            if torch.any(forced_routing < 0.0) or not torch.allclose(
                forced_routing.sum(dim=-1),
                torch.ones(batch, device=theta.device, dtype=theta.dtype),
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise ValueError("forced routing must lie on the probability simplex")
            routing = forced_routing.to(predicted_routing)
        elif forced_expert is None:
            routing = predicted_routing
        else:
            if forced_expert < 0 or forced_expert >= self.experts:
                raise ValueError("forced spatial expert is out of range")
            routing = torch.zeros_like(predicted_routing)
            routing[:, forced_expert] = 1.0
        expert_logits = self.expert_output(hidden).reshape(
            batch, channels, self.experts, height, width
        )
        mixed_logit = (
            routing[:, None, :, None, None] * expert_logits
        ).sum(dim=2)
        update = self.max_update * torch.tanh(mixed_logit)
        trust_offset = self.max_trust_offset * torch.tanh(
            self.trust_output(hidden)
        )
        trust_offset = trust_offset.reshape(batch, channels, -1).mean(dim=(1, 2))
        self.last_routing_weights = routing
        self.last_predicted_routing_weights = predicted_routing
        return update.reshape(batch, expected), trust_offset


class SpatialFieldIndependentMixtureUpdateHead(SpatialFieldMixtureUpdateHead):
    """Full-trunk spatial experts routed only by global factor geometry."""

    def __init__(
        self,
        hidden_dim: int = 32,
        experts: int = 2,
        router_hidden_dim: int = 16,
        global_feature_dim: int = 9,
        include_null_expert: bool = False,
        max_update: float = 0.1,
        max_trust_offset: float = 4.0,
    ) -> None:
        nn.Module.__init__(self)
        if hidden_dim < 8 or experts < 2 or router_hidden_dim < 4:
            raise ValueError("invalid independent spatial mixture dimensions")
        if global_feature_dim < 1:
            raise ValueError("independent experts require global router features")
        self.hidden_dim = int(hidden_dim)
        self.experts = int(experts)
        self.router_hidden_dim = int(router_hidden_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.expert_router_feature_dim = int(global_feature_dim)
        self.include_null_expert = bool(include_null_expert)
        self.routing_options = self.experts + int(self.include_null_expert)
        self.max_update = float(max_update)
        self.max_trust_offset = float(max_trust_offset)
        self.expert_trunks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.FEATURE_CHANNELS, hidden_dim, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GELU(),
                )
                for _ in range(experts)
            ]
        )
        self.expert_update_outputs = nn.ModuleList(
            [nn.Conv2d(hidden_dim, 1, 3, padding=1) for _ in range(experts)]
        )
        self.expert_trust_outputs = nn.ModuleList(
            [nn.Conv2d(hidden_dim, 1, 3, padding=1) for _ in range(experts)]
        )
        self.router = nn.Sequential(
            nn.Linear(global_feature_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, self.routing_options),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.last_routing_weights: Optional[Tensor] = None
        self.last_predicted_routing_weights: Optional[Tensor] = None

    @classmethod
    def from_single_head(
        cls,
        source: SpatialFieldUpdateHead,
        *,
        experts: int = 2,
        router_hidden_dim: int = 16,
        global_feature_dim: int = 9,
        perturbation: float = 1.0e-3,
        seed: int = 0,
    ) -> "SpatialFieldIndependentMixtureUpdateHead":
        hidden_dim = int(source.network[0].out_channels)
        migrated = cls(
            hidden_dim=hidden_dim,
            experts=experts,
            router_hidden_dim=router_hidden_dim,
            global_feature_dim=global_feature_dim,
            max_update=source.max_update,
            max_trust_offset=source.max_trust_offset,
        ).to(source.network[0].weight)
        with torch.no_grad():
            generator = torch.Generator(
                device=source.network[0].weight.device
            ).manual_seed(seed)
            noises = perturbation * torch.randn(
                (experts,) + tuple(source.network[4].weight[0].shape),
                generator=generator,
                device=source.network[4].weight.device,
                dtype=source.network[4].weight.dtype,
            )
            noises = noises - noises.mean(dim=0, keepdim=True)
            for index in range(experts):
                trunk = migrated.expert_trunks[index]
                trunk[0].weight.copy_(source.network[0].weight)
                trunk[0].bias.copy_(source.network[0].bias)
                trunk[2].weight.copy_(source.network[2].weight)
                trunk[2].bias.copy_(source.network[2].bias)
                migrated.expert_update_outputs[index].weight.copy_(
                    source.network[4].weight[0:1] + noises[index : index + 1]
                )
                migrated.expert_update_outputs[index].bias.copy_(
                    source.network[4].bias[0:1]
                )
                migrated.expert_trust_outputs[index].weight.copy_(
                    source.network[4].weight[1:2]
                )
                migrated.expert_trust_outputs[index].bias.copy_(
                    source.network[4].bias[1:2]
                )
        return migrated

    def forward(
        self,
        theta: Tensor,
        observation: Tensor,
        total_gradient: Tensor,
        previous_update: Tensor,
        field_shape: tuple[int, int, int],
        *,
        forced_expert: Optional[int] = None,
        forced_routing: Optional[Tensor] = None,
        global_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if len(field_shape) != 3 or any(int(size) < 1 for size in field_shape):
            raise ValueError("field_shape must be a positive (C,H,W) tuple")
        channels, height, width = map(int, field_shape)
        expected = channels * height * width
        tensors = (theta, observation, total_gradient, previous_update)
        if any(tensor.ndim != 2 or tensor.shape != theta.shape for tensor in tensors):
            raise ValueError("spatial field tensors must share shape [B,P]")
        if theta.shape[1] != expected:
            raise ValueError("field_shape product must equal the parameter count")
        batch = theta.shape[0]
        if global_features is None or global_features.shape != (
            batch,
            self.global_feature_dim,
        ):
            raise ValueError("global spatial router features have the wrong shape")
        current = theta.reshape(batch, channels, height, width)
        measured = observation.to(theta).reshape(batch, channels, height, width)
        gradient = total_gradient.reshape(batch, channels, height, width)
        previous = previous_update.reshape(batch, channels, height, width)
        current_mean = current.mean(dim=1, keepdim=True).expand_as(current)
        measured_mean = measured.mean(dim=1, keepdim=True).expand_as(measured)
        gradient_mean = gradient.mean(dim=1, keepdim=True).expand_as(gradient)
        features = torch.stack(
            (
                current,
                measured,
                self._unit_rms(gradient),
                self._unit_rms(previous),
                current - measured,
                current_mean,
                measured_mean,
                self._unit_rms(gradient_mean),
            ),
            dim=2,
        ).reshape(batch * channels, self.FEATURE_CHANNELS, height, width)
        predicted_routing = torch.softmax(
            self.router(
                global_features.to(theta)[..., : self.expert_router_feature_dim]
            ),
            dim=-1,
        )
        if forced_expert is not None and forced_routing is not None:
            raise ValueError("forced_expert and forced_routing are mutually exclusive")
        if forced_routing is not None:
            if forced_routing.shape != predicted_routing.shape:
                raise ValueError("forced routing must have shape [B,experts]")
            if torch.any(forced_routing < 0.0) or not torch.allclose(
                forced_routing.sum(dim=-1),
                torch.ones(batch, device=theta.device, dtype=theta.dtype),
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise ValueError("forced routing must lie on the probability simplex")
            routing = forced_routing.to(predicted_routing)
        elif forced_expert is None:
            routing = predicted_routing
        else:
            if forced_expert < 0 or forced_expert >= self.routing_options:
                raise ValueError("forced spatial expert is out of range")
            routing = torch.zeros_like(predicted_routing)
            routing[:, forced_expert] = 1.0
        update_logits = []
        trust_logits = []
        for trunk, update_output, trust_output in zip(
            self.expert_trunks,
            self.expert_update_outputs,
            self.expert_trust_outputs,
        ):
            hidden = trunk(features)
            update_logits.append(update_output(hidden))
            trust_logits.append(trust_output(hidden))
        update_logits = torch.stack(update_logits, dim=2).reshape(
            batch, channels, self.experts, height, width
        )
        trust_logits = torch.stack(trust_logits, dim=2).reshape(
            batch, channels, self.experts, height, width
        )
        # The optional final route is a parameter-free abstention path.  It
        # contributes neither update nor trust, recovering the compact solver
        # within the same recurrent call budget when selected purely.
        learned_routing = routing[:, : self.experts]
        route = learned_routing[:, None, :, None, None]
        update = self.max_update * torch.tanh((route * update_logits).sum(dim=2))
        trust = self.max_trust_offset * torch.tanh(
            (route * trust_logits).sum(dim=2)
        )
        self.last_routing_weights = routing
        self.last_predicted_routing_weights = predicted_routing
        return update.reshape(batch, expected), trust.mean(dim=(1, 2, 3))


class HandwrittenSpatialResidualBranch(nn.Module):
    """Small channel-shared correction built on auditable spatial operators.

    The branch deliberately avoids a restoration backbone.  It constructs its
    input from a fixed binomial low pass, finite differences, high-frequency
    residuals, and local residual energy.  Treating every image channel as an
    independent batch element preserves channel-coordinate equivariance.  The
    last convolution is zero initialized so enabling the branch is exactly
    function preserving.
    """

    FEATURE_CHANNELS = 10

    def __init__(self, hidden_dim: int = 48, max_update: float = 0.1) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("handwritten spatial branch hidden_dim must be at least 8")
        if max_update <= 0.0:
            raise ValueError("handwritten spatial branch max_update must be positive")
        self.hidden_dim = int(hidden_dim)
        self.max_update = float(max_update)
        self.network = nn.Sequential(
            nn.Conv2d(self.FEATURE_CHANNELS, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _unit_rms(value: Tensor) -> Tensor:
        scale = value.square().mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1.0e-6)
        return value / scale

    def forward(
        self,
        theta: Tensor,
        observation: Tensor,
        total_gradient: Tensor,
        previous_update: Tensor,
        field_shape: tuple[int, int, int],
    ) -> Tensor:
        channels, height, width = map(int, field_shape)
        expected = channels * height * width
        tensors = (theta, observation, total_gradient, previous_update)
        if any(tensor.ndim != 2 or tensor.shape != theta.shape for tensor in tensors):
            raise ValueError("handwritten branch tensors must share shape [B,P]")
        if theta.shape[1] != expected:
            raise ValueError("handwritten branch field_shape product must match P")
        if height < 3 or width < 3:
            raise ValueError("handwritten branch requires spatial dimensions >= 3")

        batch = theta.shape[0]
        current = theta.reshape(batch, channels, height, width)
        measured = observation.to(theta).reshape(batch, channels, height, width)
        gradient = total_gradient.reshape(batch, channels, height, width)
        previous = previous_update.reshape(batch, channels, height, width)
        low_pass = separable_binomial_blur(measured)
        high_pass = measured - low_pass
        residual = current - measured
        residual_low_pass = separable_binomial_blur(residual)
        residual_high_pass = residual - residual_low_pass
        dx, dy = finite_difference_xy(measured)
        gradient_magnitude = torch.sqrt(dx.square() + dy.square() + 1.0e-8)
        local_high_energy = torch.sqrt(
            separable_binomial_blur(high_pass.square()) + 1.0e-8
        )
        features = torch.stack(
            (
                current,
                measured,
                low_pass,
                high_pass,
                local_high_energy,
                gradient_magnitude,
                residual,
                residual_high_pass,
                self._unit_rms(gradient),
                self._unit_rms(previous),
            ),
            dim=2,
        ).reshape(batch * channels, self.FEATURE_CHANNELS, height, width)
        correction = self.max_update * torch.tanh(self.network(features))
        return correction.reshape(batch, expected)


class SpatialFieldHardAbstainingMixtureUpdateHead(
    SpatialFieldIndependentMixtureUpdateHead
):
    """Lock each rollout to either the inherited experts or exact compact path."""

    PROBE_FEATURE_DIM = 32

    def __init__(
        self,
        hidden_dim: int = 32,
        experts: int = 2,
        router_hidden_dim: int = 16,
        abstention_hidden_dim: int = 32,
        abstention_threshold: float = 0.9995,
        expert_router_feature_dim: int = 9,
        use_specialization_override: bool = False,
        specialization_override_hidden_dim: int = 32,
        specialization_override_threshold: float = 0.9995,
        safe_residual_expert: Optional[int] = None,
        evidence_gate_hidden_dim: int = 32,
        safe_residual_initial_scale: float = 0.0,
        safe_residual_evidence_threshold: float = 0.0,
        safe_spatial_branch_hidden_dim: Optional[int] = None,
        safe_analytic_risk_low: Optional[float] = None,
        safe_analytic_risk_high: Optional[float] = None,
        safe_analytic_agreement_low: Optional[float] = None,
        safe_analytic_agreement_high: Optional[float] = None,
        max_update: float = 0.1,
        max_trust_offset: float = 4.0,
    ) -> None:
        if not 1 <= expert_router_feature_dim <= self.PROBE_FEATURE_DIM:
            raise ValueError("expert_router_feature_dim must lie in [1,32]")
        super().__init__(
            hidden_dim=hidden_dim,
            experts=experts,
            router_hidden_dim=router_hidden_dim,
            global_feature_dim=expert_router_feature_dim,
            include_null_expert=False,
            max_update=max_update,
            max_trust_offset=max_trust_offset,
        )
        if not 0.5 <= abstention_threshold < 1.0:
            raise ValueError("abstention_threshold must lie in [0.5,1)")
        self.global_feature_dim = self.PROBE_FEATURE_DIM
        self.include_null_expert = True
        self.routing_options = self.experts + 1
        self.abstention_hidden_dim = int(abstention_hidden_dim)
        self.abstention_threshold = float(abstention_threshold)
        self.abstention_router = nn.Sequential(
            nn.LayerNorm(self.PROBE_FEATURE_DIM),
            nn.Linear(self.PROBE_FEATURE_DIM, abstention_hidden_dim),
            nn.GELU(),
            nn.Linear(abstention_hidden_dim, 1),
        )
        if use_specialization_override:
            if specialization_override_hidden_dim < 8:
                raise ValueError("specialization override hidden_dim must be at least 8")
            if not 0.5 <= specialization_override_threshold < 1.0:
                raise ValueError("specialization override threshold must lie in [0.5,1)")
            self.specialization_override_router: Optional[nn.Module] = nn.Sequential(
                nn.LayerNorm(self.PROBE_FEATURE_DIM),
                nn.Linear(self.PROBE_FEATURE_DIM, specialization_override_hidden_dim),
                nn.GELU(),
                nn.Linear(specialization_override_hidden_dim, 1),
            )
            nn.init.zeros_(self.specialization_override_router[-1].weight)
            nn.init.constant_(self.specialization_override_router[-1].bias, -8.0)
        else:
            self.specialization_override_router = None
        self.specialization_override_hidden_dim = int(
            specialization_override_hidden_dim
        )
        self.specialization_override_threshold = float(
            specialization_override_threshold
        )
        if safe_residual_expert is not None and not 0 <= safe_residual_expert < experts:
            raise ValueError("safe residual expert must index a learned expert")
        if evidence_gate_hidden_dim < 8:
            raise ValueError("evidence gate hidden_dim must be at least 8")
        if not 0.0 <= safe_residual_initial_scale <= 1.0:
            raise ValueError("safe residual initial scale must lie in [0,1]")
        if not 0.0 <= safe_residual_evidence_threshold < 1.0:
            raise ValueError("safe residual evidence threshold must lie in [0,1)")
        self.safe_residual_expert = safe_residual_expert
        self.evidence_gate_hidden_dim = int(evidence_gate_hidden_dim)
        self.safe_residual_evidence_threshold = float(
            safe_residual_evidence_threshold
        )
        if safe_residual_expert is None:
            self.evidence_gate: Optional[nn.Module] = None
            self.register_parameter("safe_residual_scale", None)
        else:
            self.evidence_gate = nn.Sequential(
                nn.LayerNorm(self.PROBE_FEATURE_DIM),
                nn.Linear(self.PROBE_FEATURE_DIM, evidence_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(evidence_gate_hidden_dim, 1),
            )
            nn.init.zeros_(self.evidence_gate[-1].weight)
            nn.init.zeros_(self.evidence_gate[-1].bias)
            # ReZero makes the image update exactly null at initialization.
            # Training projects this scalar back to [0,1] after each update.
            self.safe_residual_scale = nn.Parameter(
                torch.tensor(float(safe_residual_initial_scale))
            )
        if safe_spatial_branch_hidden_dim is not None:
            if safe_residual_expert is None:
                raise ValueError("safe spatial branch requires a safe residual expert")
            self.safe_spatial_branch: Optional[HandwrittenSpatialResidualBranch] = (
                HandwrittenSpatialResidualBranch(
                    hidden_dim=int(safe_spatial_branch_hidden_dim),
                    max_update=max_update,
                )
            )
        else:
            self.safe_spatial_branch = None
        self.safe_spatial_branch_hidden_dim = (
            None
            if safe_spatial_branch_hidden_dim is None
            else int(safe_spatial_branch_hidden_dim)
        )
        if (safe_analytic_risk_low is None) != (safe_analytic_risk_high is None):
            raise ValueError("analytic risk low/high must be configured together")
        if safe_analytic_risk_low is not None:
            if safe_residual_expert is None:
                raise ValueError("analytic risk requires a safe residual expert")
            if not float(safe_analytic_risk_high) > float(safe_analytic_risk_low) >= 0.0:
                raise ValueError("analytic risk requires 0 <= low < high")
        self.safe_analytic_risk_low = (
            None if safe_analytic_risk_low is None else float(safe_analytic_risk_low)
        )
        self.safe_analytic_risk_high = (
            None if safe_analytic_risk_high is None else float(safe_analytic_risk_high)
        )
        if (safe_analytic_agreement_low is None) != (
            safe_analytic_agreement_high is None
        ):
            raise ValueError("analytic agreement low/high must be configured together")
        if safe_analytic_agreement_low is not None:
            if safe_analytic_risk_low is None:
                raise ValueError("analytic agreement requires analytic risk")
            if not 0.0 <= float(safe_analytic_agreement_low) < float(
                safe_analytic_agreement_high
            ) <= 1.0:
                raise ValueError("analytic agreement requires 0 <= low < high <= 1")
        self.safe_analytic_agreement_low = (
            None
            if safe_analytic_agreement_low is None
            else float(safe_analytic_agreement_low)
        )
        self.safe_analytic_agreement_high = (
            None
            if safe_analytic_agreement_high is None
            else float(safe_analytic_agreement_high)
        )
        self._locked_null_decision: Optional[Tensor] = None
        self._locked_specialization_override: Optional[Tensor] = None
        self._locked_safe_residual_decision: Optional[Tensor] = None
        self.last_null_probability: Optional[Tensor] = None
        self.last_specialization_override_probability: Optional[Tensor] = None
        self.last_spatial_policy_active: Optional[Tensor] = None
        self.last_safe_residual_confidence: Optional[Tensor] = None
        self.last_safe_residual_evidence_probability: Optional[Tensor] = None
        self.last_safe_residual_initial_evidence_probability: Optional[Tensor] = None
        self.last_safe_spatial_branch_update: Optional[Tensor] = None
        self._locked_safe_analytic_risk: Optional[Tensor] = None
        self.last_safe_analytic_noise_sigma: Optional[Tensor] = None
        self.last_safe_analytic_highpass_sigma: Optional[Tensor] = None
        self.last_safe_analytic_agreement: Optional[Tensor] = None
        self.last_safe_analytic_risk_confidence: Optional[Tensor] = None

    def enable_safe_spatial_branch(self, hidden_dim: int = 48) -> None:
        """Attach a zero-output residual branch without changing the function."""

        if self.safe_residual_expert is None:
            raise RuntimeError("safe spatial branch requires a safe residual expert")
        if self.safe_spatial_branch is not None:
            raise RuntimeError("safe spatial branch is already enabled")
        reference = next(self.parameters())
        self.safe_spatial_branch = HandwrittenSpatialResidualBranch(
            hidden_dim=hidden_dim,
            max_update=self.max_update,
        ).to(reference)
        self.safe_spatial_branch_hidden_dim = int(hidden_dim)

    def configure_safe_analytic_risk(
        self,
        *,
        low: float,
        high: float,
        agreement_low: Optional[float] = None,
        agreement_high: Optional[float] = None,
    ) -> None:
        """Configure a parameter-free, rollout-locked MAD safety multiplier."""

        if self.safe_residual_expert is None:
            raise RuntimeError("analytic risk requires a safe residual expert")
        if not high > low >= 0.0:
            raise ValueError("analytic risk requires 0 <= low < high")
        if (agreement_low is None) != (agreement_high is None):
            raise ValueError("analytic agreement low/high must be configured together")
        if agreement_low is not None and not 0.0 <= agreement_low < agreement_high <= 1.0:
            raise ValueError("analytic agreement requires 0 <= low < high <= 1")
        self.safe_analytic_risk_low = float(low)
        self.safe_analytic_risk_high = float(high)
        self.safe_analytic_agreement_low = (
            None if agreement_low is None else float(agreement_low)
        )
        self.safe_analytic_agreement_high = (
            None if agreement_high is None else float(agreement_high)
        )

    def begin_rollout(self) -> None:
        self._locked_null_decision = None
        self._locked_specialization_override = None
        self._locked_safe_residual_decision = None
        self.last_null_probability = None
        self.last_specialization_override_probability = None
        self.last_safe_residual_confidence = None
        self.last_safe_residual_evidence_probability = None
        self.last_safe_residual_initial_evidence_probability = None
        self.last_safe_spatial_branch_update = None
        self._locked_safe_analytic_risk = None
        self.last_safe_analytic_noise_sigma = None
        self.last_safe_analytic_highpass_sigma = None
        self.last_safe_analytic_agreement = None
        self.last_safe_analytic_risk_confidence = None

    def safe_analytic_risk_confidence(
        self, observation: Tensor, field_shape: tuple[int, int, int]
    ) -> Tensor:
        """Return the no-target MAD risk multiplier, locked for the rollout."""

        if self.safe_analytic_risk_low is None or self.safe_analytic_risk_high is None:
            raise RuntimeError("analytic risk is not configured")
        channels, height, width = map(int, field_shape)
        expected = channels * height * width
        if observation.ndim != 2 or observation.shape[1] != expected:
            raise ValueError("analytic risk observation has the wrong field layout")
        if self._locked_safe_analytic_risk is None:
            image = observation.reshape(observation.shape[0], channels, height, width)
            features = analytic_noise_risk_features(image)
            sigma = features[:, 0]
            highpass_sigma = features[:, 1]
            confidence = smoothstep_risk_confidence(
                sigma,
                low=self.safe_analytic_risk_low,
                high=self.safe_analytic_risk_high,
            )
            if self.safe_analytic_agreement_low is not None:
                agreement = torch.minimum(sigma, highpass_sigma) / (
                    torch.maximum(sigma, highpass_sigma) + 1.0e-8
                )
                confidence = confidence * smoothstep_risk_confidence(
                    agreement,
                    low=self.safe_analytic_agreement_low,
                    high=self.safe_analytic_agreement_high,
                )
                self.last_safe_analytic_agreement = agreement.detach()
            self.last_safe_analytic_noise_sigma = sigma.detach()
            self.last_safe_analytic_highpass_sigma = highpass_sigma.detach()
            self._locked_safe_analytic_risk = confidence.detach()
        self.last_safe_analytic_risk_confidence = self._locked_safe_analytic_risk
        return self._locked_safe_analytic_risk.to(observation)

    def safe_residual_confidence(self, global_features: Tensor) -> Tensor:
        """Return a probe-conditioned [0,1] multiplier for the protected expert."""

        if self.evidence_gate is None or self.safe_residual_scale is None:
            raise RuntimeError("safe residual gating is not enabled")
        if global_features.ndim != 2 or global_features.shape[-1] != self.PROBE_FEATURE_DIM:
            raise ValueError("safe residual gating requires 32 probe features")
        evidence_probability = torch.sigmoid(
            self.evidence_gate(global_features).squeeze(-1)
        )
        self.last_safe_residual_evidence_probability = evidence_probability
        if self.last_safe_residual_initial_evidence_probability is None:
            self.last_safe_residual_initial_evidence_probability = evidence_probability
        if self.safe_residual_evidence_threshold > 0.0:
            # A task-free fail-closed deployment decision.  The gate itself is
            # trained through its evidence supervision; an accepted expert
            # receives an unattenuated gradient instead of a nearly-zero one.
            # Like M21 hard abstention, the decision is made once from the
            # initial program response and cannot wake or sleep mid-rollout.
            if self._locked_safe_residual_decision is None:
                self._locked_safe_residual_decision = (
                    evidence_probability >= self.safe_residual_evidence_threshold
                ).detach()
            evidence = self._locked_safe_residual_decision.to(
                evidence_probability.dtype
            )
        else:
            evidence = evidence_probability
        scale = self.safe_residual_scale.clamp(0.0, 1.0)
        return scale * evidence

    def abstention_logit(self, global_features: Tensor) -> Tensor:
        if global_features.ndim != 2 or global_features.shape[-1] != self.PROBE_FEATURE_DIM:
            raise ValueError("hard abstention requires 32 probe features")
        return self.abstention_router(global_features).squeeze(-1)

    def specialization_override_logit(self, global_features: Tensor) -> Tensor:
        if self.specialization_override_router is None:
            raise RuntimeError("specialization override is not enabled")
        if global_features.ndim != 2 or global_features.shape[-1] != self.PROBE_FEATURE_DIM:
            raise ValueError("specialization override requires 32 probe features")
        return self.specialization_override_router(global_features).squeeze(-1)

    def forward(
        self,
        theta: Tensor,
        observation: Tensor,
        total_gradient: Tensor,
        previous_update: Tensor,
        field_shape: tuple[int, int, int],
        *,
        forced_expert: Optional[int] = None,
        forced_routing: Optional[Tensor] = None,
        global_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if global_features is None or global_features.shape != (
            theta.shape[0],
            self.PROBE_FEATURE_DIM,
        ):
            raise ValueError("hard abstention requires global probe features")
        if forced_expert is not None and forced_routing is not None:
            raise ValueError("forced_expert and forced_routing are mutually exclusive")

        if forced_routing is not None:
            if forced_routing.shape != (theta.shape[0], self.routing_options):
                raise ValueError("forced routing has the wrong option count")
            if torch.any(forced_routing < 0.0) or not torch.allclose(
                forced_routing.sum(dim=-1),
                torch.ones(theta.shape[0], device=theta.device, dtype=theta.dtype),
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise ValueError("forced routing must lie on the probability simplex")
            actual_routing = forced_routing.to(theta)
            active = 1.0 - actual_routing[:, -1]
            learned = actual_routing[:, : self.experts]
            normalized = learned / active[:, None].clamp_min(1.0e-8)
            normalized = torch.where(
                active[:, None] > 0.0,
                normalized,
                torch.nn.functional.one_hot(
                    torch.zeros(theta.shape[0], device=theta.device, dtype=torch.long),
                    self.experts,
                ).to(theta),
            )
            update, trust = super().forward(
                theta,
                observation,
                total_gradient,
                previous_update,
                field_shape,
                forced_routing=normalized,
                global_features=global_features,
            )
            predicted_routing = actual_routing
        elif forced_expert is not None:
            if forced_expert < 0 or forced_expert >= self.routing_options:
                raise ValueError("forced spatial expert is out of range")
            active = theta.new_zeros(theta.shape[0]) if forced_expert == self.experts else theta.new_ones(theta.shape[0])
            update, trust = super().forward(
                theta,
                observation,
                total_gradient,
                previous_update,
                field_shape,
                forced_expert=min(forced_expert, self.experts - 1),
                global_features=global_features,
            )
            actual_routing = theta.new_zeros(theta.shape[0], self.routing_options)
            actual_routing[:, forced_expert] = 1.0
            predicted_routing = actual_routing
        else:
            update, trust = super().forward(
                theta,
                observation,
                total_gradient,
                previous_update,
                field_shape,
                global_features=global_features,
            )
            expert_routing = self.last_predicted_routing_weights
            null_probability = torch.sigmoid(self.abstention_logit(global_features.to(theta)))
            if self._locked_null_decision is None:
                original_null = null_probability >= self.abstention_threshold
                if self.specialization_override_router is None:
                    specialization_override = torch.zeros_like(original_null)
                    specialization_probability = torch.zeros_like(null_probability)
                else:
                    specialization_probability = torch.sigmoid(
                        self.specialization_override_logit(global_features.to(theta))
                    )
                    specialization_override = (
                        specialization_probability
                        >= self.specialization_override_threshold
                    )
                self._locked_specialization_override = specialization_override.detach()
                self._locked_null_decision = (
                    original_null & (~specialization_override)
                ).detach()
                self.last_specialization_override_probability = (
                    specialization_probability
                )
            active = (~self._locked_null_decision).to(theta.dtype)
            actual_routing = torch.cat(
                (expert_routing * active[:, None], (1.0 - active)[:, None]), dim=-1
            )
            predicted_routing = torch.cat(
                (
                    expert_routing * (1.0 - null_probability)[:, None],
                    null_probability[:, None],
                ),
                dim=-1,
            )
            self.last_null_probability = null_probability

        if self.safe_residual_expert is None:
            safe_confidence = None
            protected_mass = theta.new_zeros(theta.shape[0])
        else:
            protected_mass = actual_routing[:, self.safe_residual_expert]
            evidence = self.safe_residual_confidence(global_features.to(theta))
            if self.safe_analytic_risk_low is not None:
                evidence = evidence * self.safe_analytic_risk_confidence(
                    observation.to(theta), field_shape
                )
            safe_confidence = 1.0 - protected_mass + protected_mass * evidence
        if self.safe_spatial_branch is not None:
            branch_update = self.safe_spatial_branch(
                theta,
                observation,
                total_gradient,
                previous_update,
                field_shape,
            )
            branch_update = protected_mass[:, None] * branch_update
            update = torch.clamp(
                update + branch_update,
                min=-self.max_update,
                max=self.max_update,
            )
            self.last_safe_spatial_branch_update = branch_update
        update = update * active[:, None]
        trust = trust * active
        self.last_routing_weights = actual_routing
        self.last_predicted_routing_weights = predicted_routing
        self.last_spatial_policy_active = active.bool()
        self.last_safe_residual_confidence = safe_confidence
        return update, trust


class DescentAnchoredDifferentialFactorSolver(nn.Module):
    """Mix learned unit-factor geometry with a raw-total-gradient anchor."""

    FEEDBACK_DIM = 3
    TOPOLOGY_FEATURE_DIM = 7
    TOPOLOGY_CONTROL_DIM = 4
    BLOCK_CONTEXT_FEATURE_DIM = 6
    BLOCK_CONTEXT_CONTROL_DIM = 4
    SCHEMA_EXPERT_FEATURE_DIM = 14
    SCHEMA_EXPERT_CONTROL_DIM = 4
    SCHEMA_LOCAL_TOPOLOGY_FEATURE_DIM = 4

    def __init__(
        self,
        hidden_dim: int = 64,
        heads: int = 4,
        max_step_norm: float = 0.25,
        eps: float = 1.0e-8,
        use_curvature_anchor: bool = False,
        curvature_damping: float = 1.0e-2,
        curvature_max_block_dim: Optional[int] = None,
        use_topology_context: bool = False,
        use_block_context: bool = False,
        use_context_gate: bool = False,
        context_gate_hidden_dim: int = 16,
        use_schema_expert: bool = False,
        schema_expert_hidden_dim: int = 128,
        schema_expert_feature_version: str = "absolute_v1",
        schema_expert_hard_admission: bool = False,
        schema_expert_direction_version: str = "three_direction_v1",
        use_schema_local_admission: bool = False,
        schema_local_admission_hidden_dim: int = 32,
        schema_local_admission_feature_version: str = "expert_v1",
        use_fast_aggregate_expert: bool = False,
        fast_aggregate_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if curvature_damping <= 0:
            raise ValueError("curvature_damping must be positive")
        if curvature_max_block_dim is not None and curvature_max_block_dim < 1:
            raise ValueError("curvature_max_block_dim must be positive when set")
        self.hidden_dim = hidden_dim
        self.max_step_norm = max_step_norm
        self.eps = eps
        self.use_curvature_anchor = use_curvature_anchor
        self.curvature_damping = curvature_damping
        self.curvature_max_block_dim = curvature_max_block_dim
        self.use_topology_context = use_topology_context
        self.use_block_context = use_block_context
        self.use_context_gate = use_context_gate
        self.context_gate_hidden_dim = int(context_gate_hidden_dim)
        self.use_schema_expert = use_schema_expert
        self.schema_expert_hidden_dim = int(schema_expert_hidden_dim)
        self.schema_expert_feature_version = str(schema_expert_feature_version)
        self.schema_expert_hard_admission = bool(schema_expert_hard_admission)
        self.schema_expert_direction_version = str(schema_expert_direction_version)
        self.use_schema_local_admission = bool(use_schema_local_admission)
        self.schema_local_admission_hidden_dim = int(
            schema_local_admission_hidden_dim
        )
        self.schema_local_admission_feature_version = str(
            schema_local_admission_feature_version
        )
        self.fast_aggregate_hidden_dim = int(fast_aggregate_hidden_dim)
        if self.fast_aggregate_hidden_dim < 8:
            raise ValueError("fast aggregate hidden_dim must be at least eight")
        # Deployment-only mathematical pacing.  This flag owns no parameters
        # and is intentionally absent from historical checkpoint metadata.
        self.use_schema_topology_pacing = False
        self.last_schema_topology_pacing_active: Optional[Tensor] = None
        if use_context_gate and not (use_topology_context or use_block_context):
            raise ValueError("context gate requires at least one context branch")
        if context_gate_hidden_dim < 4:
            raise ValueError("context gate hidden_dim must be at least four")
        if use_schema_expert and not use_context_gate:
            raise ValueError("schema expert requires context admission")
        if use_schema_local_admission and not use_schema_expert:
            raise ValueError("local schema admission requires a schema expert")
        if schema_local_admission_hidden_dim < 4:
            raise ValueError("local schema admission hidden_dim must be at least four")
        if schema_local_admission_feature_version not in (
            "expert_v1",
            "expert_plus_incidence_v2",
        ):
            raise ValueError("unknown local schema admission feature version")
        if schema_expert_hidden_dim < 8:
            raise ValueError("schema expert hidden_dim must be at least eight")
        if schema_expert_feature_version not in ("absolute_v1", "bounded_ratio_v2"):
            raise ValueError("unknown schema expert feature version")
        if schema_expert_direction_version not in (
            "three_direction_v1",
            "current_direction_v2",
        ):
            raise ValueError("unknown schema expert direction version")
        self.schema_expert_control_dim = (
            5 if schema_expert_direction_version == "current_direction_v2" else 4
        )
        self.factor_solver = DifferentialFactorSolver(
            hidden_dim=hidden_dim,
            heads=heads,
            global_feature_dim=self.FEEDBACK_DIM,
            max_step_norm=max_step_norm,
            eps=eps,
        )
        self.basis_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.FEEDBACK_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.basis_head[-1].weight)
        with torch.no_grad():
            self.basis_head[-1].bias.copy_(torch.tensor([0.0, 1.0]))
        if use_curvature_anchor:
            self.curvature_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(2 * hidden_dim + self.FEEDBACK_DIM, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.curvature_head[-1].weight)
            nn.init.constant_(self.curvature_head[-1].bias, -4.0)
        else:
            self.curvature_head = None
        if use_topology_context:
            context_hidden = max(hidden_dim // 2, 8)
            self.topology_context_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(self.TOPOLOGY_FEATURE_DIM, context_hidden),
                nn.GELU(),
                nn.Linear(context_hidden, self.TOPOLOGY_CONTROL_DIM),
            )
            nn.init.zeros_(self.topology_context_head[-1].weight)
            nn.init.zeros_(self.topology_context_head[-1].bias)
        else:
            self.topology_context_head = None
        if use_block_context:
            context_hidden = max(hidden_dim // 2, 8)
            self.block_context_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(self.BLOCK_CONTEXT_FEATURE_DIM, context_hidden),
                nn.GELU(),
                nn.Linear(context_hidden, self.BLOCK_CONTEXT_CONTROL_DIM),
            )
            nn.init.zeros_(self.block_context_head[-1].weight)
            nn.init.zeros_(self.block_context_head[-1].bias)
        else:
            self.block_context_head = None
        if use_context_gate:
            self.context_gate_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(self.TOPOLOGY_FEATURE_DIM, context_gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(context_gate_hidden_dim, 1),
            )
        else:
            self.context_gate_head = None
        if use_schema_expert:
            self.schema_expert_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(self.SCHEMA_EXPERT_FEATURE_DIM, schema_expert_hidden_dim),
                nn.SiLU(),
                nn.Linear(schema_expert_hidden_dim, schema_expert_hidden_dim),
                nn.SiLU(),
                nn.Linear(
                    schema_expert_hidden_dim, self.schema_expert_control_dim
                ),
            )
            nn.init.zeros_(self.schema_expert_head[-1].weight)
            nn.init.zeros_(self.schema_expert_head[-1].bias)
            self.schema_expert_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.schema_expert_head = None
            self.register_parameter("schema_expert_scale", None)
        if use_schema_local_admission:
            local_feature_dim = self.SCHEMA_EXPERT_FEATURE_DIM + (
                self.SCHEMA_LOCAL_TOPOLOGY_FEATURE_DIM
                if schema_local_admission_feature_version
                == "expert_plus_incidence_v2"
                else 0
            )
            self.schema_local_admission_feature_dim = local_feature_dim
            self.schema_local_admission_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(
                    local_feature_dim,
                    schema_local_admission_hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(schema_local_admission_hidden_dim, 1),
            )
            nn.init.zeros_(self.schema_local_admission_head[-1].weight)
            nn.init.zeros_(self.schema_local_admission_head[-1].bias)
        else:
            self.schema_local_admission_head = None
            self.schema_local_admission_feature_dim = 0
        self.last_context_gate: Optional[Tensor] = None
        self.last_schema_expert_blend: Optional[Tensor] = None
        self.last_schema_expert_weights: Optional[Tensor] = None
        self.last_schema_local_admission_probability: Optional[Tensor] = None
        self.last_schema_local_admission: Optional[Tensor] = None
        self.fast_aggregate_edge_head: Optional[nn.Module] = None
        self.fast_aggregate_block_head: Optional[nn.Module] = None
        if use_fast_aggregate_expert:
            self.configure_fast_aggregate_expert(
                hidden_dim=self.fast_aggregate_hidden_dim
            )

    def configure_fast_aggregate_expert(self, *, hidden_dim: int = 32) -> None:
        """Attach the low-launch, coordinate-equivariant compact-graph path.

        The path predicts only invariant scalar coefficients.  Updates remain
        linear combinations of normalized factor gradients, the total descent
        direction, and the previous accepted tangent, so blockwise orthogonal
        coordinate equivariance is preserved by construction.
        """

        if self.fast_aggregate_edge_head is not None:
            raise RuntimeError("fast aggregate expert is already configured")
        if hidden_dim < 8:
            raise ValueError("fast aggregate hidden_dim must be at least eight")
        self.fast_aggregate_hidden_dim = int(hidden_dim)
        self.fast_aggregate_edge_head = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.fast_aggregate_block_head = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.fast_aggregate_edge_head[-1].weight)
        nn.init.zeros_(self.fast_aggregate_edge_head[-1].bias)
        nn.init.zeros_(self.fast_aggregate_block_head[-1].weight)
        with torch.no_grad():
            # Start as a small, functioning normalized-gradient step.  The
            # unavailable history basis is masked exactly in forward().
            self.fast_aggregate_block_head[-1].bias.copy_(
                # Compact SE(3)/homography charts typically need an initial
                # normalized step near 1e-2; sigmoid(-3.2)*0.25 ~= 9.8e-3.
                torch.tensor((0.0, 2.0, -4.0, -3.2))
            )

    def _fast_aggregate_step(
        self,
        theta: Tensor,
        directions: Tensor,
        factor_values: Tensor,
        factor_mask: Tensor,
        block_id: Tensor,
        blocks: int,
        trust: Tensor,
        solver_state: Optional[DifferentialFactorState],
        topology_pacing_active: Tensor,
    ) -> DescentAnchoredStep:
        """Evaluate the attention-free compact-graph expert."""

        assert self.fast_aggregate_edge_head is not None
        assert self.fast_aggregate_block_head is not None
        batch, factors, parameters = directions.shape
        mask_float = factor_mask.to(theta.dtype)
        edge_energy = self.factor_solver._sum_coordinates_by_block(
            directions.square(), block_id, blocks
        ).transpose(1, 2)
        edge_norm = edge_energy.clamp_min(self.eps**2).sqrt()
        edge_mask = (edge_energy > self.eps**2) & factor_mask[:, None, :]
        unit_directions = directions / edge_norm.transpose(1, 2)[
            ..., block_id
        ].clamp_min(self.eps)

        total_gradient = (directions * mask_float[..., None]).sum(dim=1)
        total_energy = self.factor_solver._sum_coordinates_by_block(
            total_gradient.square(), block_id, blocks
        )
        total_norm = total_energy.clamp_min(self.eps**2).sqrt()
        total_descent = -total_gradient / total_norm[..., block_id].clamp_min(
            self.eps
        )
        previous_update = (
            torch.zeros_like(theta)
            if solver_state is None
            else solver_state.previous_update.to(theta)
        )
        previous_energy = self.factor_solver._sum_coordinates_by_block(
            previous_update.square(), block_id, blocks
        )
        previous_norm = previous_energy.clamp_min(self.eps**2).sqrt()
        previous_unit = previous_update / previous_norm[..., block_id].clamp_min(
            self.eps
        )
        history_available = previous_energy > self.eps**2

        def block_dot(first: Tensor, second: Tensor) -> Tensor:
            return self.factor_solver._sum_coordinates_by_block(
                first * second[:, None, :], block_id, blocks
            ).transpose(1, 2)

        edge_total_cosine = -block_dot(unit_directions, total_descent)
        edge_previous_cosine = block_dot(unit_directions, previous_unit)
        active_count = edge_mask.to(theta.dtype).sum(dim=-1).clamp_min(1.0)
        mean_edge_norm = (
            edge_norm * edge_mask.to(theta.dtype)
        ).sum(dim=-1) / active_count
        relative_edge_norm = torch.tanh(
            torch.log(
                (edge_norm + self.eps)
                / (mean_edge_norm[..., None] + self.eps)
            )
        )
        active_factor_count = mask_float.sum(dim=-1).clamp_min(1.0)
        mean_factor_value = (
            factor_values * mask_float
        ).sum(dim=-1) / active_factor_count
        relative_factor_value = torch.tanh(
            torch.log(
                (factor_values.abs() + self.eps)
                / (mean_factor_value.abs()[:, None] + self.eps)
            )
        )
        dimensions = torch.bincount(block_id, minlength=blocks).to(theta)
        dimension_fraction = torch.log1p(dimensions) / max(
            math.log1p(parameters), self.eps
        )
        edge_features = torch.stack(
            (
                relative_factor_value[:, None, :].expand(-1, blocks, -1),
                relative_edge_norm,
                edge_total_cosine,
                edge_previous_cosine,
                torch.tanh(factor_values)[:, None, :].expand(-1, blocks, -1),
                dimension_fraction[None, :, None].expand(batch, -1, factors),
                history_available.to(theta.dtype)[..., None].expand(-1, -1, factors),
                edge_mask.to(theta.dtype),
            ),
            dim=-1,
        )
        edge_logits = self.fast_aggregate_edge_head(edge_features).squeeze(-1)
        edge_logits = edge_logits.masked_fill(~edge_mask, -1.0e4)
        edge_weights = torch.softmax(edge_logits, dim=-1)
        edge_weights = edge_weights * edge_mask.to(theta.dtype)
        edge_weights = edge_weights / edge_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        coefficient_by_coordinate = edge_weights.transpose(1, 2)[..., block_id]
        learned_factor_descent = -(
            coefficient_by_coordinate * unit_directions
        ).sum(dim=1)

        def block_cosine(first: Tensor, second: Tensor) -> Tensor:
            numerator = self.factor_solver._sum_coordinates_by_block(
                first * second, block_id, blocks
            )
            first_norm = self.factor_solver._sum_coordinates_by_block(
                first.square(), block_id, blocks
            ).clamp_min(self.eps**2).sqrt()
            second_norm = self.factor_solver._sum_coordinates_by_block(
                second.square(), block_id, blocks
            ).clamp_min(self.eps**2).sqrt()
            return (
                numerator / (first_norm * second_norm).clamp_min(self.eps)
            ).clamp(-1.0, 1.0)

        value_mean = mean_factor_value[:, None].expand(-1, blocks)
        value_variance = (
            (factor_values - mean_factor_value[:, None]).square() * mask_float
        ).sum(dim=-1) / active_factor_count
        value_cv = value_variance.sqrt() / mean_factor_value.abs().clamp_min(
            self.eps
        )
        norm_variance = (
            (edge_norm - mean_edge_norm[..., None]).square()
            * edge_mask.to(theta.dtype)
        ).sum(dim=-1) / active_count
        norm_cv = norm_variance.sqrt() / mean_edge_norm.clamp_min(self.eps)
        entropy = -(
            edge_weights.clamp_min(self.eps).log()
            * edge_weights
        ).sum(dim=-1) / active_count.log().clamp_min(1.0)
        block_features = torch.stack(
            (
                torch.tanh(value_mean),
                torch.tanh(value_cv)[:, None].expand(-1, blocks),
                torch.tanh(norm_cv),
                torch.tanh(
                    torch.log(
                        (total_norm + self.eps)
                        / (mean_edge_norm + self.eps)
                    )
                ),
                torch.tanh(
                    torch.log(
                        (previous_norm + self.eps)
                        / (total_norm + self.eps)
                    )
                ),
                block_cosine(learned_factor_descent, total_descent),
                block_cosine(previous_unit, total_descent),
                entropy,
                dimension_fraction[None].expand(batch, -1),
                history_available.to(theta.dtype),
                trust[:, None].expand(-1, blocks),
            ),
            dim=-1,
        )
        controls = self.fast_aggregate_block_head(block_features)
        basis_logits = controls[..., :3]
        basis_logits[..., 2] = basis_logits[..., 2].masked_fill(
            ~history_available, -1.0e4
        )
        basis_weights = torch.softmax(basis_logits, dim=-1)
        mixed_unit = (
            basis_weights[..., 0][..., block_id] * learned_factor_descent
            + basis_weights[..., 1][..., block_id] * total_descent
            + basis_weights[..., 2][..., block_id] * previous_unit
        )
        block_gains = self.max_step_norm * torch.sigmoid(controls[..., 3])
        block_gains = block_gains * trust[:, None]
        block_gains = torch.where(
            topology_pacing_active[:, None],
            torch.zeros_like(block_gains),
            block_gains,
        )
        update = block_gains[..., block_id] * mixed_unit
        proposed_state = (
            self.factor_solver._initial_state(
                batch, factors, blocks, parameters, theta
            )
            if solver_state is None
            else solver_state
        )
        proposed_state = replace(proposed_state, previous_update=update)
        base_step = DifferentialFactorStep(
            update=update,
            state=proposed_state,
            factor_coefficients=edge_weights,
            history_coefficients=basis_weights[..., 2],
            block_gains=block_gains,
            edge_mask=edge_mask,
        )
        self.last_context_gate = (~topology_pacing_active).to(theta.dtype)
        self.last_schema_expert_blend = theta.new_zeros(())
        self.last_schema_expert_weights = None
        self.last_schema_local_admission_probability = None
        self.last_schema_local_admission = None
        return DescentAnchoredStep(
            update=update,
            parent_update=update,
            proposed_solver_state=proposed_state,
            basis_weights=basis_weights,
            block_gains=block_gains,
            base_step=base_step,
            topology_controls=theta.new_zeros(
                batch, self.TOPOLOGY_CONTROL_DIM
            ),
            block_context_controls=theta.new_zeros(
                batch, blocks, self.BLOCK_CONTEXT_CONTROL_DIM
            ),
        )
    def _topology_features(
        self,
        theta: Tensor,
        directions: Tensor,
        factor_mask: Tensor,
        block_id: Tensor,
        blocks: int,
    ) -> Tensor:
        """Return the bounded task-free schema descriptor used by context."""

        batch, parameters = theta.shape
        factors = directions.shape[1]
        dimensions = torch.bincount(block_id, minlength=blocks).to(theta)
        log_dimensions = torch.log1p(dimensions)
        active_fraction = factor_mask.to(theta.dtype).mean(dim=-1)
        constant = torch.stack(
            (
                theta.new_tensor(math.log1p(factors) / 8.0),
                theta.new_tensor(math.log1p(parameters) / 10.0),
                theta.new_tensor(math.log1p(blocks) / 5.0),
                log_dimensions.mean() / 10.0,
                log_dimensions.std(unbiased=False) / 5.0,
                log_dimensions.max() / 10.0,
            )
        )
        return torch.cat(
            (constant[None].expand(batch, -1), active_fraction[:, None]), dim=-1
        )

    @staticmethod
    def _incidence_overdispersed(
        directions: Tensor,
        factor_mask: Tensor,
        block_id: Tensor,
        *,
        eps: float = 1.0e-8,
    ) -> Tensor:
        """Return the dimensionless hub--periphery incidence predicate."""

        blocks = int(block_id.max().item()) + 1
        incidence = []
        for block_index in range(blocks):
            coordinates = block_id == block_index
            energy = directions[..., coordinates].square().sum(dim=-1)
            incidence.append((energy > eps**2) & factor_mask)
        degree = torch.stack(incidence, dim=-1).to(directions.dtype).sum(dim=1)
        return (
            degree.std(dim=-1, unbiased=False) > degree.mean(dim=-1)
        ).detach()

    def _topology_controls(
        self,
        theta: Tensor,
        directions: Tensor,
        factor_mask: Tensor,
        block_id: Tensor,
        blocks: int,
    ) -> Tensor:
        """Encode graph structure without a task, operator, or dataset label."""

        if self.topology_context_head is None:
            return theta.new_zeros(theta.shape[0], self.TOPOLOGY_CONTROL_DIM)
        features = self._topology_features(
            theta, directions, factor_mask, block_id, blocks
        )
        return self.topology_context_head(features)

    def _context_gate(
        self,
        theta: Tensor,
        directions: Tensor,
        factor_mask: Tensor,
        block_id: Tensor,
        blocks: int,
    ) -> Tensor:
        if self.context_gate_head is None:
            return theta.new_ones(theta.shape[0])
        features = self._topology_features(
            theta, directions, factor_mask, block_id, blocks
        )
        return torch.sigmoid(self.context_gate_head(features).squeeze(-1))

    def _block_context_controls(
        self,
        theta: Tensor,
        directions: Tensor,
        factor_values: Tensor,
        block_id: Tensor,
        edge_mask: Tensor,
        blocks: int,
    ) -> Tensor:
        """Produce blockwise controls from invariant local graph statistics."""

        batch, parameters = theta.shape
        if self.block_context_head is None:
            return theta.new_zeros(batch, blocks, self.BLOCK_CONTEXT_CONTROL_DIM)
        dimensions = torch.bincount(block_id, minlength=blocks).to(theta)
        log_dimensions = torch.log1p(dimensions)
        squared_norm = self.factor_solver._sum_coordinates_by_block(
            directions.square(), block_id, blocks
        ).transpose(1, 2)
        log_norm = torch.log1p(squared_norm.clamp_min(0.0).sqrt())
        active = edge_mask.to(theta.dtype)
        count = active.sum(dim=-1).clamp_min(1.0)
        mean_log_norm = (log_norm * active).sum(dim=-1) / count
        variance_log_norm = (
            (log_norm - mean_log_norm[..., None]).square() * active
        ).sum(dim=-1) / count
        expanded_values = torch.log1p(factor_values.abs())[:, None].expand(
            -1, blocks, -1
        )
        mean_log_value = (expanded_values * active).sum(dim=-1) / count
        dimension_fraction = log_dimensions / max(math.log1p(parameters), self.eps)
        features = torch.stack(
            (
                (log_dimensions / 10.0)[None].expand(batch, -1),
                dimension_fraction[None].expand(batch, -1),
                active.mean(dim=-1),
                mean_log_norm / 5.0,
                variance_log_norm.clamp_min(0.0).sqrt() / 5.0,
                mean_log_value / 5.0,
            ),
            dim=-1,
        )
        return self.block_context_head(features)

    def _gauss_newton_unit(
        self,
        directions: Tensor,
        factor_values: Tensor,
        factor_mask: Tensor,
        block_id: Tensor,
        blocks: int,
        supported_blocks: Optional[Tensor] = None,
    ) -> Tensor:
        """Build a task-agnostic damped curvature direction from scalar costs.

        For a least-squares factor ``phi=0.5*r^2``, ``grad(phi)/sqrt(2*phi)``
        is its signed residual Jacobian.  Treating every non-negative factor as
        this local surrogate gives an exact Gauss--Newton direction for scalar
        least squares and a conservative curvature anchor for aggregated costs.
        The solve is performed in the smaller of factor space and coordinate
        space.  The two damped systems are algebraically equivalent, while the
        adaptive choice avoids an ``F x F`` solve for compact camera or
        homography blocks with hundreds of observed factors.
        """

        if block_id.ndim == 2:
            batch, parameters = directions.shape[0], directions.shape[2]
            if block_id.shape != (batch, parameters):
                raise ValueError(
                    "batched curvature block identifiers must have shape [B,P]"
                )
            if supported_blocks is not None and (
                supported_blocks.ndim != 2
                or supported_blocks.shape[0] != batch
            ):
                raise ValueError(
                    "batched supported curvature blocks must have shape [B,K]"
                )
            output = torch.zeros_like(directions[:, 0])
            unique_schemas, schema_assignment = torch.unique(
                block_id, dim=0, return_inverse=True
            )
            for schema_index, schema in enumerate(unique_schemas):
                batch_indices = (schema_assignment == schema_index).nonzero(
                    as_tuple=False
                ).flatten()
                local_blocks = int(schema.max().item()) + 1
                local_supported = (
                    None
                    if supported_blocks is None
                    else supported_blocks[batch_indices[0], :local_blocks]
                )
                grouped_output = self._gauss_newton_unit(
                    directions.index_select(0, batch_indices),
                    factor_values.index_select(0, batch_indices),
                    factor_mask.index_select(0, batch_indices),
                    schema,
                    local_blocks,
                    local_supported,
                )
                output.index_copy_(0, batch_indices, grouped_output)
            return output

        residual = torch.sqrt(2.0 * factor_values.clamp_min(0.0) + self.eps**2)
        active = factor_mask.to(directions.dtype)
        jacobian = directions / residual[..., None].clamp_min(self.eps)
        jacobian = jacobian * active[..., None]
        residual = residual * active
        output = torch.zeros_like(directions[:, 0])
        for block_index in range(blocks):
            if supported_blocks is not None and not bool(supported_blocks[block_index]):
                continue
            coordinates = block_id == block_index
            local = jacobian[..., coordinates]
            factor_count = local.shape[-2]
            coordinate_count = local.shape[-1]
            # Match the original factor-space damping exactly:
            # mean(diag(J J^T)) = ||J||_F^2 / F.
            trace_scale = local.square().sum(dim=(-2, -1)) / factor_count
            damping = (
                self.curvature_damping * trace_scale.clamp_min(self.eps)
                + self.eps
            )
            if coordinate_count <= factor_count:
                gram = torch.matmul(local.transpose(-2, -1), local)
                identity = torch.eye(
                    coordinate_count,
                    device=directions.device,
                    dtype=directions.dtype,
                )
                right_hand_side = torch.matmul(
                    local.transpose(-2, -1), residual[..., None]
                )
                local_output = -torch.linalg.solve(
                    gram + damping[:, None, None] * identity[None],
                    right_hand_side,
                ).squeeze(-1)
            else:
                gram = torch.matmul(local, local.transpose(-2, -1))
                identity = torch.eye(
                    factor_count,
                    device=directions.device,
                    dtype=directions.dtype,
                )
                coefficients = torch.linalg.solve(
                    gram + damping[:, None, None] * identity[None],
                    residual[..., None],
                )
                local_output = -torch.matmul(
                    local.transpose(-2, -1), coefficients
                ).squeeze(-1)
            output[..., coordinates] = local_output
        squared_norm = self.factor_solver._sum_coordinates_by_block(
            output.square(), block_id, blocks
        )
        norm = squared_norm.clamp_min(self.eps**2).sqrt()
        return output / norm[..., block_id].clamp_min(self.eps)

    def _feedback(
        self,
        theta: Tensor,
        state: Optional[DescentAnchoredState],
    ) -> tuple[Tensor, Tensor, Tensor, Optional[DifferentialFactorState]]:
        batch = theta.shape[0]
        if state is None:
            trust = theta.new_ones(batch)
            acceptance = theta.new_ones(batch)
            reduction = theta.new_zeros(batch)
            solver_state = None
        else:
            trust = state.trust_scale.to(theta)
            acceptance = state.previous_acceptance.to(theta)
            reduction = state.previous_reduction.to(theta)
            solver_state = state.solver_state
        features = torch.stack(
            (
                torch.log(trust.clamp_min(self.eps)),
                acceptance,
                torch.tanh(reduction),
            ),
            dim=-1,
        )
        return features, trust, acceptance, solver_state

    def forward(
        self,
        theta: Tensor,
        directions: Tensor,
        factor_values: Tensor,
        block_id: Tensor,
        factor_mask: Optional[Tensor] = None,
        state: Optional[DescentAnchoredState] = None,
        curvature_block_id: Optional[Tensor] = None,
        curvature_logit_offset: Optional[Tensor] = None,
        allow_schema_topology_pacing: bool = True,
        allow_fast_aggregate: bool = False,
    ) -> DescentAnchoredStep:
        batch, parameters = theta.shape
        factors = directions.shape[1]
        if factor_mask is None:
            factor_mask = torch.ones(
                batch, factors, device=theta.device, dtype=torch.bool
            )
        else:
            factor_mask = factor_mask.to(device=theta.device, dtype=torch.bool)
        feedback, trust, _, solver_state = self._feedback(theta, state)
        block_id = block_id.to(device=theta.device, dtype=torch.long)
        blocks = int(block_id.max().item()) + 1
        topology_pacing_active = (
            self._incidence_overdispersed(
                directions, factor_mask, block_id, eps=self.eps
            )
            if self.use_schema_topology_pacing and allow_schema_topology_pacing
            else torch.zeros(batch, device=theta.device, dtype=torch.bool)
        )
        self.last_schema_topology_pacing_active = topology_pacing_active
        if curvature_block_id is None:
            curvature_block_id = block_id
        else:
            curvature_block_id = curvature_block_id.to(
                device=theta.device, dtype=torch.long
            )
            if curvature_block_id.shape not in (
                block_id.shape,
                (batch, parameters),
            ):
                raise ValueError(
                    "curvature_block_id must have shape [P] or [B,P]"
                )
            if torch.any(curvature_block_id < 0):
                raise ValueError("curvature block identifiers must be non-negative")
            rows = (
                curvature_block_id[None]
                if curvature_block_id.ndim == 1
                else curvature_block_id
            )
            for row in rows:
                curvature_labels = torch.unique(row, sorted=True)
                if not torch.equal(
                    curvature_labels,
                    torch.arange(
                        curvature_labels.numel(),
                        device=theta.device,
                        dtype=torch.long,
                    ),
                ):
                    raise ValueError(
                        "curvature block identifiers must be contiguous per batch item"
                    )
        if self.fast_aggregate_edge_head is not None and allow_fast_aggregate:
            return self._fast_aggregate_step(
                theta,
                directions,
                factor_values,
                factor_mask,
                block_id,
                blocks,
                trust,
                solver_state,
                topology_pacing_active,
            )
        topology_controls = self._topology_controls(
            theta, directions, factor_mask, block_id, blocks
        )
        context_gate = self._context_gate(
            theta, directions, factor_mask, block_id, blocks
        )
        # A zero gate restores the inherited solver computation exactly: the
        # topology and block controls vanish, the recurrent factor solver sees
        # the original feedback, and hard/soft schema-expert admission is zero.
        context_gate = torch.where(
            topology_pacing_active,
            torch.zeros_like(context_gate),
            context_gate,
        )
        self.last_context_gate = context_gate
        topology_controls = topology_controls * context_gate[:, None]
        conditioned_feedback = feedback + 0.25 * torch.tanh(
            topology_controls[:, : self.FEEDBACK_DIM]
        )
        base_step = self.factor_solver(
            theta,
            directions,
            factor_values,
            block_id,
            factor_mask=factor_mask,
            global_features=conditioned_feedback,
            state=solver_state,
        )

        base_squared_norm = self.factor_solver._sum_coordinates_by_block(
            base_step.update.square(), block_id, blocks
        )
        base_norm = base_squared_norm.clamp_min(self.eps**2).sqrt()
        base_unit = base_step.update / base_norm[..., block_id].clamp_min(self.eps)

        total_gradient = (
            directions * factor_mask[..., None].to(directions.dtype)
        ).sum(dim=1)
        total_squared_norm = self.factor_solver._sum_coordinates_by_block(
            total_gradient.square(), block_id, blocks
        )
        total_norm = total_squared_norm.clamp_min(self.eps**2).sqrt()
        total_descent_unit = -total_gradient / total_norm[..., block_id].clamp_min(
            self.eps
        )

        active_blocks = base_step.edge_mask.any(dim=-1)
        block_context_controls = self._block_context_controls(
            theta,
            directions,
            factor_values,
            block_id,
            base_step.edge_mask,
            blocks,
        )
        block_context_controls = (
            block_context_controls * context_gate[:, None, None]
        )
        block_context = torch.cat(
            (
                base_step.state.block_hidden,
                base_step.state.global_hidden[:, None].expand(-1, blocks, -1),
                conditioned_feedback[:, None].expand(-1, blocks, -1),
            ),
            dim=-1,
        )
        basis_logits = self.basis_head(block_context) + 0.5 * torch.tanh(
            block_context_controls[..., :2]
        )
        if self.curvature_head is None:
            basis_weights = torch.softmax(basis_logits, dim=-1)
            curvature_unit = None
        else:
            curvature_logits = self.curvature_head(block_context) + 0.5 * torch.tanh(
                block_context_controls[..., 2:3]
            )
            if curvature_logit_offset is not None:
                curvature_logit_offset = torch.as_tensor(
                    curvature_logit_offset,
                    device=theta.device,
                    dtype=theta.dtype,
                )
                if curvature_logit_offset.ndim == 0:
                    curvature_logit_offset = curvature_logit_offset.expand(batch)
                if curvature_logit_offset.shape != (batch,):
                    raise ValueError(
                        "curvature_logit_offset must be scalar or have shape [B]"
                    )
                if not torch.isfinite(curvature_logit_offset).all():
                    raise ValueError("curvature_logit_offset must be finite")
                curvature_logits = curvature_logits + curvature_logit_offset[
                    :, None, None
                ]
            curvature_blocks = int(curvature_block_id.max().item()) + 1
            if curvature_block_id.ndim == 1:
                block_dimensions = torch.bincount(
                    curvature_block_id, minlength=curvature_blocks
                )
            else:
                block_dimensions = torch.stack(
                    [
                        torch.bincount(row, minlength=curvature_blocks)
                        for row in curvature_block_id
                    ]
                )
            if self.curvature_max_block_dim is None:
                curvature_supported = torch.ones_like(
                    block_dimensions, dtype=torch.bool
                )
            else:
                curvature_supported = (
                    block_dimensions <= self.curvature_max_block_dim
                )
            if curvature_block_id.ndim == 1:
                base_curvature_supported = torch.ones(
                    blocks, device=theta.device, dtype=torch.bool
                )
                for block_index in range(blocks):
                    coordinates = block_id == block_index
                    base_curvature_supported[block_index] = bool(
                        curvature_supported[
                            curvature_block_id[coordinates]
                        ].all()
                    )
                curvature_mask = ~base_curvature_supported[None, :, None]
            else:
                base_curvature_supported = torch.ones(
                    batch, blocks, device=theta.device, dtype=torch.bool
                )
                for block_index in range(blocks):
                    metric_labels = curvature_block_id[:, block_id == block_index]
                    supported = torch.gather(
                        curvature_supported, 1, metric_labels
                    )
                    base_curvature_supported[:, block_index] = supported.all(dim=1)
                curvature_mask = ~base_curvature_supported[..., None]
            curvature_logits = curvature_logits.masked_fill(
                curvature_mask, -1.0e4
            )
            basis_weights = torch.softmax(
                torch.cat((basis_logits, curvature_logits), dim=-1), dim=-1
            )
            curvature_unit = self._gauss_newton_unit(
                directions,
                factor_values,
                factor_mask,
                curvature_block_id,
                curvature_blocks,
                curvature_supported,
            )
        basis_weights = basis_weights * active_blocks[..., None].to(theta.dtype)
        topology_gain = torch.exp(
            0.5 * torch.tanh(topology_controls[:, self.FEEDBACK_DIM])
        )
        block_context_gain = torch.exp(
            0.5 * torch.tanh(block_context_controls[..., 3])
        )
        block_gains = (
            base_step.block_gains
            * trust[:, None]
            * topology_gain[:, None]
            * block_context_gain
        )
        block_gains = block_gains * active_blocks.to(theta.dtype)
        base_weight = basis_weights[..., 0][..., block_id]
        gradient_weight = basis_weights[..., 1][..., block_id]
        gain_by_coordinate = block_gains[..., block_id]
        mixed_unit = (
            base_weight * base_unit + gradient_weight * total_descent_unit
        )
        if curvature_unit is not None:
            curvature_weight = basis_weights[..., 2][..., block_id]
            mixed_unit = mixed_unit + curvature_weight * curvature_unit
        update = gain_by_coordinate * mixed_unit
        # Preserve the inherited equivariant proposal before the optional
        # schema expert is blended in.  The five-step wrapper can compare the
        # two proposals using support factors without another network call.
        parent_update = update
        if self.schema_expert_head is not None:
            previous_update = (
                torch.zeros_like(update)
                if solver_state is None
                else solver_state.previous_update.to(update)
            )
            previous_squared_norm = self.factor_solver._sum_coordinates_by_block(
                previous_update.square(), block_id, blocks
            )
            previous_norm = previous_squared_norm.clamp_min(self.eps**2).sqrt()
            previous_unit = previous_update / previous_norm[
                ..., block_id
            ].clamp_min(self.eps)

            def block_cosine(first: Tensor, second: Tensor) -> Tensor:
                numerator = self.factor_solver._sum_coordinates_by_block(
                    first * second, block_id, blocks
                )
                first_norm = self.factor_solver._sum_coordinates_by_block(
                    first.square(), block_id, blocks
                ).clamp_min(self.eps**2).sqrt()
                second_norm = self.factor_solver._sum_coordinates_by_block(
                    second.square(), block_id, blocks
                ).clamp_min(self.eps**2).sqrt()
                return (
                    numerator / (first_norm * second_norm).clamp_min(self.eps)
                ).clamp(-1.0, 1.0)

            dimensions = torch.bincount(block_id, minlength=blocks).to(theta)
            dimension_fraction = torch.log1p(dimensions) / max(
                math.log1p(parameters), self.eps
            )
            if self.schema_expert_feature_version == "bounded_ratio_v2":
                def bounded_norm_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
                    return torch.tanh(
                        torch.log(
                            (numerator + self.eps) / (denominator + self.eps)
                        )
                    )

                norm_features = (
                    bounded_norm_ratio(total_norm, base_norm)[..., None],
                    bounded_norm_ratio(previous_norm, base_norm)[..., None],
                    bounded_norm_ratio(total_norm, previous_norm)[..., None],
                )
                expert_feedback = torch.tanh(feedback)
                expert_topology = torch.tanh(topology_controls)
            else:
                norm_features = (
                    torch.log1p(total_norm)[..., None],
                    torch.log1p(base_norm)[..., None],
                    torch.log1p(previous_norm)[..., None],
                )
                expert_feedback = feedback
                expert_topology = topology_controls
            expert_features = torch.cat(
                (
                    *norm_features,
                    block_cosine(total_descent_unit, base_unit)[..., None],
                    block_cosine(total_descent_unit, previous_unit)[..., None],
                    block_cosine(base_unit, previous_unit)[..., None],
                    expert_feedback[:, None].expand(-1, blocks, -1),
                    expert_topology[:, None].expand(-1, blocks, -1),
                    dimension_fraction[None, :, None].expand(batch, -1, -1),
                ),
                dim=-1,
            )
            if expert_features.shape[-1] != self.SCHEMA_EXPERT_FEATURE_DIM:
                raise RuntimeError("schema expert feature dimension changed")
            expert_controls = self.schema_expert_head(expert_features)
            direction_count = (
                4
                if self.schema_expert_direction_version == "current_direction_v2"
                else 3
            )
            expert_weights = torch.softmax(
                expert_controls[..., :direction_count], dim=-1
            )
            expert_gain = block_gains * torch.exp(
                torch.tanh(expert_controls[..., direction_count])
            )
            if self.schema_expert_direction_version == "current_direction_v2":
                expert_unit = (
                    expert_weights[..., 0][..., block_id] * mixed_unit
                    + expert_weights[..., 1][..., block_id] * base_unit
                    + expert_weights[..., 2][..., block_id] * total_descent_unit
                    + expert_weights[..., 3][..., block_id] * previous_unit
                )
            else:
                expert_unit = (
                    expert_weights[..., 0][..., block_id] * base_unit
                    + expert_weights[..., 1][..., block_id] * total_descent_unit
                    + expert_weights[..., 2][..., block_id] * previous_unit
                )
            expert_update = expert_gain[..., block_id] * expert_unit
            blend = torch.tanh(self.schema_expert_scale)
            expert_admission = (
                (context_gate >= 0.5).to(context_gate.dtype)
                if self.schema_expert_hard_admission
                else context_gate
            )
            if self.schema_local_admission_head is not None:
                if (
                    self.schema_local_admission_feature_version
                    == "expert_plus_incidence_v2"
                ):
                    incidence = []
                    for block_index in range(blocks):
                        coordinates = block_id == block_index
                        block_energy = directions[..., coordinates].square().sum(dim=-1)
                        incidence.append(
                            (block_energy > self.eps**2) & factor_mask
                        )
                    incidence_mask = torch.stack(incidence, dim=-1)
                    degree = incidence_mask.to(theta.dtype).sum(dim=1)
                    active_factor_count = factor_mask.to(theta.dtype).sum(dim=1)
                    degree_fraction = degree / active_factor_count[:, None].clamp_min(1.0)
                    mean_degree = degree.mean(dim=-1, keepdim=True)
                    relative_degree = torch.tanh(
                        torch.log(
                            (degree + 1.0) / (mean_degree + 1.0)
                        )
                    )
                    incidence_share = degree / degree.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1.0)
                    degree_per_dimension = torch.tanh(
                        torch.log(
                            (degree + 1.0)
                            / (dimensions[None] + 1.0)
                        )
                    )
                    local_features = torch.cat(
                        (
                            expert_features,
                            degree_fraction[..., None],
                            relative_degree[..., None],
                            incidence_share[..., None],
                            degree_per_dimension[..., None],
                        ),
                        dim=-1,
                    )
                else:
                    local_features = expert_features
                if (
                    local_features.shape[-1]
                    != self.schema_local_admission_feature_dim
                ):
                    raise RuntimeError("local schema admission feature dimension changed")
                local_probability = torch.sigmoid(
                    self.schema_local_admission_head(local_features).squeeze(-1)
                )
                local_hard = (local_probability >= 0.5).to(
                    local_probability.dtype
                )
                # Training uses a straight-through derivative while preserving
                # a genuinely discrete forward decision. Evaluation takes the
                # hard branch directly so an all-open gate is functionally
                # identical to the inherited checkpoint.
                local_admission = (
                    local_probability
                    + (local_hard - local_probability).detach()
                    if self.training
                    else local_hard
                )
                admission_by_coordinate = (
                    expert_admission[:, None]
                    * local_admission[..., block_id]
                )
                self.last_schema_local_admission_probability = local_probability
                self.last_schema_local_admission = local_hard
            else:
                admission_by_coordinate = expert_admission[:, None]
                self.last_schema_local_admission_probability = None
                self.last_schema_local_admission = None
            update = update + admission_by_coordinate * blend * (
                expert_update - update
            )
            self.last_schema_expert_blend = blend
            self.last_schema_expert_weights = expert_weights
        else:
            self.last_schema_expert_blend = theta.new_zeros(())
            self.last_schema_expert_weights = None
            self.last_schema_local_admission_probability = None
            self.last_schema_local_admission = None
        if update.shape != (batch, parameters):
            raise RuntimeError("anchored update reconstruction produced a bad shape")
        return DescentAnchoredStep(
            update=update,
            parent_update=parent_update,
            proposed_solver_state=base_step.state,
            basis_weights=basis_weights,
            block_gains=block_gains,
            base_step=base_step,
            topology_controls=topology_controls,
            block_context_controls=block_context_controls,
        )


class FiveStepDescentAnchoredOptimizer(nn.Module):
    """Five hard-gated DA-DFS calls with explicit accept/reduction feedback."""

    def __init__(
        self,
        solver: Optional[DescentAnchoredDifferentialFactorSolver] = None,
        steps: int = 5,
        rejection_shrink: float = 0.5,
        acceptance_growth: float = 1.25,
        acceptance_tolerance: float = 1.0e-8,
        line_search_scales: tuple[float, ...] = (1.0,),
        short_circuit_line_search: bool = False,
        generalization_trust_head: Optional[GeneralizationTrustHead] = None,
        transfer_threshold: float = 0.5,
        spatial_field_head: Optional[SpatialFieldUpdateHead] = None,
        spatial_transfer_threshold: Optional[float] = None,
        allow_spatial_support_override: bool = False,
        spatial_max_support_increase: float = 0.25,
        spatial_expanded_support_admission_head: Optional[
            SpatialExpandedSupportAdmissionHead
        ] = None,
        spatial_legacy_max_support_increase: float = 0.0,
        spatial_expanded_support_threshold: float = 0.5,
        spatial_terminal_rollback_head: Optional[
            SpatialTerminalRollbackHead
        ] = None,
        spatial_terminal_keep_threshold: float = 0.5,
        spatial_terminal_joint_latent_only: bool = False,
        spatial_joint_latent_pacing_head: Optional[
            SpatialJointLatentPacingHead
        ] = None,
        inference_vjp_chunk_size: Optional[int] = 8,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        if inference_vjp_chunk_size is not None and inference_vjp_chunk_size < 1:
            raise ValueError("inference_vjp_chunk_size must be positive when provided")
        if not 0.0 < rejection_shrink < 1.0:
            raise ValueError("rejection_shrink must lie in (0, 1)")
        if acceptance_growth < 1.0:
            raise ValueError("acceptance_growth must be at least one")
        if not 0.0 < transfer_threshold < 1.0:
            raise ValueError("transfer_threshold must lie in (0,1)")
        if spatial_transfer_threshold is not None and not (
            0.0 < spatial_transfer_threshold < 1.0
        ):
            raise ValueError("spatial_transfer_threshold must lie in (0,1)")
        if spatial_max_support_increase < 0.0:
            raise ValueError("spatial_max_support_increase must be non-negative")
        if not 0.0 <= spatial_legacy_max_support_increase <= spatial_max_support_increase:
            raise ValueError(
                "legacy spatial support cap must lie within the maximum support cap"
            )
        if not 0.0 < spatial_expanded_support_threshold < 1.0:
            raise ValueError("expanded support threshold must lie in (0,1)")
        if not 0.0 < spatial_terminal_keep_threshold < 1.0:
            raise ValueError("terminal keep threshold must lie in (0,1)")
        if (
            not line_search_scales
            or line_search_scales[0] != 1.0
            or any(scale <= 0.0 or scale > 1.0 for scale in line_search_scales)
            or any(
                first <= second
                for first, second in zip(
                    line_search_scales, line_search_scales[1:]
                )
            )
        ):
            raise ValueError(
                "line_search_scales must start at 1 and be strictly decreasing in (0,1]"
            )
        self.solver = solver or DescentAnchoredDifferentialFactorSolver()
        self.steps = steps
        self.rejection_shrink = rejection_shrink
        self.acceptance_growth = acceptance_growth
        self.acceptance_tolerance = acceptance_tolerance
        self.line_search_scales = tuple(float(scale) for scale in line_search_scales)
        # Later line-search scales cannot be selected after every batch element
        # has accepted a larger scale. Keep the optimization opt-in so frozen
        # historical protocols retain their original call accounting.
        self.short_circuit_line_search = bool(short_circuit_line_search)
        self.generalization_trust_head = generalization_trust_head
        self.transfer_threshold = float(transfer_threshold)
        self.spatial_field_head = spatial_field_head
        self.spatial_transfer_threshold = (
            None
            if spatial_transfer_threshold is None
            else float(spatial_transfer_threshold)
        )
        self.allow_spatial_support_override = bool(allow_spatial_support_override)
        self.spatial_max_support_increase = float(spatial_max_support_increase)
        self.spatial_expanded_support_admission_head = (
            spatial_expanded_support_admission_head
        )
        self.spatial_legacy_max_support_increase = float(
            spatial_legacy_max_support_increase
        )
        self.spatial_expanded_support_threshold = float(
            spatial_expanded_support_threshold
        )
        self.spatial_terminal_rollback_head = spatial_terminal_rollback_head
        self.spatial_terminal_keep_threshold = float(
            spatial_terminal_keep_threshold
        )
        self.spatial_terminal_joint_latent_only = bool(
            spatial_terminal_joint_latent_only
        )
        self.spatial_joint_latent_pacing_head = spatial_joint_latent_pacing_head
        self.inference_vjp_chunk_size = inference_vjp_chunk_size

    @staticmethod
    def _masked_stats(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        weights = mask.to(values.dtype)
        count = weights.sum(dim=-1).clamp_min(1.0)
        mean = (values * weights).sum(dim=-1) / count
        variance = (
            (values - mean[:, None]).square() * weights
        ).sum(dim=-1) / count
        minimum = values.masked_fill(~mask, torch.inf).amin(dim=-1)
        maximum = values.masked_fill(~mask, -torch.inf).amax(dim=-1)
        return mean, variance.clamp_min(0.0).sqrt(), minimum, maximum

    @staticmethod
    def _query_response_features(
        current_values: Tensor, candidate_values: Tensor
    ) -> Tensor:
        if (
            current_values.ndim != 2
            or candidate_values.shape != current_values.shape
            or current_values.shape[1] < 1
        ):
            raise ValueError("query factors must share a non-empty [B,Q] shape")
        if (
            not torch.isfinite(current_values).all()
            or not torch.isfinite(candidate_values).all()
        ):
            raise ValueError("query factors must be finite")
        relative = torch.tanh(
            (candidate_values - current_values)
            / current_values.abs().clamp_min(1.0e-6)
        )
        relative_mean = relative.mean(dim=-1)
        relative_std = relative.std(dim=-1, unbiased=False)
        concentration = candidate_values.abs().max(dim=-1).values / (
            candidate_values.abs().sum(dim=-1).clamp_min(1.0e-8)
        )
        return torch.stack(
            (
                torch.log1p(current_values.abs().mean(dim=-1)),
                torch.log1p(candidate_values.abs().mean(dim=-1)),
                relative_mean,
                relative_std,
                relative.min(dim=-1).values,
                relative.max(dim=-1).values,
                (candidate_values < current_values).to(current_values.dtype).mean(
                    dim=-1
                ),
                concentration,
            ),
            dim=-1,
        ).detach()

    def _trust_features(
        self,
        *,
        theta: Tensor,
        current_values: Tensor,
        candidate_values: Tensor,
        factor_mask: Tensor,
        proposal: DescentAnchoredStep,
        step_scale: Tensor,
        state: Optional[DescentAnchoredState],
        block_id: Tensor,
    ) -> Tensor:
        dtype = theta.dtype
        weights = factor_mask.to(dtype)
        current_score = (current_values * weights).sum(dim=-1)
        candidate_score = (candidate_values * weights).sum(dim=-1)
        reduction = (current_score - candidate_score) / current_score.abs().clamp_min(
            1.0e-8
        )
        per_factor = (current_values - candidate_values) / current_values.abs().clamp_min(
            1.0e-6
        )
        delta_mean, delta_std, delta_min, delta_max = self._masked_stats(
            per_factor, factor_mask
        )
        count = weights.sum(dim=-1).clamp_min(1.0)
        current_mean = current_score / count
        candidate_mean = candidate_score / count
        current_cv = (
            (((current_values - current_mean[:, None]).square() * weights).sum(dim=-1) / count)
            .clamp_min(0.0)
            .sqrt()
            / current_mean.abs().clamp_min(1.0e-6)
        )
        candidate_cv = (
            (((candidate_values - candidate_mean[:, None]).square() * weights).sum(dim=-1) / count)
            .clamp_min(0.0)
            .sqrt()
            / candidate_mean.abs().clamp_min(1.0e-6)
        )
        current_share = (
            current_values * weights / current_score.abs().clamp_min(1.0e-6)[:, None]
        ).amax(dim=-1)
        candidate_share = (
            candidate_values * weights / candidate_score.abs().clamp_min(1.0e-6)[:, None]
        ).amax(dim=-1)
        update_rms = proposal.update.square().mean(dim=-1).clamp_min(0.0).sqrt()
        if state is None:
            previous_trust = theta.new_ones(theta.shape[0])
            previous_acceptance = theta.new_ones(theta.shape[0])
            previous_reduction = theta.new_zeros(theta.shape[0])
        else:
            previous_trust = state.trust_scale.to(theta)
            previous_acceptance = state.previous_acceptance.to(theta)
            previous_reduction = torch.tanh(state.previous_reduction.to(theta))
        active_blocks = proposal.base_step.edge_mask.any(dim=-1)
        block_weights = active_blocks.to(dtype)
        active_count = block_weights.sum(dim=-1).clamp_min(1.0)
        mean_gain = (proposal.block_gains * block_weights).sum(dim=-1) / active_count
        max_gain = proposal.block_gains.masked_fill(~active_blocks, 0.0).amax(dim=-1)
        basis = proposal.basis_weights
        if basis.shape[-1] < 3:
            basis = torch.nn.functional.pad(basis, (0, 3 - basis.shape[-1]))
        mean_basis = (basis[..., :3] * block_weights[..., None]).sum(dim=1) / active_count[:, None]
        blocks = int(block_id.max().item()) + 1
        structural = torch.stack(
            (
                theta.new_full((theta.shape[0],), math.log1p(current_values.shape[1]) / 8.0),
                theta.new_full((theta.shape[0],), math.log1p(theta.shape[1]) / 10.0),
                theta.new_full((theta.shape[0],), math.log1p(blocks) / 5.0),
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                torch.log1p(current_score.abs())[:, None],
                torch.log1p(candidate_score.abs())[:, None],
                reduction[:, None],
                delta_mean[:, None],
                delta_std[:, None],
                delta_min[:, None],
                delta_max[:, None],
                current_cv[:, None],
                candidate_cv[:, None],
                (candidate_share - current_share)[:, None],
                update_rms[:, None],
                step_scale[:, None],
                previous_trust[:, None],
                previous_acceptance[:, None],
                previous_reduction[:, None],
                mean_gain[:, None],
                max_gain[:, None],
                mean_basis,
                structural,
            ),
            dim=-1,
        )
        if features.shape[-1] != GeneralizationTrustHead.FEATURE_DIM:
            raise RuntimeError("generalization trust feature dimension changed")
        return features.detach()

    def forward(
        self,
        initial_theta: Tensor,
        factor_fn: FactorFunction,
        block_id: Tensor,
        factor_mask: Optional[Tensor] = None,
        *,
        create_graph: bool = False,
        curvature_block_id: Optional[Tensor] = None,
        curvature_logit_offset: Optional[Tensor] = None,
        spatial_context: Optional[SpatialFieldContext] = None,
        allow_fast_aggregate_expert: bool = False,
        query_factor_fn: Optional[FactorFunction] = None,
        typed_observable_fn: Optional[FactorFunction] = None,
        coordinate_metric_fn: Optional[CoordinateMetricFunction] = None,
        coordinate_scale: Optional[Tensor] = None,
        factor_scale: Optional[Tensor] = None,
        force_spatial_accept: bool = False,
        force_spatial_expert: Optional[int] = None,
        force_spatial_routing: Optional[Tensor] = None,
        ablate_solver_recurrence: bool = False,
    ) -> DescentAnchoredOptimizationResult:
        if initial_theta.ndim != 2:
            raise ValueError("initial_theta must have shape [B, P]")
        theta = initial_theta
        if not theta.requires_grad:
            theta = theta.detach().clone().requires_grad_(True)
        batch = theta.shape[0]
        if spatial_context is not None:
            field_parameters = math.prod(spatial_context.field_shape)
            if spatial_context.observation.shape != (batch, field_parameters):
                raise ValueError("spatial observation must match the declared field")
            if spatial_context.parameter_indices is None:
                if field_parameters != initial_theta.shape[1]:
                    raise ValueError(
                        "a partial spatial field requires parameter_indices"
                    )
            else:
                indices = spatial_context.parameter_indices
                if (
                    indices.ndim != 1
                    or indices.numel() != field_parameters
                    or indices.dtype != torch.long
                    or torch.any(indices < 0)
                    or torch.any(indices >= initial_theta.shape[1])
                    or torch.unique(indices).numel() != indices.numel()
                ):
                    raise ValueError(
                        "spatial parameter_indices must be unique valid long indices"
                    )
        if force_spatial_accept and (
            spatial_context is None or self.spatial_field_head is None
        ):
            raise ValueError("force_spatial_accept requires an active spatial head")
        if force_spatial_expert is not None and not isinstance(
            self.spatial_field_head, SpatialFieldMixtureUpdateHead
        ):
            raise ValueError("force_spatial_expert requires a mixture spatial head")
        if force_spatial_routing is not None and not isinstance(
            self.spatial_field_head, SpatialFieldMixtureUpdateHead
        ):
            raise ValueError("force_spatial_routing requires a mixture spatial head")
        if force_spatial_expert is not None and force_spatial_routing is not None:
            raise ValueError(
                "force_spatial_expert and force_spatial_routing are mutually exclusive"
            )
        if (typed_observable_fn is None) != (coordinate_metric_fn is None):
            raise ValueError(
                "typed observable and coordinate metric functions must be provided together"
            )
        if typed_observable_fn is not None and query_factor_fn is None:
            raise ValueError("typed pacing requires a query factor function")
        if coordinate_scale is None:
            canonical_scale = None
        else:
            canonical_scale = torch.as_tensor(
                coordinate_scale, device=theta.device, dtype=theta.dtype
            )
            if canonical_scale.ndim == 1:
                canonical_scale = canonical_scale[None].expand(batch, -1)
            if canonical_scale.shape != theta.shape:
                raise ValueError(
                    "coordinate_scale must have shape [P] or [B,P]"
                )
            if (
                not torch.isfinite(canonical_scale).all()
                or torch.any(canonical_scale <= 0.0)
            ):
                raise ValueError("coordinate_scale must be finite and positive")
            if spatial_context is not None:
                raise ValueError(
                    "coordinate_scale is currently supported only for compact graphs"
                )
        if factor_scale is None:
            canonical_factor_scale = None
        else:
            canonical_factor_scale = torch.as_tensor(
                factor_scale, device=theta.device, dtype=theta.dtype
            )
            if canonical_factor_scale.ndim == 0:
                canonical_factor_scale = canonical_factor_scale.expand(batch)
            if canonical_factor_scale.shape != (batch,):
                raise ValueError("factor_scale must be scalar or have shape [B]")
            if (
                not torch.isfinite(canonical_factor_scale).all()
                or torch.any(canonical_factor_scale <= 0.0)
            ):
                raise ValueError("factor_scale must be finite and positive")
            if spatial_context is not None:
                raise ValueError(
                    "factor_scale is currently supported only for compact graphs"
                )
        state: Optional[DescentAnchoredState] = None
        theta_trace = [theta.detach()]
        value_trace: list[Tensor] = []
        proposal_values: list[Tensor] = []
        proposal_thetas: list[Tensor] = []
        acceptances: list[Tensor] = []
        selected_scales: list[Tensor] = []
        updates: list[Tensor] = []
        trust_trace = [theta.new_ones(batch)]
        basis_trace: list[Tensor] = []
        topology_trace: list[Tensor] = []
        block_context_trace: list[Tensor] = []
        support_candidate_thetas: list[Tensor] = []
        support_candidate_values: list[Tensor] = []
        support_acceptances: list[Tensor] = []
        transfer_logits: list[Tensor] = []
        transfer_probabilities: list[Tensor] = []
        transfer_delta_predictions: list[Tensor] = []
        expanded_support_admission_logits: list[Tensor] = []
        expanded_support_admission_probabilities: list[Tensor] = []
        expanded_support_overrides: list[Tensor] = []
        terminal_keep_logits: list[Tensor] = []
        terminal_keep_probabilities: list[Tensor] = []
        joint_latent_pacing_logits: list[Tensor] = []
        joint_latent_pacing_scales: list[Tensor] = []
        joint_latent_pacing_gate_logits: list[Tensor] = []
        joint_latent_pacing_gate_probabilities: list[Tensor] = []
        joint_latent_pacing_gate_active: list[Tensor] = []
        joint_latent_continue_logits: list[Tensor] = []
        joint_latent_continue_probabilities: list[Tensor] = []
        joint_latent_continue_active: list[Tensor] = []
        joint_latent_unpaced_updates: list[Tensor] = []
        joint_latent_pacing_features: list[Tensor] = []
        joint_latent_typed_features: list[Tensor] = []
        spatial_updates: list[Tensor] = []
        spatial_trust_offsets: list[Tensor] = []
        support_overrides: list[Tensor] = []
        schema_expert_admissions: list[Tensor] = []
        schema_topology_pacing_steps: list[Tensor] = []
        locked_expanded_support_admission: Optional[Tensor] = None
        terminal_initial_score: Optional[Tensor] = None
        terminal_first_proposal_ratio: Optional[Tensor] = None
        terminal_max_proposal_ratio: Optional[Tensor] = None
        terminal_min_selected_ratio: Optional[Tensor] = None
        terminal_initial_transfer_probability: Optional[Tensor] = None
        terminal_acceptance_count = theta.new_zeros(batch)
        terminal_support_acceptance_count = theta.new_zeros(batch)
        terminal_override_count = theta.new_zeros(batch)
        terminal_expanded_override_count = theta.new_zeros(batch)
        terminal_cumulative_scale = theta.new_zeros(batch)
        terminal_cumulative_update_energy = theta.new_zeros(batch)
        pacing_initial_query_score: Optional[Tensor] = None
        pacing_previous_query_score: Optional[Tensor] = None
        pacing_previous_latent_gradient: Optional[Tensor] = None
        pacing_previous_total_gradient: Optional[Tensor] = None
        locked_joint_latent_gate_logit: Optional[Tensor] = None
        locked_joint_latent_gate_probability: Optional[Tensor] = None
        locked_joint_latent_gate_active: Optional[Tensor] = None
        joint_latent_continue_alive: Optional[Tensor] = None

        if self.spatial_field_head is not None and hasattr(
            self.spatial_field_head, "begin_rollout"
        ):
            self.spatial_field_head.begin_rollout()

        inference_only = not create_graph and not self.training
        for step_index in range(self.steps):
            current_values, directions = factor_value_and_gradients(
                theta,
                factor_fn,
                create_graph=create_graph,
                vjp_chunk_size=(
                    self.inference_vjp_chunk_size if inference_only else None
                ),
            )
            canonical_current_values = (
                current_values
                if canonical_factor_scale is None
                else current_values / canonical_factor_scale[:, None]
            )
            canonical_directions = (
                directions
                if canonical_factor_scale is None
                else directions / canonical_factor_scale[:, None, None]
            )

            def canonicalize_factor_values(values: Tensor) -> Tensor:
                return (
                    values
                    if canonical_factor_scale is None
                    else values / canonical_factor_scale[:, None]
                )
            factors = current_values.shape[1]
            if factor_mask is None:
                factor_mask = torch.ones(
                    batch, factors, device=theta.device, dtype=torch.bool
                )
            if factor_mask.shape != (batch, factors):
                raise ValueError("factor_mask must have shape [B, F]")
            mask_float = factor_mask.to(current_values.dtype)
            if step_index == 0:
                value_trace.append(current_values.detach())

            solver_theta = (
                theta if canonical_scale is None else theta / canonical_scale
            )
            solver_directions = (
                canonical_directions
                if canonical_scale is None
                else canonical_directions * canonical_scale[:, None, :]
            )
            # In deployment we still differentiate factors with respect to the
            # state, but never need a backward graph through the frozen Solver.
            # Keeping that graph used to retain the [B,K,F,H] attention stack
            # once per recurrent call and made peak memory grow linearly with
            # the requested horizon. Training behavior is unchanged.
            with torch.set_grad_enabled(not inference_only):
                canonical_proposal = self.solver(
                    solver_theta,
                    solver_directions if create_graph else solver_directions.detach(),
                    canonical_current_values
                    if create_graph
                    else canonical_current_values.detach(),
                    block_id,
                    factor_mask=factor_mask,
                    state=None if ablate_solver_recurrence else state,
                    curvature_block_id=curvature_block_id,
                    curvature_logit_offset=curvature_logit_offset,
                    allow_schema_topology_pacing=spatial_context is None,
                    allow_fast_aggregate=(
                        allow_fast_aggregate_expert and spatial_context is None
                    ),
                )
            # The recurrent Solver always operates in the adapter-declared
            # dimensionless chart.  Only the tangent update is mapped back to
            # the caller's coordinate units.  Keeping a separate canonical
            # proposal also prevents the learned trust head from seeing units.
            trust_proposal = canonical_proposal
            proposal = (
                canonical_proposal
                if canonical_scale is None
                else replace(
                    canonical_proposal,
                    update=canonical_proposal.update * canonical_scale,
                    parent_update=(
                        canonical_proposal.parent_update * canonical_scale
                    ),
                )
            )
            spatial_probe_features: Optional[Tensor] = None
            if self.spatial_field_head is None or spatial_context is None:
                spatial_update = torch.zeros_like(theta)
                spatial_trust_offset = theta.new_zeros(batch)
                spatial_policy_active = torch.zeros(
                    batch, device=theta.device, dtype=torch.bool
                )
                if spatial_context is not None:
                    field_gradient = torch.zeros_like(spatial_context.observation).to(
                        theta
                    )
                    field_previous_update = torch.zeros_like(field_gradient)
            else:
                total_gradient = (
                    directions * factor_mask[..., None].to(directions.dtype)
                ).sum(dim=1)
                previous_update = (
                    torch.zeros_like(theta)
                    if state is None
                    else state.solver_state.previous_update.to(theta)
                )
                field_indices = spatial_context.parameter_indices
                if field_indices is None:
                    field_theta = theta
                    field_gradient = total_gradient
                    field_previous_update = previous_update
                    field_compact_update = proposal.update
                else:
                    field_indices = field_indices.to(theta.device)
                    field_theta = theta.index_select(1, field_indices)
                    field_gradient = total_gradient.index_select(1, field_indices)
                    field_previous_update = previous_update.index_select(
                        1, field_indices
                    )
                    field_compact_update = proposal.update.index_select(
                        1, field_indices
                    )
                if isinstance(
                    self.spatial_field_head, SpatialFieldMixtureUpdateHead
                ):
                    blocks = int(torch.unique(block_id).numel())
                    observation = spatial_context.observation.to(field_theta)
                    gradient_rms = field_gradient.square().mean(dim=-1).sqrt()
                    previous_rms = field_previous_update.square().mean(dim=-1).sqrt()
                    if self.spatial_field_head.global_feature_dim <= 9:
                        global_router_features = torch.stack(
                            (
                                torch.log1p(
                                    (current_values.detach() * mask_float)
                                    .sum(dim=-1)
                                    .abs()
                                    / max(factors, 1)
                                )
                                / 8.0,
                                torch.log10(gradient_rms.clamp_min(1.0e-12)) / 8.0,
                                observation.mean(dim=-1),
                                observation.std(dim=-1, unbiased=False),
                                (field_theta.detach() - observation)
                                .square()
                                .mean(dim=-1)
                                .sqrt(),
                                previous_rms,
                                theta.new_full(
                                    (batch,), math.log1p(factors) / 8.0
                                ),
                                theta.new_full(
                                    (batch,), math.log1p(field_theta.shape[1]) / 10.0
                                ),
                                theta.new_full(
                                    (batch,), math.log1p(blocks) / 5.0
                                ),
                            ),
                            dim=-1,
                        ).detach()
                    elif self.spatial_field_head.global_feature_dim <= 32:
                        compact_candidate_values = factor_fn(
                            theta.detach() + proposal.update.detach()
                        ).detach()
                        global_router_features = spatial_operator_probe_features(
                            theta=field_theta,
                            observation=observation,
                            current_values=current_values,
                            compact_candidate_values=compact_candidate_values,
                            total_gradient=field_gradient.detach(),
                            previous_update=field_previous_update,
                            compact_update=field_compact_update.detach(),
                            factor_mask=factor_mask,
                            block_id=block_id,
                            field_shape=spatial_context.field_shape,
                        )[:, : self.spatial_field_head.global_feature_dim]
                    else:
                        raise ValueError(
                            "spatial router feature dimension exceeds available probes"
                        )
                    spatial_probe_features = global_router_features
                    field_spatial_update, spatial_trust_offset = self.spatial_field_head(
                        field_theta,
                        spatial_context.observation,
                        field_gradient.detach(),
                        field_previous_update,
                        spatial_context.field_shape,
                        forced_expert=force_spatial_expert,
                        forced_routing=force_spatial_routing,
                        global_features=(
                            global_router_features
                            if self.spatial_field_head.global_feature_dim
                            else None
                        ),
                    )
                else:
                    field_spatial_update, spatial_trust_offset = self.spatial_field_head(
                        field_theta,
                        spatial_context.observation,
                        field_gradient.detach(),
                        field_previous_update,
                        spatial_context.field_shape,
                    )
                if spatial_context.update_mask_fn is not None:
                    field_spatial_update = field_spatial_update * spatial_context.update_mask_fn(theta)

                if field_indices is None:
                    spatial_update = field_spatial_update
                else:
                    spatial_update = torch.zeros_like(theta).scatter(
                        1,
                        field_indices[None].expand(batch, -1),
                        field_spatial_update,
                    )
                recurrent_trust = (
                    theta.new_ones(batch) if state is None else state.trust_scale
                )
                head_policy_active = getattr(
                    self.spatial_field_head, "last_spatial_policy_active", None
                )
                spatial_policy_active = (
                    torch.ones(batch, device=theta.device, dtype=torch.bool)
                    if head_policy_active is None
                    else head_policy_active.to(device=theta.device, dtype=torch.bool)
                )
                spatial_update = recurrent_trust[:, None] * spatial_update
                proposal = replace(
                    proposal, update=proposal.update + spatial_update
                )
                safe_confidence = getattr(
                    self.spatial_field_head,
                    "last_safe_residual_confidence",
                    None,
                )
                if safe_confidence is not None:
                    if safe_confidence.shape != (batch,):
                        raise RuntimeError("safe residual confidence has the wrong shape")
                    if field_indices is None:
                        proposal = replace(
                            proposal,
                            update=proposal.update * safe_confidence[:, None],
                        )
                        spatial_update = spatial_update * safe_confidence[:, None]
                    else:
                        field_update = proposal.update.index_select(1, field_indices)
                        gated_field_update = field_update * safe_confidence[:, None]
                        proposal = replace(
                            proposal,
                            update=proposal.update.scatter(
                                1,
                                field_indices[None].expand(batch, -1),
                                gated_field_update,
                            ),
                        )
                        spatial_update = spatial_update * safe_confidence[:, None]
            pacing_active = (
                self.spatial_joint_latent_pacing_head is not None
                and spatial_context is not None
                and spatial_context.parameter_indices is not None
                and query_factor_fn is not None
            )
            joint_latent_unpaced_update = proposal.update.detach()
            if not pacing_active:
                joint_latent_pacing_logit = theta.new_full((batch,), 20.0)
                joint_latent_pacing_scale = theta.new_ones(batch)
                joint_latent_pacing_gate_logit = theta.new_full((batch,), 20.0)
                joint_latent_pacing_gate_probability = theta.new_ones(batch)
                joint_latent_pacing_gate_is_active = torch.ones(
                    batch, device=theta.device, dtype=torch.bool
                )
                joint_latent_continue_logit = theta.new_full((batch,), 20.0)
                joint_latent_continue_probability = theta.new_ones(batch)
                joint_latent_continue_is_active = torch.ones(
                    batch, device=theta.device, dtype=torch.bool
                )
                pacing_features = theta.new_zeros(batch, 0)
                typed_continue_features = theta.new_zeros(batch, 0)
            else:
                pacing_head = self.spatial_joint_latent_pacing_head
                field_indices = spatial_context.parameter_indices.to(theta.device)
                coordinate_mask = torch.ones(
                    theta.shape[1], device=theta.device, dtype=torch.bool
                )
                coordinate_mask[field_indices] = False
                latent_indices = coordinate_mask.nonzero(as_tuple=False).flatten()
                if latent_indices.numel() == 0:
                    raise RuntimeError("joint-latent pacing requires latent coordinates")
                raw_candidate = theta + proposal.update
                raw_candidate_values = factor_fn(raw_candidate)
                if raw_candidate_values.shape != current_values.shape:
                    raise ValueError("factor_fn changed shape during pacing")
                pacing_trust_features = self._trust_features(
                    theta=theta,
                    current_values=canonical_current_values.detach(),
                    candidate_values=canonicalize_factor_values(
                        raw_candidate_values
                    ).detach(),
                    factor_mask=factor_mask,
                    proposal=(
                        proposal if canonical_scale is None else trust_proposal
                    ),
                    step_scale=theta.new_ones(batch),
                    state=state,
                    block_id=block_id,
                )
                pacing_probe_dim = pacing_head.spatial_probe_dim
                if spatial_probe_features is None:
                    pacing_probe_features = theta.new_zeros(
                        batch, pacing_probe_dim
                    )
                elif spatial_probe_features.shape[1] < pacing_probe_dim:
                    pacing_probe_features = torch.nn.functional.pad(
                        spatial_probe_features,
                        (0, pacing_probe_dim - spatial_probe_features.shape[1]),
                    )
                else:
                    pacing_probe_features = spatial_probe_features[
                        :, :pacing_probe_dim
                    ].to(theta)
                with torch.no_grad():
                    current_query_values = query_factor_fn(theta.detach())
                    candidate_query_values = query_factor_fn(
                        raw_candidate.detach()
                    )
                query_features = self._query_response_features(
                    current_query_values.to(theta),
                    candidate_query_values.to(theta),
                )
                total_gradient = (
                    directions * factor_mask[..., None].to(directions.dtype)
                ).sum(dim=1).detach()
                previous_update = (
                    torch.zeros_like(theta)
                    if state is None
                    else state.solver_state.previous_update.to(theta)
                ).detach()
                proposal_update = proposal.update.detach()

                def rms(value: Tensor) -> Tensor:
                    return value.square().mean(dim=-1).sqrt()

                def cosine(first: Tensor, second: Tensor) -> Tensor:
                    numerator = (first * second).sum(dim=-1)
                    denominator = (
                        first.square().sum(dim=-1).sqrt()
                        * second.square().sum(dim=-1).sqrt()
                    ).clamp_min(1.0e-8)
                    return (numerator / denominator).clamp(-1.0, 1.0)

                field_gradient = total_gradient.index_select(1, field_indices)
                latent_gradient = total_gradient.index_select(1, latent_indices)
                field_previous = previous_update.index_select(1, field_indices)
                latent_previous = previous_update.index_select(1, latent_indices)
                field_proposal = proposal_update.index_select(1, field_indices)
                latent_proposal = proposal_update.index_select(1, latent_indices)
                field_gradient_rms = rms(field_gradient)
                latent_gradient_rms = rms(latent_gradient)
                recurrent_trust = (
                    theta.new_ones(batch) if state is None else state.trust_scale
                )
                latent_features = torch.stack(
                    (
                        theta.new_full(
                            (batch,), float(step_index + 1) / self.steps
                        ),
                        torch.log10(field_gradient_rms.clamp_min(1.0e-12)) / 8.0,
                        torch.log10(latent_gradient_rms.clamp_min(1.0e-12)) / 8.0,
                        torch.tanh(
                            torch.log(
                                (field_gradient_rms + 1.0e-12)
                                / (latent_gradient_rms + 1.0e-12)
                            )
                        ),
                        torch.log10(rms(field_previous).clamp_min(1.0e-12)) / 8.0,
                        torch.log10(rms(latent_previous).clamp_min(1.0e-12)) / 8.0,
                        torch.log10(rms(field_proposal).clamp_min(1.0e-12)) / 8.0,
                        torch.log10(rms(latent_proposal).clamp_min(1.0e-12)) / 8.0,
                        cosine(-field_gradient, field_proposal),
                        cosine(-latent_gradient, latent_proposal),
                        cosine(field_previous, field_proposal),
                        cosine(latent_previous, latent_proposal),
                        recurrent_trust.detach(),
                    ),
                    dim=-1,
                )
                history_feature_dim = pacing_head.history_feature_dim
                if history_feature_dim == 0:
                    history_features = theta.new_zeros(batch, 0)
                else:
                    current_query_score = current_query_values.to(theta).mean(dim=-1)
                    candidate_query_score = candidate_query_values.to(theta).mean(dim=-1)
                    if pacing_initial_query_score is None:
                        pacing_initial_query_score = current_query_score.detach()
                        pacing_previous_query_score = current_query_score.detach()
                    assert pacing_previous_query_score is not None
                    query_denominator = pacing_initial_query_score.abs().clamp_min(1.0e-8)
                    previous_query_denominator = (
                        pacing_previous_query_score.abs().clamp_min(1.0e-8)
                    )
                    if pacing_previous_latent_gradient is None:
                        previous_latent_alignment = theta.new_zeros(batch)
                        latent_gradient_ratio = theta.new_zeros(batch)
                    else:
                        previous_latent_alignment = cosine(
                            pacing_previous_latent_gradient, latent_gradient
                        )
                        latent_gradient_ratio = torch.tanh(
                            torch.log(
                                (rms(latent_gradient) + 1.0e-12)
                                / (
                                    rms(pacing_previous_latent_gradient)
                                    + 1.0e-12
                                )
                            )
                        )
                    history_features = torch.stack(
                        (
                            torch.tanh(
                                torch.log(
                                    (current_query_score.abs() + 1.0e-8)
                                    / (pacing_initial_query_score.abs() + 1.0e-8)
                                )
                            ),
                            torch.tanh(
                                (pacing_initial_query_score - current_query_score)
                                / query_denominator
                            ),
                            torch.tanh(
                                (pacing_previous_query_score - current_query_score)
                                / previous_query_denominator
                            ),
                            torch.tanh(
                                (current_query_score - candidate_query_score)
                                / current_query_score.abs().clamp_min(1.0e-8)
                            ),
                            previous_latent_alignment,
                            latent_gradient_ratio,
                        ),
                        dim=-1,
                    )
                    pacing_previous_query_score = current_query_score.detach()
                    pacing_previous_latent_gradient = latent_gradient.detach()
                pacing_features = torch.cat(
                    (
                        pacing_trust_features,
                        pacing_probe_features.detach(),
                        query_features,
                        latent_features,
                        history_features,
                    ),
                    dim=-1,
                )
                if typed_observable_fn is None:
                    typed_continue_features = theta.new_zeros(batch, 0)
                else:
                    if pacing_head.typed_continue_gate_network is None:
                        raise RuntimeError(
                            "typed pacing functions require a configured typed gate"
                        )
                    assert coordinate_metric_fn is not None
                    with torch.no_grad():
                        current_typed = typed_observable_fn(theta.detach()).to(theta)
                        candidate_typed = typed_observable_fn(
                            raw_candidate.detach()
                        ).to(theta)
                        coordinate_metric = coordinate_metric_fn(
                            theta.detach()
                        ).to(theta)
                    if current_typed.shape != (batch, 2) or candidate_typed.shape != (
                        batch,
                        2,
                    ):
                        raise ValueError(
                            "M35 typed pacing requires two noise observables"
                        )
                    if (
                        coordinate_metric.shape != theta.shape
                        or not torch.isfinite(coordinate_metric).all()
                        or torch.any(coordinate_metric <= 0.0)
                    ):
                        raise ValueError(
                            "coordinate metric must be finite, positive, and match theta"
                        )
                    field_metric = coordinate_metric.index_select(1, field_indices)
                    latent_metric = coordinate_metric.index_select(1, latent_indices)
                    field_gradient_energy = metric_dual_energy(
                        field_gradient, field_metric
                    )
                    latent_gradient_energy = metric_dual_energy(
                        latent_gradient, latent_metric
                    )
                    field_update_energy = metric_tangent_energy(
                        field_proposal, field_metric
                    )
                    latent_update_energy = metric_tangent_energy(
                        latent_proposal, latent_metric
                    )
                    if pacing_previous_total_gradient is None:
                        field_history_alignment = theta.new_zeros(batch)
                        latent_history_alignment = theta.new_zeros(batch)
                        field_history_energy_ratio = theta.new_zeros(batch)
                        latent_history_energy_ratio = theta.new_zeros(batch)
                    else:
                        previous_field_gradient = (
                            pacing_previous_total_gradient.index_select(
                                1, field_indices
                            )
                        )
                        previous_latent_gradient = (
                            pacing_previous_total_gradient.index_select(
                                1, latent_indices
                            )
                        )
                        field_history_alignment = metric_dual_cosine(
                            previous_field_gradient,
                            field_gradient,
                            field_metric,
                        )
                        latent_history_alignment = metric_dual_cosine(
                            previous_latent_gradient,
                            latent_gradient,
                            latent_metric,
                        )
                        field_history_energy_ratio = bounded_log_ratio(
                            field_gradient_energy,
                            metric_dual_energy(
                                previous_field_gradient, field_metric
                            ),
                        )
                        latent_history_energy_ratio = bounded_log_ratio(
                            latent_gradient_energy,
                            metric_dual_energy(
                                previous_latent_gradient, latent_metric
                            ),
                        )
                    current_support_score = current_values.detach().sum(dim=1)
                    candidate_support_score = raw_candidate_values.detach().sum(dim=1)
                    actual_reduction = (
                        current_support_score - candidate_support_score
                    )
                    predicted_reduction = -(
                        total_gradient * proposal_update
                    ).sum(dim=1)
                    typed_response = bounded_log_ratio(
                        current_typed, candidate_typed
                    )
                    typed_continue_features = torch.stack(
                        (
                            theta.new_full(
                                (batch,), float(step_index + 1) / self.steps
                            ),
                            current_typed[:, 0],
                            current_typed[:, 1],
                            candidate_typed[:, 0],
                            candidate_typed[:, 1],
                            typed_response[:, 0],
                            typed_response[:, 1],
                            torch.tanh(
                                actual_reduction
                                / current_support_score.abs().clamp_min(1.0e-12)
                            ),
                            torch.tanh(
                                actual_reduction
                                / predicted_reduction.abs().clamp_min(1.0e-12)
                            ),
                            bounded_log_ratio(
                                latent_gradient_energy, field_gradient_energy
                            ),
                            bounded_log_ratio(
                                latent_update_energy, field_update_energy
                            ),
                            metric_gradient_update_alignment(
                                field_gradient,
                                field_proposal,
                                field_metric,
                            ),
                            metric_gradient_update_alignment(
                                latent_gradient,
                                latent_proposal,
                                latent_metric,
                            ),
                            field_history_alignment,
                            latent_history_alignment,
                            field_history_energy_ratio,
                            latent_history_energy_ratio,
                        ),
                        dim=-1,
                    ).detach()
                    if typed_continue_features.shape != (
                        batch,
                        SpatialJointLatentPacingHead.TYPED_CONTINUE_FEATURE_DIM,
                    ):
                        raise RuntimeError("typed continue feature contract changed")
                    if not torch.isfinite(typed_continue_features).all():
                        raise RuntimeError("typed continue features must be finite")
                    pacing_previous_total_gradient = total_gradient.detach()
                gate_observation_features = (
                    spatial_observation_summary_features(
                        spatial_context.observation.to(theta),
                        spatial_context.field_shape,
                    )
                    if pacing_head.gate_observation_feature_dim > 0
                    else None
                )
                joint_latent_pacing_logit = pacing_head(
                    pacing_features,
                    gate_observation_features=gate_observation_features,
                    typed_continue_features=(
                        typed_continue_features
                        if typed_continue_features.shape[1] > 0
                        else None
                    ),
                )
                typed_policy_active = typed_continue_features.shape[1] > 0
                continuous_pacing_scale = (
                    theta.new_ones(batch)
                    if typed_policy_active
                    else torch.sigmoid(joint_latent_pacing_logit)
                )
                if pacing_head.exact_null_gate and not typed_policy_active:
                    if pacing_head.last_gate_logit is None:
                        raise RuntimeError("exact-null pacing did not expose a gate logit")
                    joint_latent_pacing_gate_logit = pacing_head.last_gate_logit
                    joint_latent_pacing_gate_probability = torch.sigmoid(
                        joint_latent_pacing_gate_logit
                    )
                    joint_latent_pacing_gate_is_active = (
                        joint_latent_pacing_gate_probability >= 0.5
                    )
                    if pacing_head.lock_exact_null_gate:
                        if locked_joint_latent_gate_active is None:
                            locked_joint_latent_gate_logit = (
                                joint_latent_pacing_gate_logit
                            )
                            locked_joint_latent_gate_probability = (
                                joint_latent_pacing_gate_probability
                            )
                            locked_joint_latent_gate_active = (
                                joint_latent_pacing_gate_is_active.detach()
                            )
                        assert locked_joint_latent_gate_logit is not None
                        assert locked_joint_latent_gate_probability is not None
                        joint_latent_pacing_gate_logit = (
                            locked_joint_latent_gate_logit
                        )
                        joint_latent_pacing_gate_probability = (
                            locked_joint_latent_gate_probability
                        )
                        joint_latent_pacing_gate_is_active = (
                            locked_joint_latent_gate_active
                        )
                    hard_gate = joint_latent_pacing_gate_is_active.to(theta.dtype)
                    # Forward is exactly binary.  The straight-through path is
                    # active only while training so gate supervision can reach
                    # the shared mathematical feature extractor.
                    gate = (
                        hard_gate
                        + joint_latent_pacing_gate_probability
                        - joint_latent_pacing_gate_probability.detach()
                        if self.training
                        else hard_gate
                    )
                    joint_latent_pacing_scale = continuous_pacing_scale * gate
                else:
                    joint_latent_pacing_gate_logit = theta.new_full((batch,), 20.0)
                    joint_latent_pacing_gate_probability = theta.new_ones(batch)
                    joint_latent_pacing_gate_is_active = torch.ones(
                        batch, device=theta.device, dtype=torch.bool
                    )
                    joint_latent_pacing_scale = continuous_pacing_scale
                if pacing_head.last_continue_gate_logit is None:
                    joint_latent_continue_logit = theta.new_full((batch,), 20.0)
                    joint_latent_continue_probability = theta.new_ones(batch)
                    joint_latent_continue_is_active = torch.ones(
                        batch, device=theta.device, dtype=torch.bool
                    )
                    continue_gate = theta.new_ones(batch)
                else:
                    joint_latent_continue_logit = (
                        pacing_head.last_continue_gate_logit
                    )
                    joint_latent_continue_probability = torch.sigmoid(
                        joint_latent_continue_logit
                    )
                    joint_latent_continue_is_active = (
                        joint_latent_continue_probability >= 0.5
                    )
                    if pacing_head.continue_gate_absorbing:
                        if joint_latent_continue_alive is None:
                            joint_latent_continue_alive = torch.ones(
                                batch, device=theta.device, dtype=torch.bool
                            )
                        joint_latent_continue_is_active = (
                            joint_latent_continue_is_active
                            & joint_latent_continue_alive
                        )
                        joint_latent_continue_alive = (
                            joint_latent_continue_is_active.detach()
                        )
                    hard_continue = joint_latent_continue_is_active.to(theta.dtype)
                    continue_gate = (
                        hard_continue
                        + joint_latent_continue_probability
                        - joint_latent_continue_probability.detach()
                        if self.training
                        else hard_continue
                    )
                joint_latent_pacing_scale = (
                    joint_latent_pacing_scale * continue_gate
                )
                field_update = proposal.update.index_select(1, field_indices)
                paced_field_update = (
                    field_update * joint_latent_pacing_scale[:, None]
                )
                proposal = replace(
                    proposal,
                    update=proposal.update.scatter(
                        1,
                        field_indices[None].expand(batch, -1),
                        paced_field_update,
                    ),
                )
                spatial_update = (
                    spatial_update * joint_latent_pacing_scale[:, None]
                )
            blocks = int(block_id.max().item()) + 1
            schema_expert_admission = torch.zeros(
                batch, blocks, device=theta.device, dtype=torch.bool
            )
            incidence_overdispersed = getattr(
                self.solver,
                "last_schema_topology_pacing_active",
                None,
            )
            if incidence_overdispersed is None:
                incidence_overdispersed = torch.zeros(
                    batch, device=theta.device, dtype=torch.bool
                )
            schema_expert_admission = torch.where(
                incidence_overdispersed[:, None],
                torch.zeros_like(schema_expert_admission),
                torch.ones_like(schema_expert_admission),
            )
            schema_expert_admissions.append(
                schema_expert_admission
            )
            schema_topology_pacing_steps.append(
                incidence_overdispersed
            )
            full_candidate = theta + proposal.update
            full_candidate_values = factor_fn(full_candidate)
            if full_candidate_values.shape != current_values.shape:
                raise ValueError("factor_fn changed shape during optimization")
            current_score = (
                canonical_current_values.detach() * mask_float
            ).sum(dim=-1)
            full_candidate_score = (
                (
                    full_candidate_values.detach()
                    if canonical_factor_scale is None
                    else full_candidate_values.detach()
                    / canonical_factor_scale[:, None]
                )
                * mask_float
            ).sum(dim=-1)
            accepted = torch.zeros(batch, device=theta.device, dtype=torch.bool)
            step_scale = theta.new_zeros(batch)
            selected_theta = theta
            selected_values = current_values
            selected_score = current_score
            for scale_index, scale in enumerate(self.line_search_scales):
                if scale_index == 0:
                    candidate = full_candidate
                    candidate_values = full_candidate_values
                else:
                    candidate = theta + scale * proposal.update
                    candidate_values = factor_fn(candidate)
                canonical_candidate_values = (
                    candidate_values.detach()
                    if canonical_factor_scale is None
                    else candidate_values.detach()
                    / canonical_factor_scale[:, None]
                )
                candidate_score = (
                    canonical_candidate_values * mask_float
                ).sum(dim=-1)
                choose = (~accepted) & (
                    candidate_score
                    <= current_score + self.acceptance_tolerance
                )
                selected_theta = torch.where(
                    choose[:, None], candidate, selected_theta
                )
                selected_values = torch.where(
                    choose[:, None], candidate_values, selected_values
                )
                selected_score = torch.where(choose, candidate_score, selected_score)
                step_scale = torch.where(
                    choose, theta.new_full((batch,), scale), step_scale
                )
                accepted = accepted | choose
                if self.short_circuit_line_search and bool(accepted.all()):
                    break
            support_accepted = accepted
            support_theta = selected_theta
            support_values = selected_values
            support_scale = step_scale
            spatial_override_enabled = spatial_policy_active & (
                self.allow_spatial_support_override or force_spatial_accept
            )
            use_full_candidate = spatial_override_enabled & (~support_accepted)
            decision_theta = torch.where(
                use_full_candidate[:, None], full_candidate, support_theta
            )
            decision_values = torch.where(
                use_full_candidate[:, None], full_candidate_values, support_values
            )
            decision_score = torch.where(
                use_full_candidate, full_candidate_score, selected_score
            )
            decision_scale = torch.where(
                use_full_candidate, theta.new_ones(batch), support_scale
            )
            trust_features: Optional[Tensor] = None
            if self.generalization_trust_head is None:
                transfer_logit = theta.new_full((batch,), 20.0)
                transfer_probability = theta.new_ones(batch)
                transfer_delta_prediction = theta.new_zeros(batch)
            else:
                trust_features = self._trust_features(
                    theta=theta,
                    current_values=canonical_current_values.detach(),
                    candidate_values=canonicalize_factor_values(
                        decision_values
                    ).detach(),
                    factor_mask=factor_mask,
                    proposal=(
                        proposal if canonical_scale is None else trust_proposal
                    ),
                    step_scale=decision_scale,
                    state=state,
                    block_id=block_id,
                )
                transfer_logit, transfer_delta_prediction = (
                    self.generalization_trust_head(trust_features)
                )
                transfer_logit = transfer_logit + spatial_trust_offset
                transfer_probability = torch.sigmoid(transfer_logit)
            if spatial_context is not None and self.spatial_transfer_threshold is not None:
                threshold = torch.where(
                    spatial_policy_active,
                    theta.new_full((batch,), self.spatial_transfer_threshold),
                    theta.new_full((batch,), self.transfer_threshold),
                )
            else:
                threshold = theta.new_full((batch,), self.transfer_threshold)
            transfer_pass = transfer_probability >= threshold
            relative_support_increase = (
                (decision_score - current_score)
                / current_score.abs().clamp_min(1.0e-8)
            )
            within_maximum_support_cap = (
                relative_support_increase <= self.spatial_max_support_increase
            )
            if self.spatial_expanded_support_admission_head is None:
                expanded_support_admission_logit = theta.new_full((batch,), 20.0)
                expanded_support_admission_probability = theta.new_ones(batch)
                override_eligible = (
                    spatial_override_enabled
                    & (~support_accepted)
                    & within_maximum_support_cap
                )
                expanded_override_eligible = torch.zeros_like(override_eligible)
            else:
                if trust_features is None:
                    trust_features = self._trust_features(
                        theta=theta,
                        current_values=canonical_current_values.detach(),
                        candidate_values=canonicalize_factor_values(
                            decision_values
                        ).detach(),
                        factor_mask=factor_mask,
                        proposal=(
                            proposal if canonical_scale is None else trust_proposal
                        ),
                        step_scale=decision_scale,
                        state=state,
                        block_id=block_id,
                    )
                admission_features = trust_features
                probe_dim = (
                    self.spatial_expanded_support_admission_head.spatial_probe_dim
                )
                if probe_dim > 0:
                    if spatial_probe_features is None:
                        probe_features = theta.new_zeros(batch, probe_dim)
                    elif spatial_probe_features.shape[1] < probe_dim:
                        raise RuntimeError(
                            "spatial head exposes fewer probes than admission requires"
                        )
                    else:
                        probe_features = spatial_probe_features[:, :probe_dim].to(theta)
                    admission_features = torch.cat(
                        (admission_features, probe_features), dim=-1
                    )
                expanded_support_admission_logit = (
                    self.spatial_expanded_support_admission_head(admission_features)
                )
                expanded_support_admission_probability = torch.sigmoid(
                    expanded_support_admission_logit
                )
                if locked_expanded_support_admission is None:
                    locked_expanded_support_admission = (
                        expanded_support_admission_probability
                        >= self.spatial_expanded_support_threshold
                    ).detach()
                legacy_override_eligible = (
                    spatial_override_enabled
                    & (~support_accepted)
                    & (
                        relative_support_increase
                        <= self.spatial_legacy_max_support_increase
                    )
                )
                expanded_override_eligible = (
                    spatial_override_enabled
                    & (~support_accepted)
                    & within_maximum_support_cap
                    & (
                        relative_support_increase
                        > self.spatial_legacy_max_support_increase
                    )
                    & locked_expanded_support_admission
                )
                override_eligible = (
                    legacy_override_eligible | expanded_override_eligible
                )
            if force_spatial_accept:
                accepted = torch.ones(batch, device=theta.device, dtype=torch.bool)
                selected_theta = full_candidate
                selected_values = full_candidate_values
                selected_score = full_candidate_score
                step_scale = theta.new_ones(batch)
                support_override = ~support_accepted
            else:
                accepted = transfer_pass & (support_accepted | override_eligible)
                selected_theta = torch.where(
                    accepted[:, None], decision_theta, theta
                )
                selected_values = torch.where(
                    accepted[:, None], decision_values, current_values
                )
                selected_score = torch.where(
                    accepted, decision_score, current_score
                )
                step_scale = torch.where(
                    accepted, decision_scale, theta.new_zeros(batch)
                )
                support_override = accepted & (~support_accepted)
            expanded_support_override = support_override & expanded_override_eligible
            accepted_update = step_scale[:, None] * proposal.update

            # The terminal decision is still made inside the fifth learned
            # call.  These summaries expose only coordinate-invariant rollout
            # geometry, never a task label or query target.
            if terminal_initial_score is None:
                terminal_initial_score = current_score.detach()
            score_denominator = terminal_initial_score.abs().clamp_min(1.0e-8)
            proposal_ratio = full_candidate_score.detach() / score_denominator
            selected_ratio = selected_score.detach() / score_denominator
            if terminal_first_proposal_ratio is None:
                terminal_first_proposal_ratio = proposal_ratio
                terminal_max_proposal_ratio = proposal_ratio
                terminal_min_selected_ratio = selected_ratio
                terminal_initial_transfer_probability = transfer_probability.detach()
            else:
                terminal_max_proposal_ratio = torch.maximum(
                    terminal_max_proposal_ratio, proposal_ratio
                )
                terminal_min_selected_ratio = torch.minimum(
                    terminal_min_selected_ratio, selected_ratio
                )
            terminal_acceptance_count = (
                terminal_acceptance_count + accepted.detach().to(theta.dtype)
            )
            terminal_support_acceptance_count = (
                terminal_support_acceptance_count
                + support_accepted.detach().to(theta.dtype)
            )
            terminal_override_count = (
                terminal_override_count + support_override.detach().to(theta.dtype)
            )
            terminal_expanded_override_count = (
                terminal_expanded_override_count
                + expanded_support_override.detach().to(theta.dtype)
            )
            terminal_cumulative_scale = terminal_cumulative_scale + step_scale.detach()
            terminal_cumulative_update_energy = (
                terminal_cumulative_update_energy
                + accepted_update.detach().square().mean(dim=-1)
            )

            if (
                self.spatial_terminal_rollback_head is None
                or spatial_context is None
                or (
                    self.spatial_terminal_joint_latent_only
                    and spatial_context.parameter_indices is None
                )
            ):
                terminal_keep_logit = theta.new_full((batch,), 20.0)
                terminal_keep_probability = theta.new_ones(batch)
            else:
                if trust_features is None:
                    trust_features = self._trust_features(
                        theta=theta,
                        current_values=canonical_current_values.detach(),
                        candidate_values=canonicalize_factor_values(
                            decision_values
                        ).detach(),
                        factor_mask=factor_mask,
                        proposal=(
                            proposal if canonical_scale is None else trust_proposal
                        ),
                        step_scale=decision_scale,
                        state=state,
                        block_id=block_id,
                    )
                terminal_features = trust_features
                terminal_probe_dim = (
                    self.spatial_terminal_rollback_head.spatial_probe_dim
                )
                if terminal_probe_dim > 0:
                    if spatial_probe_features is None:
                        terminal_probe_features = theta.new_zeros(
                            batch, terminal_probe_dim
                        )
                    elif spatial_probe_features.shape[1] < terminal_probe_dim:
                        raise RuntimeError(
                            "spatial head exposes fewer probes than terminal rollback requires"
                        )
                    else:
                        terminal_probe_features = spatial_probe_features[
                            :, :terminal_probe_dim
                        ].to(theta)
                    terminal_features = torch.cat(
                        (terminal_features, terminal_probe_features), dim=-1
                    )
                trajectory_dim = self.spatial_terminal_rollback_head.trajectory_dim
                if trajectory_dim > 0:
                    completed_steps = float(step_index + 1)

                    def signed_log_ratio(value: Tensor) -> Tensor:
                        return torch.sign(value) * torch.log1p(value.abs())

                    displacement_rms = (
                        selected_theta.detach() - initial_theta.detach()
                    ).square().mean(dim=-1).sqrt()
                    update_rms = accepted_update.detach().square().mean(dim=-1).sqrt()
                    field_indices = spatial_context.parameter_indices
                    if field_indices is None:
                        final_field = selected_theta.detach()
                        accepted_field_update = accepted_update.detach()
                    else:
                        field_indices = field_indices.to(theta.device)
                        final_field = selected_theta.detach().index_select(
                            1, field_indices
                        )
                        accepted_field_update = accepted_update.detach().index_select(
                            1, field_indices
                        )
                    observed_field = spatial_context.observation.to(theta)
                    channels, height, width = spatial_context.field_shape

                    def image_statistics(field: Tensor) -> tuple[Tensor, ...]:
                        image = field.reshape(batch, channels, height, width)
                        rms = image.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1.0e-6)
                        dx = image[..., 1:] - image[..., :-1]
                        dy = image[..., 1:, :] - image[..., :-1, :]
                        tv = 0.5 * (
                            dx.abs().mean(dim=(1, 2, 3))
                            + dy.abs().mean(dim=(1, 2, 3))
                        ) / rms
                        smooth = torch.nn.functional.avg_pool2d(
                            image,
                            3,
                            stride=1,
                            padding=1,
                            count_include_pad=False,
                        )
                        high_frequency = (
                            (image - smooth).square().mean(dim=(1, 2, 3)).sqrt()
                            / rms
                        )
                        padded = torch.nn.functional.pad(
                            image, (1, 1, 1, 1), mode="replicate"
                        )
                        laplacian = (
                            4.0 * padded[..., 1:-1, 1:-1]
                            - padded[..., :-2, 1:-1]
                            - padded[..., 2:, 1:-1]
                            - padded[..., 1:-1, :-2]
                            - padded[..., 1:-1, 2:]
                        )
                        flat = image.flatten(1)
                        return (
                            flat.mean(dim=-1),
                            flat.std(dim=-1, unbiased=False),
                            torch.log1p(tv),
                            torch.log1p(high_frequency),
                            torch.log1p(
                                laplacian.square()
                                .mean(dim=(1, 2, 3))
                                .sqrt()
                                / rms
                            ),
                            (flat <= 0.01).to(theta.dtype).mean(dim=-1),
                            (flat >= 0.99).to(theta.dtype).mean(dim=-1),
                        )

                    def vector_cosine(first: Tensor, second: Tensor) -> Tensor:
                        numerator = (first * second).sum(dim=-1)
                        denominator = (
                            first.square().sum(dim=-1).sqrt()
                            * second.square().sum(dim=-1).sqrt()
                        ).clamp_min(1.0e-8)
                        return (numerator / denominator).clamp(-1.0, 1.0)

                    final_statistics = image_statistics(final_field)
                    observed_statistics = image_statistics(observed_field)
                    field_displacement = final_field - observed_field
                    trajectory_features = torch.stack(
                        (
                            theta.new_full((batch,), completed_steps / self.steps),
                            terminal_acceptance_count / completed_steps,
                            terminal_support_acceptance_count / completed_steps,
                            terminal_override_count / completed_steps,
                            terminal_expanded_override_count / completed_steps,
                            signed_log_ratio(selected_ratio),
                            signed_log_ratio(proposal_ratio),
                            signed_log_ratio(terminal_first_proposal_ratio),
                            signed_log_ratio(terminal_max_proposal_ratio),
                            signed_log_ratio(terminal_min_selected_ratio),
                            terminal_cumulative_scale / completed_steps,
                            step_scale.detach(),
                            terminal_initial_transfer_probability,
                            transfer_probability.detach(),
                            transfer_probability.detach()
                            - terminal_initial_transfer_probability,
                            torch.log1p(displacement_rms),
                            torch.log1p(update_rms),
                            torch.log1p(terminal_cumulative_update_energy.sqrt()),
                            final_statistics[0] - observed_statistics[0],
                            final_statistics[1] - observed_statistics[1],
                            final_statistics[2] - observed_statistics[2],
                            final_statistics[3] - observed_statistics[3],
                            final_statistics[4] - observed_statistics[4],
                            final_statistics[5] - observed_statistics[5],
                            final_statistics[6] - observed_statistics[6],
                            (
                                (final_field < 0.0) | (final_field > 1.0)
                            ).to(theta.dtype).mean(dim=-1),
                            field_displacement.abs().mean(dim=-1),
                            field_displacement.abs().max(dim=-1).values,
                            vector_cosine(-field_gradient.detach(), accepted_field_update),
                            vector_cosine(
                                field_previous_update.detach(), accepted_field_update
                            ),
                            vector_cosine(
                                -field_gradient.detach(), field_previous_update.detach()
                            ),
                            (
                                field_displacement.sign()
                                == (-field_gradient.detach()).sign()
                            ).to(theta.dtype).mean(dim=-1),
                        ),
                        dim=-1,
                    )[:, :trajectory_dim]
                    terminal_features = torch.cat(
                        (terminal_features, trajectory_features), dim=-1
                    )
                terminal_keep_logit = self.spatial_terminal_rollback_head(
                    terminal_features
                )
                terminal_keep_probability = torch.sigmoid(terminal_keep_logit)
            truncate_inference_graph = inference_only
            theta = (
                selected_theta.detach().requires_grad_(True)
                if truncate_inference_graph
                else selected_theta
            )

            reduction = (current_score - selected_score) / current_score.abs().clamp_min(
                1.0e-8
            )
            previous_trust = (
                theta.new_ones(batch) if state is None else state.trust_scale
            )
            next_trust = torch.where(
                accepted,
                (
                    self.acceptance_growth * previous_trust * step_scale
                ).clamp_max(1.0),
                self.rejection_shrink * previous_trust,
            ).detach()
            canonical_accepted_update = (
                accepted_update
                if canonical_scale is None
                else accepted_update / canonical_scale
            )
            solver_state = replace(
                proposal.proposed_solver_state,
                previous_update=canonical_accepted_update,
            )
            if truncate_inference_graph:
                solver_state = DifferentialFactorState(
                    global_hidden=solver_state.global_hidden.detach(),
                    factor_hidden=solver_state.factor_hidden.detach(),
                    block_hidden=solver_state.block_hidden.detach(),
                    previous_update=solver_state.previous_update.detach(),
                )
            state = DescentAnchoredState(
                solver_state=solver_state,
                trust_scale=next_trust,
                previous_acceptance=accepted.to(theta.dtype).detach(),
                previous_reduction=reduction.detach(),
            )

            theta_trace.append(theta.detach())
            value_trace.append(selected_values.detach())
            proposal_values.append(
                full_candidate_values.detach()
                if truncate_inference_graph
                else full_candidate_values
            )
            proposal_thetas.append(
                full_candidate.detach() if truncate_inference_graph else full_candidate
            )
            acceptances.append(accepted)
            selected_scales.append(step_scale)
            updates.append(accepted_update.detach())
            trust_trace.append(next_trust)
            basis_trace.append(proposal.basis_weights.detach())
            topology_trace.append(
                proposal.topology_controls
                if self.training
                else proposal.topology_controls.detach()
            )
            block_context_trace.append(
                proposal.block_context_controls
                if self.training
                else proposal.block_context_controls.detach()
            )
            support_candidate_thetas.append(support_theta.detach())
            support_candidate_values.append(support_values.detach())
            support_acceptances.append(support_accepted.detach())
            transfer_logits.append(
                transfer_logit.detach() if truncate_inference_graph else transfer_logit
            )
            transfer_probabilities.append(transfer_probability.detach())
            transfer_delta_predictions.append(
                transfer_delta_prediction.detach()
                if truncate_inference_graph
                else transfer_delta_prediction
            )
            expanded_support_admission_logits.append(
                expanded_support_admission_logit.detach()
                if truncate_inference_graph
                else expanded_support_admission_logit
            )
            expanded_support_admission_probabilities.append(
                expanded_support_admission_probability.detach()
            )
            expanded_support_overrides.append(expanded_support_override.detach())
            terminal_keep_logits.append(
                terminal_keep_logit.detach()
                if truncate_inference_graph
                else terminal_keep_logit
            )
            terminal_keep_probabilities.append(
                terminal_keep_probability.detach()
            )
            joint_latent_pacing_logits.append(
                joint_latent_pacing_logit.detach()
                if truncate_inference_graph
                else joint_latent_pacing_logit
            )
            joint_latent_pacing_scales.append(
                joint_latent_pacing_scale.detach()
            )
            joint_latent_pacing_gate_logits.append(
                joint_latent_pacing_gate_logit.detach()
                if truncate_inference_graph
                else joint_latent_pacing_gate_logit
            )
            joint_latent_pacing_gate_probabilities.append(
                joint_latent_pacing_gate_probability.detach()
            )
            joint_latent_pacing_gate_active.append(
                joint_latent_pacing_gate_is_active.detach()
            )
            joint_latent_continue_logits.append(
                joint_latent_continue_logit.detach()
                if truncate_inference_graph
                else joint_latent_continue_logit
            )
            joint_latent_continue_probabilities.append(
                joint_latent_continue_probability.detach()
            )
            joint_latent_continue_active.append(
                joint_latent_continue_is_active.detach()
            )
            joint_latent_unpaced_updates.append(
                joint_latent_unpaced_update.detach()
                if truncate_inference_graph
                else joint_latent_unpaced_update
            )
            joint_latent_pacing_features.append(
                pacing_features.detach()
            )
            joint_latent_typed_features.append(
                typed_continue_features.detach()
            )
            spatial_updates.append(spatial_update.detach())
            spatial_trust_offsets.append(
                spatial_trust_offset.detach()
                if truncate_inference_graph
                else spatial_trust_offset
            )
            support_overrides.append(support_override.detach())

        pre_terminal_final_theta = theta
        terminal_rollback = (
            terminal_keep_probabilities[-1]
            < self.spatial_terminal_keep_threshold
        )
        theta = torch.where(
            terminal_rollback[:, None], initial_theta, pre_terminal_final_theta
        )
        if inference_only:
            theta = theta.detach()
            with torch.no_grad():
                final_values = factor_fn(theta)
        else:
            final_values = factor_fn(theta)
        theta_trace[-1] = theta.detach()
        value_trace[-1] = final_values.detach()
        assert state is not None
        return DescentAnchoredOptimizationResult(
            initial_theta=initial_theta,
            final_theta=theta,
            final_factor_values=final_values,
            proposal_factor_value_trace=torch.stack(proposal_values, dim=1),
            proposal_theta_trace=torch.stack(proposal_thetas, dim=1),
            factor_value_trace=torch.stack(value_trace, dim=1),
            theta_trace=torch.stack(theta_trace, dim=1),
            acceptance_trace=torch.stack(acceptances, dim=1),
            step_scale_trace=torch.stack(selected_scales, dim=1),
            update_trace=torch.stack(updates, dim=1),
            trust_trace=torch.stack(trust_trace, dim=1),
            basis_weight_trace=torch.stack(basis_trace, dim=1),
            topology_control_trace=torch.stack(topology_trace, dim=1),
            block_context_control_trace=torch.stack(block_context_trace, dim=1),
            support_candidate_theta_trace=torch.stack(
                support_candidate_thetas, dim=1
            ),
            support_candidate_factor_value_trace=torch.stack(
                support_candidate_values, dim=1
            ),
            support_acceptance_trace=torch.stack(support_acceptances, dim=1),
            transfer_logit_trace=torch.stack(transfer_logits, dim=1),
            transfer_probability_trace=torch.stack(
                transfer_probabilities, dim=1
            ),
            transfer_delta_prediction_trace=torch.stack(
                transfer_delta_predictions, dim=1
            ),
            expanded_support_admission_logit_trace=torch.stack(
                expanded_support_admission_logits, dim=1
            ),
            expanded_support_admission_probability_trace=torch.stack(
                expanded_support_admission_probabilities, dim=1
            ),
            expanded_support_override_trace=torch.stack(
                expanded_support_overrides, dim=1
            ),
            terminal_keep_logit_trace=torch.stack(terminal_keep_logits, dim=1),
            terminal_keep_probability_trace=torch.stack(
                terminal_keep_probabilities, dim=1
            ),
            terminal_rollback=terminal_rollback.detach(),
            pre_terminal_final_theta=pre_terminal_final_theta,
            joint_latent_pacing_logit_trace=torch.stack(
                joint_latent_pacing_logits, dim=1
            ),
            joint_latent_pacing_scale_trace=torch.stack(
                joint_latent_pacing_scales, dim=1
            ),
            joint_latent_pacing_gate_logit_trace=torch.stack(
                joint_latent_pacing_gate_logits, dim=1
            ),
            joint_latent_pacing_gate_probability_trace=torch.stack(
                joint_latent_pacing_gate_probabilities, dim=1
            ),
            joint_latent_pacing_gate_active_trace=torch.stack(
                joint_latent_pacing_gate_active, dim=1
            ),
            joint_latent_continue_logit_trace=torch.stack(
                joint_latent_continue_logits, dim=1
            ),
            joint_latent_continue_probability_trace=torch.stack(
                joint_latent_continue_probabilities, dim=1
            ),
            joint_latent_continue_active_trace=torch.stack(
                joint_latent_continue_active, dim=1
            ),
            joint_latent_unpaced_update_trace=torch.stack(
                joint_latent_unpaced_updates, dim=1
            ),
            joint_latent_pacing_feature_trace=torch.stack(
                joint_latent_pacing_features, dim=1
            ),
            joint_latent_typed_feature_trace=torch.stack(
                joint_latent_typed_features, dim=1
            ),
            spatial_update_trace=torch.stack(spatial_updates, dim=1),
            spatial_trust_offset_trace=torch.stack(spatial_trust_offsets, dim=1),
            support_override_trace=torch.stack(support_overrides, dim=1),
            schema_expert_admission_trace=torch.stack(
                schema_expert_admissions, dim=1
            ),
            schema_topology_pacing_trace=torch.stack(
                schema_topology_pacing_steps, dim=1
            ),
            state=state,
        )


__all__ = [
    "DescentAnchoredDifferentialFactorSolver",
    "DescentAnchoredOptimizationResult",
    "DescentAnchoredState",
    "DescentAnchoredStep",
    "FiveStepDescentAnchoredOptimizer",
    "GeneralizationTrustHead",
    "SpatialExpandedSupportAdmissionHead",
    "SpatialJointLatentPacingHead",
    "SpatialTerminalRollbackHead",
    "SpatialFieldContext",
    "SpatialFieldIndependentMixtureUpdateHead",
    "SpatialFieldMixtureUpdateHead",
    "SpatialFieldUpdateHead",
]
