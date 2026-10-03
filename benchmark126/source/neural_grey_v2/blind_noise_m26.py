"""Task-ID-free Poisson--Gaussian blind-denoising factors for DA-DFS M26-A."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .descent_anchored_solver import SpatialFieldContext
from .factor_sketch import GridFactorSketchSpec, hierarchical_grid_sketch
from .typed_observables import noise_residual_observables


def _logit(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError("logit input must lie in (0,1)")
    return math.log(value / (1.0 - value))


@dataclass(frozen=True)
class BlindNoiseConfig:
    """Frozen observable-model constants for the M26-A diagnostic."""

    shot_range: tuple[float, float] = (5.0e-4, 5.0e-2)
    read_std_range: tuple[float, float] = (5.0e-4, 3.0e-2)
    initial_shot: float = 1.5e-2
    initial_read_std: float = 8.0e-3
    data_weight: float = 1.0
    complexity_weight: float = 2.0e-3
    variance_match_weight: float = 2.0e-2
    gradient_weight: float = 1.5e-3
    latent_prior_weight: float = 1.0e-4
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        for name, bounds in (
            ("shot_range", self.shot_range),
            ("read_std_range", self.read_std_range),
        ):
            if len(bounds) != 2 or not 0.0 <= bounds[0] < bounds[1]:
                raise ValueError(f"{name} must contain increasing non-negative bounds")
        if not self.shot_range[0] < self.initial_shot < self.shot_range[1]:
            raise ValueError("initial_shot must lie strictly inside shot_range")
        if not self.read_std_range[0] < self.initial_read_std < self.read_std_range[1]:
            raise ValueError("initial_read_std must lie strictly inside read_std_range")
        weights = (
            self.data_weight,
            self.complexity_weight,
            self.variance_match_weight,
            self.gradient_weight,
            self.latent_prior_weight,
        )
        if self.data_weight <= 0.0 or any(weight < 0.0 for weight in weights):
            raise ValueError("factor weights must be non-negative with positive data weight")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


def sample_poisson_gaussian(
    clean: Tensor,
    shot: Tensor,
    read_std: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample the differentiable-likelihood Gaussian approximation used by M26-A."""

    if clean.ndim != 4 or clean.shape[1] not in (1, 3):
        raise ValueError("clean must have shape [B,1|3,H,W]")
    if shot.shape != (clean.shape[0],) or read_std.shape != (clean.shape[0],):
        raise ValueError("shot and read_std must have shape [B]")
    if torch.any(shot < 0.0) or torch.any(read_std < 0.0):
        raise ValueError("noise parameters must be non-negative")
    variance = (
        shot[:, None, None, None] * clean.clamp(0.0, 1.0)
        + read_std[:, None, None, None].square()
    )
    noise = torch.randn(
        clean.shape,
        dtype=clean.dtype,
        device=clean.device,
        generator=generator,
    )
    return (clean + noise * variance.clamp_min(0.0).sqrt()).clamp(0.0, 1.0)


class BlindNoiseFactorGraph:
    """Compile one observation into a joint image-plus-noise factor graph."""

    def __init__(self, observation: Tensor, config: BlindNoiseConfig | None = None) -> None:
        if observation.ndim != 4 or observation.shape[1] not in (1, 3):
            raise ValueError("observation must have shape [B,1|3,H,W]")
        if not observation.is_floating_point() or not torch.isfinite(observation).all():
            raise ValueError("observation must be finite floating point")
        if min(observation.shape[-2:]) < 8:
            raise ValueError("observation must be at least 8x8")
        self.observation = observation.detach()
        self.config = config or BlindNoiseConfig()
        self.batch, self.channels, self.height, self.width = observation.shape
        self.image_parameters = self.channels * self.height * self.width
        self.latent_parameters = 2
        self.parameters = self.image_parameters + self.latent_parameters
        self.sketch = GridFactorSketchSpec(
            self.height, self.width, ((1, 1), (2, 2), (4, 4))
        )

    @property
    def factor_count(self) -> int:
        return 4 * self.sketch.factors + 2

    @property
    def query_factor_count(self) -> int:
        return 6

    @property
    def aligned_query_factor_count(self) -> int:
        return 1

    @property
    def typed_observable_count(self) -> int:
        return 2

    @property
    def typed_observable_names(self) -> tuple[str, ...]:
        return (
            "noise_residual_energy_mismatch",
            "noise_residual_lag_correlation",
        )

    @staticmethod
    def _bounded(raw: Tensor, bounds: tuple[float, float]) -> Tensor:
        low, high = bounds
        return low + (high - low) * torch.sigmoid(raw)

    @staticmethod
    def _initial_raw(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        return _logit((value - low) / (high - low))

    def initial_theta(self) -> Tensor:
        raw = self.observation.new_tensor(
            [
                self._initial_raw(self.config.initial_shot, self.config.shot_range),
                self._initial_raw(
                    self.config.initial_read_std, self.config.read_std_range
                ),
            ]
        ).expand(self.batch, -1)
        return torch.cat((self.observation.flatten(1), raw), dim=1)

    def decode(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if theta.ndim != 2 or theta.shape != (self.batch, self.parameters):
            raise ValueError("theta has the wrong blind-noise layout")
        image = theta[:, : self.image_parameters].reshape(
            self.batch, self.channels, self.height, self.width
        )
        raw = theta[:, self.image_parameters :]
        shot = self._bounded(raw[:, 0], self.config.shot_range)
        read_std = self._bounded(raw[:, 1], self.config.read_std_range)
        return image, shot, read_std

    def factor_values(self, theta: Tensor) -> Tensor:
        image, shot, read_std = self.decode(theta)
        clipped_image = image.clamp(0.0, 1.0)
        variance = (
            shot[:, None, None, None] * clipped_image
            + read_std[:, None, None, None].square()
        ).clamp_min(self.config.eps)
        residual = image - self.observation
        data_cost = 0.5 * self.config.data_weight * residual.square() / variance
        data_cost = data_cost.mean(dim=1, keepdim=True)

        complexity_cost = self.config.complexity_weight * torch.log1p(
            variance / self.config.eps
        ).mean(dim=1, keepdim=True)

        observation_low = F.avg_pool2d(
            self.observation, 3, stride=1, padding=1
        )
        # The evidence must not vanish at the deterministic x=y initializer.
        # A residual-only statistic creates the same latent-collapse shortcut
        # seen in blind deblurring: zero residual falsely implies zero noise.
        # The observable high-pass energy is imperfect on edges but supplies an
        # independent, task-free signal whose response changes with noise level.
        observation_high = self.observation - observation_low
        observed_high_energy = F.avg_pool2d(
            observation_high.square().mean(dim=1, keepdim=True),
            3,
            stride=1,
            padding=1,
        )
        mean_variance = variance.mean(dim=1, keepdim=True)
        variance_match = self.config.variance_match_weight * (
            torch.log(observed_high_energy + self.config.eps)
            - torch.log(mean_variance + self.config.eps)
        ).square()

        dx = F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))
        dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
        gradient_cost = self.config.gradient_weight * (
            dx.square() + dy.square() + self.config.eps
        ).sqrt().mean(dim=1, keepdim=True)

        raw = theta[:, self.image_parameters :]
        latent_prior = self.config.latent_prior_weight * raw.square()
        factors = torch.cat(
            (
                hierarchical_grid_sketch(data_cost, self.sketch),
                hierarchical_grid_sketch(complexity_cost, self.sketch),
                hierarchical_grid_sketch(variance_match, self.sketch),
                hierarchical_grid_sketch(gradient_cost, self.sketch),
                latent_prior,
            ),
            dim=1,
        )
        if factors.shape != (self.batch, self.factor_count):
            raise RuntimeError("blind-noise factor count changed")
        return factors

    def query_factor_values(self, theta: Tensor) -> Tensor:
        """Target-free posterior-predictive checks for pacing diagnostics.

        These factors are deliberately excluded from ``factor_values`` and do
        not alter the inherited update trajectory.  The first two terms avoid
        the blind-denoising identity shortcut by checking whether the inferred
        residual has the variance predicted by the joint noise latents.  The
        remaining terms bound observable low-frequency, detail, chroma, and
        range drift.  No clean image or true noise parameter is available here.
        """

        image, shot, read_std = self.decode(theta)
        config = self.config
        clipped = image.clamp(0.0, 1.0)
        variance = (
            shot[:, None, None, None] * clipped
            + read_std[:, None, None, None].square()
        ).clamp_min(config.eps)
        residual = self.observation - image
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=image.device),
            torch.arange(self.width, device=image.device),
            indexing="ij",
        )
        query_mask = ((xx + yy) % 4 == 0).to(image.dtype)[None, None]
        query_count = query_mask.sum().clamp_min(1.0)

        def masked_mean(value: Tensor) -> Tensor:
            return (value * query_mask).sum(dim=(1, 2, 3)) / (
                query_count * value.shape[1]
            )

        whitened_energy = masked_mean(residual.square() / variance)
        whitened_moment = (whitened_energy - 1.0).square()

        residual_energy = F.avg_pool2d(
            residual.square().mean(1, keepdim=True), 3, stride=1, padding=1
        )
        predicted_energy = F.avg_pool2d(
            variance.mean(1, keepdim=True), 3, stride=1, padding=1
        )
        local_variance_calibration = masked_mean(
            (
                torch.log(residual_energy + config.eps)
                - torch.log(predicted_energy + config.eps)
            ).square()
        )

        observation_low = F.avg_pool2d(
            self.observation, 7, stride=1, padding=3
        )
        image_low = F.avg_pool2d(clipped, 7, stride=1, padding=3)
        low_frequency_drift = masked_mean((image_low - observation_low).square())

        observed_high = self.observation - observation_low
        image_high = clipped - image_low
        observed_detail = masked_mean(observed_high.square()).sqrt()
        image_detail = masked_mean(image_high.square()).sqrt()
        detail_drift = torch.log(
            (image_detail + config.eps) / (observed_detail + config.eps)
        ).square()

        if self.channels == 1:
            chroma_drift = image.new_zeros(self.batch)
        else:
            observed_chroma = self.observation / self.observation.sum(
                1, keepdim=True
            ).clamp_min(1.0e-3)
            image_chroma = clipped / clipped.sum(1, keepdim=True).clamp_min(1.0e-3)
            chroma_drift = masked_mean((image_chroma - observed_chroma).square())
        range_drift = (
            torch.relu(-image).square().mean(dim=(1, 2, 3))
            + torch.relu(image - 1.0).square().mean(dim=(1, 2, 3))
        )
        factors = torch.stack(
            (
                whitened_moment,
                local_variance_calibration,
                low_frequency_drift,
                detail_drift,
                chroma_drift,
                range_drift,
            ),
            dim=-1,
        )
        if factors.shape != (self.batch, self.query_factor_count):
            raise RuntimeError("blind-noise query factor count changed")
        if torch.any(factors < 0.0) or not torch.isfinite(factors).all():
            raise RuntimeError("blind-noise query factors must be finite and non-negative")
        return factors

    def aligned_query_factor_values(self, theta: Tensor) -> Tensor:
        """Return only query costs empirically aligned with clean-image regret.

        This is the M31 development candidate.  It intentionally excludes
        detail/chroma distance to the noisy observation: those are useful
        safety descriptors, but minimizing them rewards retaining noise and is
        directionally wrong as a query objective.
        """

        factors = self.query_factor_values(theta)[:, 1:2]
        if factors.shape != (self.batch, self.aligned_query_factor_count):
            raise RuntimeError("blind-noise aligned query factor count changed")
        return factors

    def coordinate_metric_diag(self, theta: Tensor) -> Tensor:
        """Diagonal tangent metric in normalized physical coordinates.

        Image coordinates already live in normalized intensity units.  Each
        raw noise latent is measured through the derivative of its normalized
        bounded physical value, making dual-gradient energy insensitive to
        the numerical width of the configured shot/read ranges.
        """

        self.decode(theta)
        raw = theta[:, self.image_parameters :]
        probability = torch.sigmoid(raw)
        normalized_derivative = probability * (1.0 - probability)
        latent_metric = normalized_derivative.square().clamp_min(
            self.config.eps**2
        )
        image_metric = theta.new_ones(self.batch, self.image_parameters)
        metric = torch.cat((image_metric, latent_metric), dim=1)
        if metric.shape != theta.shape or torch.any(metric <= 0.0):
            raise RuntimeError("blind-noise coordinate metric changed")
        return metric

    def typed_observable_values(self, theta: Tensor) -> Tensor:
        """Return target-free, bounded noise-model observables in ``[0,1]``."""

        image, shot, read_std = self.decode(theta)
        clipped = image.clamp(0.0, 1.0)
        variance = (
            shot[:, None, None, None] * clipped
            + read_std[:, None, None, None].square()
        ).clamp_min(self.config.eps)
        residual = self.observation - image
        values = noise_residual_observables(
            residual,
            variance,
            eps=self.config.eps,
        ).as_tensor()
        if values.shape != (self.batch, self.typed_observable_count):
            raise RuntimeError("blind-noise typed observable count changed")
        return values

    def block_id(self) -> Tensor:
        y = torch.arange(self.height, device=self.observation.device)
        x = torch.arange(self.width, device=self.observation.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        image_blocks = (
            torch.div(yy * 4, self.height, rounding_mode="floor") * 4
            + torch.div(xx * 4, self.width, rounding_mode="floor")
        ).flatten()
        image_blocks = image_blocks.repeat(self.channels)
        latent_blocks = torch.tensor(
            [16, 17], device=self.observation.device, dtype=torch.long
        )
        return torch.cat((image_blocks.long(), latent_blocks))

    def factor_mask(self) -> Tensor:
        return torch.ones(
            self.batch,
            self.factor_count,
            device=self.observation.device,
            dtype=torch.bool,
        )

    def spatial_context(self) -> SpatialFieldContext:
        return SpatialFieldContext(
            observation=self.observation.flatten(1),
            field_shape=(self.channels, self.height, self.width),
            parameter_indices=torch.arange(
                self.image_parameters,
                device=self.observation.device,
                dtype=torch.long,
            ),
        )


__all__ = [
    "BlindNoiseConfig",
    "BlindNoiseFactorGraph",
    "sample_poisson_gaussian",
]
