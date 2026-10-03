"""Task-ID-free blind low-light joint factor graph for DA-DFS M27."""

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
class BlindLowLightConfig:
    """Observable physics and regularizer constants for the M27 ceiling."""

    illumination_grid: tuple[int, int] = (6, 6)
    illumination_range: tuple[float, float] = (0.02, 1.0)
    initial_illumination: float = 0.95
    exposure_target: float = 0.55
    shot_range: tuple[float, float] = (5.0e-4, 8.0e-2)
    read_std_range: tuple[float, float] = (5.0e-4, 4.0e-2)
    initial_shot: float = 1.5e-2
    initial_read_std: float = 8.0e-3
    data_weight: float = 1.0
    noise_complexity_weight: float = 2.0e-3
    variance_match_weight: float = 1.0e-2
    exposure_weight: float = 8.0e-2
    reflectance_gradient_weight: float = 1.5e-3
    chroma_weight: float = 2.0e-2
    illumination_smoothness_weight: float = 2.0e-2
    illumination_anchor_weight: float = 2.0e-2
    range_weight: float = 2.0
    latent_prior_weight: float = 1.0e-4
    query_holdout_stride: int = 0
    query_probe_stride: int = 0
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        gh, gw = self.illumination_grid
        if gh < 2 or gw < 2:
            raise ValueError("illumination grid dimensions must be at least two")
        for name, bounds in (
            ("illumination_range", self.illumination_range),
            ("shot_range", self.shot_range),
            ("read_std_range", self.read_std_range),
        ):
            if len(bounds) != 2 or not 0.0 <= bounds[0] < bounds[1]:
                raise ValueError(f"{name} must contain increasing non-negative bounds")
        if not self.illumination_range[0] < self.initial_illumination < self.illumination_range[1]:
            raise ValueError("initial illumination must lie inside its range")
        if not self.shot_range[0] < self.initial_shot < self.shot_range[1]:
            raise ValueError("initial shot must lie inside its range")
        if not self.read_std_range[0] < self.initial_read_std < self.read_std_range[1]:
            raise ValueError("initial read std must lie inside its range")
        if not 0.0 < self.exposure_target < 1.0:
            raise ValueError("exposure target must lie in (0,1)")
        weights = (
            self.data_weight,
            self.noise_complexity_weight,
            self.variance_match_weight,
            self.exposure_weight,
            self.reflectance_gradient_weight,
            self.chroma_weight,
            self.illumination_smoothness_weight,
            self.illumination_anchor_weight,
            self.range_weight,
            self.latent_prior_weight,
        )
        if self.data_weight <= 0.0 or any(weight < 0.0 for weight in weights):
            raise ValueError("factor weights must be non-negative with positive data weight")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.query_holdout_stride not in (0,) and self.query_holdout_stride < 2:
            raise ValueError("query_holdout_stride must be zero or at least two")
        if self.query_probe_stride not in (0,) and self.query_probe_stride < 2:
            raise ValueError("query_probe_stride must be zero or at least two")


class BlindLowLightFactorGraph:
    """Joint RGB radiance, illumination-field, and sensor-noise graph.

    The paired normal-light image is deliberately absent.  It may be used by
    an outer evaluator as a query target, never by this inference graph.
    """

    def __init__(
        self, observation: Tensor, config: BlindLowLightConfig | None = None
    ) -> None:
        if observation.ndim != 4 or observation.shape[1] != 3:
            raise ValueError("observation must have shape [B,3,H,W]")
        if not observation.is_floating_point() or not torch.isfinite(observation).all():
            raise ValueError("observation must be finite floating point")
        if min(observation.shape[-2:]) < 8:
            raise ValueError("observation must be at least 8x8")
        self.observation = observation.detach()
        self.config = config or BlindLowLightConfig()
        self.batch, self.channels, self.height, self.width = observation.shape
        self.grid_height, self.grid_width = self.config.illumination_grid
        self.image_parameters = self.channels * self.height * self.width
        self.illumination_parameters = self.grid_height * self.grid_width
        self.latent_parameters = self.illumination_parameters + 2
        self.parameters = self.image_parameters + self.latent_parameters
        self.sketch = GridFactorSketchSpec(
            self.height, self.width, ((1, 1), (2, 2), (4, 4))
        )
        self.compact_sketch = GridFactorSketchSpec(
            self.height, self.width, ((1, 1), (2, 2))
        )
        observed_low = F.avg_pool2d(self.observation, 5, stride=1, padding=2)
        self.observed_high_energy = F.avg_pool2d(
            (self.observation - observed_low).square().mean(1, keepdim=True),
            3,
            stride=1,
            padding=1,
        )
        luma = self._luma(self.observation)
        smooth_luma = F.avg_pool2d(luma, 15, stride=1, padding=7)
        self.illumination_anchor = (
            smooth_luma / self.config.exposure_target
        ).clamp(*self.config.illumination_range)
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=self.observation.device),
            torch.arange(self.width, device=self.observation.device),
            indexing="ij",
        )
        query_stride = (
            self.config.query_probe_stride
            if self.config.query_probe_stride > 0
            else self.config.query_holdout_stride
        )
        if query_stride == 0:
            self.query_mask = torch.zeros_like(xx, dtype=self.observation.dtype)[
                None, None
            ]
        else:
            self.query_mask = (
                (xx + yy) % query_stride == 0
            )[None, None].to(self.observation.dtype)
        if self.config.query_holdout_stride == 0:
            self.support_mask = torch.ones_like(self.query_mask)
        else:
            support_query_mask = (
                (xx + yy) % self.config.query_holdout_stride == 0
            )[None, None].to(self.observation.dtype)
            # The support split remains tied to query_holdout_stride for M29
            # reproducibility.  query_probe_stride is observation-only and
            # cannot perturb the inherited primary factor trajectory.
            self.support_mask = 1.0 - support_query_mask
            self.support_mask = 1.0 - self.query_mask

    @property
    def factor_count(self) -> int:
        return (
            5 * self.sketch.factors
            + (2 * self.channels + 2) * self.compact_sketch.factors
            + 2
        )

    @property
    def query_factor_count(self) -> int:
        return 6

    @property
    def typed_observable_count(self) -> int:
        return 2

    @property
    def typed_observable_names(self) -> tuple[str, ...]:
        return (
            "residual_energy_mismatch",
            "residual_lag_correlation",
        )

    @staticmethod
    def _bounded(raw: Tensor, bounds: tuple[float, float]) -> Tensor:
        low, high = bounds
        return low + (high - low) * torch.sigmoid(raw)

    @staticmethod
    def _initial_raw(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        return _logit((value - low) / (high - low))

    @staticmethod
    def _luma(image: Tensor) -> Tensor:
        weights = image.new_tensor((0.299, 0.587, 0.114))
        return (image * weights[None, :, None, None]).sum(1, keepdim=True)

    def initial_theta(self) -> Tensor:
        illumination_raw = self.observation.new_full(
            (self.batch, self.illumination_parameters),
            self._initial_raw(
                self.config.initial_illumination,
                self.config.illumination_range,
            ),
        )
        noise_raw = self.observation.new_tensor(
            (
                self._initial_raw(self.config.initial_shot, self.config.shot_range),
                self._initial_raw(
                    self.config.initial_read_std, self.config.read_std_range
                ),
            )
        ).expand(self.batch, -1)
        return torch.cat(
            (self.observation.flatten(1), illumination_raw, noise_raw), dim=1
        )

    def decode(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if theta.ndim != 2 or theta.shape != (self.batch, self.parameters):
            raise ValueError("theta has the wrong blind-low-light layout")
        image = theta[:, : self.image_parameters].reshape(
            self.batch, self.channels, self.height, self.width
        )
        start = self.image_parameters
        stop = start + self.illumination_parameters
        raw_illumination = theta[:, start:stop].reshape(
            self.batch, 1, self.grid_height, self.grid_width
        )
        illumination_grid = self._bounded(
            raw_illumination, self.config.illumination_range
        )
        illumination = F.interpolate(
            illumination_grid,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        raw_noise = theta[:, stop:]
        shot = self._bounded(raw_noise[:, 0], self.config.shot_range)
        read_std = self._bounded(raw_noise[:, 1], self.config.read_std_range)
        return image, illumination, shot, read_std

    def render(self, theta: Tensor) -> Tensor:
        image, illumination, _, _ = self.decode(theta)
        return image.clamp(0.0, 1.0) * illumination

    def factor_values(self, theta: Tensor) -> Tensor:
        image, illumination, shot, read_std = self.decode(theta)
        config = self.config
        clipped_image = image.clamp(0.0, 1.0)
        rendered = clipped_image * illumination
        variance = (
            shot[:, None, None, None] * rendered
            + read_std[:, None, None, None].square()
        ).clamp_min(config.eps)
        residual = rendered - self.observation
        data_cost = (
            0.5 * config.data_weight * residual.square() / variance
        ).mean(1, keepdim=True)
        if config.query_holdout_stride > 0:
            support_fraction = self.support_mask.mean().clamp_min(config.eps)
            data_cost = data_cost * self.support_mask / support_fraction
        noise_complexity = config.noise_complexity_weight * torch.log1p(
            variance / config.eps
        ).mean(1, keepdim=True)
        variance_match = config.variance_match_weight * (
            torch.log(self.observed_high_energy + config.eps)
            - torch.log(variance.mean(1, keepdim=True) + config.eps)
        ).square()

        image_luma = self._luma(clipped_image)
        local_exposure = F.avg_pool2d(image_luma, 15, stride=1, padding=7)
        exposure_cost = config.exposure_weight * (
            local_exposure - config.exposure_target
        ).square()

        log_image = torch.log(clipped_image + 1.0e-3)
        log_observation = torch.log(self.observation + 1.0e-3)
        image_dx = F.pad(log_image[..., 1:] - log_image[..., :-1], (0, 1, 0, 0))
        image_dy = F.pad(log_image[..., 1:, :] - log_image[..., :-1, :], (0, 0, 0, 1))
        obs_dx = F.pad(
            log_observation[..., 1:] - log_observation[..., :-1], (0, 1, 0, 0)
        )
        obs_dy = F.pad(
            log_observation[..., 1:, :] - log_observation[..., :-1, :],
            (0, 0, 0, 1),
        )
        edge_weight = torch.exp(-4.0 * (obs_dx.abs() + obs_dy.abs()).mean(1, keepdim=True))
        reflectance_gradient = config.reflectance_gradient_weight * edge_weight * (
            image_dx.square() + image_dy.square() + config.eps
        ).sqrt().mean(1, keepdim=True)

        image_chroma = clipped_image / clipped_image.sum(1, keepdim=True).clamp_min(1.0e-3)
        observed_chroma = self.observation / self.observation.sum(
            1, keepdim=True
        ).clamp_min(1.0e-3)
        chroma_cost = config.chroma_weight * (image_chroma - observed_chroma).square()

        log_illumination = torch.log(illumination + config.eps)
        illum_dx = F.pad(
            log_illumination[..., 1:] - log_illumination[..., :-1],
            (0, 1, 0, 0),
        )
        illum_dy = F.pad(
            log_illumination[..., 1:, :] - log_illumination[..., :-1, :],
            (0, 0, 0, 1),
        )
        illumination_smoothness = config.illumination_smoothness_weight * (
            illum_dx.square() + illum_dy.square()
        )
        illumination_anchor = config.illumination_anchor_weight * (
            log_illumination - torch.log(self.illumination_anchor + config.eps)
        ).square()
        range_cost = config.range_weight * (
            torch.relu(-image).square() + torch.relu(image - 1.0).square()
        )

        raw_noise = theta[:, -2:]
        latent_prior = config.latent_prior_weight * raw_noise.square()
        factors = torch.cat(
            (
                hierarchical_grid_sketch(data_cost, self.sketch),
                hierarchical_grid_sketch(noise_complexity, self.sketch),
                hierarchical_grid_sketch(variance_match, self.sketch),
                hierarchical_grid_sketch(exposure_cost, self.sketch),
                hierarchical_grid_sketch(reflectance_gradient, self.sketch),
                hierarchical_grid_sketch(chroma_cost, self.compact_sketch),
                hierarchical_grid_sketch(
                    illumination_smoothness, self.compact_sketch
                ),
                hierarchical_grid_sketch(illumination_anchor, self.compact_sketch),
                hierarchical_grid_sketch(range_cost, self.compact_sketch),
                latent_prior,
            ),
            dim=1,
        )
        if factors.shape != (self.batch, self.factor_count):
            raise RuntimeError("blind-low-light factor count changed")
        if torch.any(factors < 0.0) or not torch.isfinite(factors).all():
            raise RuntimeError("blind-low-light factors must be finite and non-negative")
        return factors

    def query_factor_values(self, theta: Tensor) -> Tensor:
        """Inference-visible factors excluded from the primary data update.

        The fixed checkerboard residual tests physical transfer, while the
        remaining terms measure target-free no-harm geometry.  No paired
        normal-light image is stored or read by this method.
        """

        image, illumination, shot, read_std = self.decode(theta)
        config = self.config
        clipped_image = image.clamp(0.0, 1.0)
        rendered = clipped_image * illumination
        variance = (
            shot[:, None, None, None] * rendered
            + read_std[:, None, None, None].square()
        ).clamp_min(config.eps)
        query_mask = (
            self.query_mask
            if config.query_holdout_stride > 0 or config.query_probe_stride > 0
            else torch.ones_like(self.query_mask)
        )
        query_count = query_mask.sum().clamp_min(1.0)

        def masked_mean(value: Tensor) -> Tensor:
            return (value * query_mask).sum(dim=(1, 2, 3)) / (
                query_count * value.shape[1]
            )

        residual = rendered - self.observation
        heldout_data = masked_mean(0.5 * residual.square() / variance)
        render_dx = F.pad(rendered[..., 1:] - rendered[..., :-1], (0, 1, 0, 0))
        render_dy = F.pad(rendered[..., 1:, :] - rendered[..., :-1, :], (0, 0, 0, 1))
        observed_dx = F.pad(
            self.observation[..., 1:] - self.observation[..., :-1],
            (0, 1, 0, 0),
        )
        observed_dy = F.pad(
            self.observation[..., 1:, :] - self.observation[..., :-1, :],
            (0, 0, 0, 1),
        )
        heldout_gradient = 0.5 * (
            masked_mean((render_dx - observed_dx).square())
            + masked_mean((render_dy - observed_dy).square())
        )

        observation_luma = self._luma(self.observation)
        brightness_confidence = (
            observation_luma.mean(dim=(1, 2, 3)) / config.exposure_target
        ).clamp(0.0, 1.0).square()
        identity_update = brightness_confidence * (
            clipped_image - self.observation
        ).square().mean(dim=(1, 2, 3))

        observed_low = F.avg_pool2d(self.observation, 5, stride=1, padding=2)
        image_low = F.avg_pool2d(clipped_image, 5, stride=1, padding=2)
        observed_detail = (
            self.observation - observed_low
        ).square().mean(dim=(1, 2, 3)).sqrt()
        image_detail = (
            clipped_image - image_low
        ).square().mean(dim=(1, 2, 3)).sqrt()
        detail_drift = torch.log(
            (image_detail + config.eps) / (observed_detail + config.eps)
        ).square()

        image_chroma = clipped_image / clipped_image.sum(
            1, keepdim=True
        ).clamp_min(1.0e-3)
        observed_chroma = self.observation / self.observation.sum(
            1, keepdim=True
        ).clamp_min(1.0e-3)
        chroma_drift = (image_chroma - observed_chroma).square().mean(
            dim=(1, 2, 3)
        )
        range_and_saturation = (
            torch.relu(-image).square().mean(dim=(1, 2, 3))
            + torch.relu(image - 1.0).square().mean(dim=(1, 2, 3))
            + (
                (clipped_image >= 0.995).to(image.dtype).mean(dim=(1, 2, 3))
                - (self.observation >= 0.995)
                .to(image.dtype)
                .mean(dim=(1, 2, 3))
            ).clamp_min(0.0).square()
        )
        factors = torch.stack(
            (
                heldout_data,
                heldout_gradient,
                identity_update,
                detail_drift,
                chroma_drift,
                range_and_saturation,
            ),
            dim=-1,
        )
        if factors.shape != (self.batch, self.query_factor_count):
            raise RuntimeError("blind-low-light query factor count changed")
        if torch.any(factors < 0.0) or not torch.isfinite(factors).all():
            raise RuntimeError("blind-low-light query factors must be finite and non-negative")
        return factors

    def coordinate_metric_diag(self, theta: Tensor) -> Tensor:
        """Diagonal tangent metric in normalized physical latent units."""

        self.decode(theta)
        raw_latent = theta[:, self.image_parameters :]
        probability = torch.sigmoid(raw_latent)
        normalized_derivative = probability * (1.0 - probability)
        latent_metric = normalized_derivative.square().clamp_min(
            self.config.eps**2
        )
        image_metric = theta.new_ones(self.batch, self.image_parameters)
        metric = torch.cat((image_metric, latent_metric), dim=1)
        if metric.shape != theta.shape or torch.any(metric <= 0.0):
            raise RuntimeError("blind-low-light coordinate metric changed")
        return metric

    def typed_observable_values(self, theta: Tensor) -> Tensor:
        """Return target-free residual calibration/structure observables."""

        image, illumination, shot, read_std = self.decode(theta)
        rendered = image.clamp(0.0, 1.0) * illumination
        variance = (
            shot[:, None, None, None] * rendered
            + read_std[:, None, None, None].square()
        ).clamp_min(self.config.eps)
        values = noise_residual_observables(
            rendered - self.observation,
            variance,
            eps=self.config.eps,
        ).as_tensor()
        if values.shape != (self.batch, self.typed_observable_count):
            raise RuntimeError("blind-low-light typed observable count changed")
        return values

    def block_id(self) -> Tensor:
        y = torch.arange(self.height, device=self.observation.device)
        x = torch.arange(self.width, device=self.observation.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        image_blocks = (
            torch.div(yy * 4, self.height, rounding_mode="floor") * 4
            + torch.div(xx * 4, self.width, rounding_mode="floor")
        ).flatten().repeat(self.channels)
        gy = torch.arange(self.grid_height, device=self.observation.device)
        gx = torch.arange(self.grid_width, device=self.observation.device)
        gyy, gxx = torch.meshgrid(gy, gx, indexing="ij")
        illumination_blocks = (
            16
            + torch.div(gyy * 4, self.grid_height, rounding_mode="floor") * 4
            + torch.div(gxx * 4, self.grid_width, rounding_mode="floor")
        ).flatten()
        noise_blocks = torch.tensor(
            (32, 33), device=self.observation.device, dtype=torch.long
        )
        return torch.cat((image_blocks.long(), illumination_blocks.long(), noise_blocks))

    def factor_mask(self) -> Tensor:
        return torch.ones(
            self.batch,
            self.factor_count,
            dtype=torch.bool,
            device=self.observation.device,
        )

    def spatial_context(self) -> SpatialFieldContext:
        return SpatialFieldContext(
            observation=self.observation.flatten(1),
            field_shape=(self.channels, self.height, self.width),
            parameter_indices=torch.arange(
                self.image_parameters,
                dtype=torch.long,
                device=self.observation.device,
            ),
        )


__all__ = ["BlindLowLightConfig", "BlindLowLightFactorGraph"]
