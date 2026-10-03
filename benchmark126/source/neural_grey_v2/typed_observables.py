"""Metric-aware, dimensionless observables for selective DA-DFS pacing.

The functions in this module deliberately distinguish coordinate tangents
(updates) from coordinate covectors (gradients).  A Euclidean norm of a raw
gradient is not invariant to a change of parameter units; the corresponding
dual norm is.  Adapters may therefore expose a positive diagonal tangent
metric and use these primitives without exposing a task identifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _validate_same_shape(first: Tensor, second: Tensor, names: str) -> None:
    if first.ndim < 2 or second.shape != first.shape:
        raise ValueError(f"{names} must share a shape [B,...]")
    if not first.is_floating_point() or not second.is_floating_point():
        raise ValueError(f"{names} must be floating point")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError(f"{names} must be finite")


def _validate_metric(reference: Tensor, tangent_metric_diag: Tensor) -> None:
    if tangent_metric_diag.shape != reference.shape:
        raise ValueError("tangent metric must match the coordinate tensor")
    if (
        not tangent_metric_diag.is_floating_point()
        or not torch.isfinite(tangent_metric_diag).all()
        or torch.any(tangent_metric_diag <= 0.0)
    ):
        raise ValueError("tangent metric must be finite and strictly positive")


def metric_tangent_energy(tangent: Tensor, tangent_metric_diag: Tensor) -> Tensor:
    """Return mean squared tangent length under a diagonal primal metric."""

    _validate_same_shape(tangent, tangent_metric_diag, "tangent and metric")
    _validate_metric(tangent, tangent_metric_diag)
    return (tangent.square() * tangent_metric_diag).flatten(1).mean(dim=1)


def metric_dual_energy(covector: Tensor, tangent_metric_diag: Tensor) -> Tensor:
    """Return mean squared covector length under the corresponding dual metric."""

    _validate_same_shape(covector, tangent_metric_diag, "covector and metric")
    _validate_metric(covector, tangent_metric_diag)
    return (covector.square() / tangent_metric_diag).flatten(1).mean(dim=1)


def metric_tangent_rms(tangent: Tensor, tangent_metric_diag: Tensor) -> Tensor:
    return metric_tangent_energy(tangent, tangent_metric_diag).clamp_min(0.0).sqrt()


def metric_dual_rms(covector: Tensor, tangent_metric_diag: Tensor) -> Tensor:
    return metric_dual_energy(covector, tangent_metric_diag).clamp_min(0.0).sqrt()


def metric_dual_cosine(
    first_covector: Tensor,
    second_covector: Tensor,
    tangent_metric_diag: Tensor,
    *,
    eps: float = 1.0e-12,
) -> Tensor:
    """Cosine between two covectors in the dual of the declared metric."""

    _validate_same_shape(first_covector, second_covector, "covectors")
    _validate_metric(first_covector, tangent_metric_diag)
    inverse_metric = tangent_metric_diag.reciprocal()
    numerator = (
        first_covector * second_covector * inverse_metric
    ).flatten(1).sum(dim=1)
    first_norm = (
        first_covector.square() * inverse_metric
    ).flatten(1).sum(dim=1).sqrt()
    second_norm = (
        second_covector.square() * inverse_metric
    ).flatten(1).sum(dim=1).sqrt()
    denominator = (first_norm * second_norm).clamp_min(eps)
    return (numerator / denominator).clamp(-1.0, 1.0)


def metric_gradient_update_alignment(
    gradient_covector: Tensor,
    update_tangent: Tensor,
    tangent_metric_diag: Tensor,
    *,
    eps: float = 1.0e-12,
) -> Tensor:
    """Return alignment of an update with the metric steepest-descent direction.

    The numerator is the coordinate-invariant natural pairing ``-g(u)``.  A
    value near one means descent alignment, zero means orthogonality, and a
    negative value means ascent alignment.
    """

    _validate_same_shape(gradient_covector, update_tangent, "gradient and update")
    _validate_metric(gradient_covector, tangent_metric_diag)
    pairing = -(gradient_covector * update_tangent).flatten(1).sum(dim=1)
    gradient_norm = (
        gradient_covector.square() / tangent_metric_diag
    ).flatten(1).sum(dim=1).sqrt()
    update_norm = (
        update_tangent.square() * tangent_metric_diag
    ).flatten(1).sum(dim=1).sqrt()
    denominator = (gradient_norm * update_norm).clamp_min(eps)
    return (pairing / denominator).clamp(-1.0, 1.0)


def bounded_log_ratio(
    numerator: Tensor,
    denominator: Tensor,
    *,
    eps: float = 1.0e-12,
) -> Tensor:
    """Map a positive dimensionless ratio to ``[-1,1]``."""

    if numerator.shape != denominator.shape:
        raise ValueError("ratio terms must share a shape")
    if (
        not torch.isfinite(numerator).all()
        or not torch.isfinite(denominator).all()
        or torch.any(numerator < 0.0)
        or torch.any(denominator < 0.0)
    ):
        raise ValueError("ratio terms must be finite and non-negative")
    return torch.tanh(
        torch.log((numerator + eps) / (denominator + eps))
    )


@dataclass(frozen=True)
class NoiseResidualObservables:
    """Target-free checks for an Adapter declaring a stochastic noise model."""

    energy_mismatch: Tensor
    lag_correlation: Tensor

    def as_tensor(self) -> Tensor:
        return torch.stack((self.energy_mismatch, self.lag_correlation), dim=-1)


def noise_residual_observables(
    residual: Tensor,
    predicted_variance: Tensor,
    *,
    eps: float = 1.0e-8,
) -> NoiseResidualObservables:
    """Measure white-noise energy calibration and spatial lag correlation.

    ``predicted_variance`` may contain one channel and is broadcast over the
    residual channels.  Jointly scaling residual units by ``a`` and variance
    units by ``a**2`` leaves both observables unchanged.
    """

    if residual.ndim != 4 or residual.shape[1] < 1:
        raise ValueError("residual must have shape [B,C,H,W]")
    if min(residual.shape[-2:]) < 2:
        raise ValueError("residual must be at least 2x2")
    if not residual.is_floating_point() or not torch.isfinite(residual).all():
        raise ValueError("residual must be finite floating point")
    try:
        variance = torch.broadcast_to(predicted_variance, residual.shape)
    except RuntimeError as error:
        raise ValueError("predicted variance must broadcast to residual") from error
    if (
        not variance.is_floating_point()
        or not torch.isfinite(variance).all()
        or torch.any(variance <= 0.0)
    ):
        raise ValueError("predicted variance must be finite and strictly positive")

    whitened = residual / variance.sqrt()
    energy = whitened.square().mean(dim=(1, 2, 3))
    energy_mismatch = bounded_log_ratio(
        energy,
        torch.ones_like(energy),
        eps=eps,
    ).abs()

    centered = whitened - whitened.mean(dim=(2, 3), keepdim=True)
    correlations = []
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        if dx >= 0:
            first = centered[..., : centered.shape[-2] - dy, : centered.shape[-1] - dx or None]
            second = centered[..., dy:, dx:]
        else:
            first = centered[..., : centered.shape[-2] - dy, -dx:]
            second = centered[..., dy:, :dx]
        numerator = (first * second).mean(dim=(1, 2, 3))
        denominator = (
            first.square().mean(dim=(1, 2, 3))
            * second.square().mean(dim=(1, 2, 3))
        ).clamp_min(0.0).sqrt().clamp_min(eps)
        correlations.append((numerator / denominator).clamp(-1.0, 1.0).abs())
    lag_correlation = torch.stack(correlations, dim=-1).mean(dim=-1)
    return NoiseResidualObservables(
        energy_mismatch=energy_mismatch.clamp(0.0, 1.0),
        lag_correlation=lag_correlation.clamp(0.0, 1.0),
    )


__all__ = [
    "NoiseResidualObservables",
    "bounded_log_ratio",
    "metric_dual_cosine",
    "metric_dual_energy",
    "metric_dual_rms",
    "metric_gradient_update_alignment",
    "metric_tangent_energy",
    "metric_tangent_rms",
    "noise_residual_observables",
]
