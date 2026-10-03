"""Small differentiable spatial operators implemented directly in PyTorch.

These operators intentionally avoid restoration libraries and pretrained
models.  They provide auditable finite-difference and binomial-scale query
signals while preserving translation/coordinate equivariance.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def finite_difference_xy(image: Tensor) -> tuple[Tensor, Tensor]:
    """Forward horizontal/vertical differences with an exact-null boundary."""

    if image.ndim != 4:
        raise ValueError("image must have shape [B,C,H,W]")
    dx = F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def separable_binomial_blur(image: Tensor, passes: int = 1) -> Tensor:
    """Apply a hand-specified [1,4,6,4,1]/16 separable low-pass filter."""

    if image.ndim != 4:
        raise ValueError("image must have shape [B,C,H,W]")
    if passes < 1:
        raise ValueError("passes must be positive")
    channels = image.shape[1]
    taps = image.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    horizontal = taps.reshape(1, 1, 1, 5).expand(channels, 1, 1, 5)
    vertical = taps.reshape(1, 1, 5, 1).expand(channels, 1, 5, 1)
    output = image
    for _ in range(passes):
        output = F.conv2d(
            F.pad(output, (2, 2, 0, 0), mode="reflect"),
            horizontal,
            groups=channels,
        )
        output = F.conv2d(
            F.pad(output, (0, 0, 2, 2), mode="reflect"),
            vertical,
            groups=channels,
        )
    return output


def charbonnier(value: Tensor, epsilon: float = 1.0e-3) -> Tensor:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return torch.sqrt(value.square() + epsilon * epsilon) - epsilon


def diagonal_mad_noise_sigma(image: Tensor) -> Tensor:
    """Estimate per-image white-noise scale from a diagonal second difference.

    The stencil ``[[1,-1],[-1,1]]`` has an iid-noise gain of two.  Median
    absolute deviation makes the estimate resistant to sparse image edges;
    no clean target, task label, or learned parameter is used.
    """

    if image.ndim != 4:
        raise ValueError("image must have shape [B,C,H,W]")
    if min(image.shape[-2:]) < 2:
        raise ValueError("diagonal MAD requires spatial dimensions >= 2")
    diagonal = (
        image[..., 1:, 1:]
        - image[..., 1:, :-1]
        - image[..., :-1, 1:]
        + image[..., :-1, :-1]
    ).flatten(1)
    center = diagonal.median(dim=1).values[:, None]
    mad = (diagonal - center).abs().median(dim=1).values
    return mad / (2.0 * 0.6744897501960817)


def analytic_noise_risk_features(image: Tensor) -> Tensor:
    """Return task-free robust noise/texture statistics with shape ``[B,4]``.

    Columns are diagonal-MAD sigma, high-pass-MAD sigma, low-pass gradient
    median, and the bounded noise-to-structure ratio.  The high-pass sigma is
    corrected for the exact gain of the hand-specified binomial residual.
    """

    if image.ndim != 4:
        raise ValueError("image must have shape [B,C,H,W]")
    diagonal_sigma = diagonal_mad_noise_sigma(image)
    low_pass = separable_binomial_blur(image)
    high_pass = (image - low_pass).flatten(1)
    high_center = high_pass.median(dim=1).values[:, None]
    high_mad = (high_pass - high_center).abs().median(dim=1).values
    # ||delta - k_5x5||_2 for the separable [1,4,6,4,1]/16 kernel.
    high_pass_gain = 0.8907977430352227
    high_sigma = high_mad / (0.6744897501960817 * high_pass_gain)
    dx, dy = finite_difference_xy(low_pass)
    structure = torch.sqrt(dx.square() + dy.square() + 1.0e-12)
    structure_median = structure.flatten(1).median(dim=1).values
    noise_to_structure = diagonal_sigma / (
        diagonal_sigma + structure_median + 1.0e-8
    )
    return torch.stack(
        (diagonal_sigma, high_sigma, structure_median, noise_to_structure), dim=-1
    )


def symmetric_noise_estimator_agreement(image: Tensor) -> Tensor:
    """Return bounded agreement between two hand-derived noise estimates.

    White noise excites the diagonal second difference and the binomial
    residual at their analytically corrected gains.  Repeated image texture
    tends to contaminate the two estimates unequally.  The symmetric ratio is
    permutation-free, contains no learned parameter, and is exactly bounded
    in ``[0,1]``.
    """

    features = analytic_noise_risk_features(image)
    diagonal_sigma, highpass_sigma = features[:, 0], features[:, 1]
    smaller = torch.minimum(diagonal_sigma, highpass_sigma)
    larger = torch.maximum(diagonal_sigma, highpass_sigma)
    return smaller / (larger + 1.0e-8)


def smoothstep_risk_confidence(
    score: Tensor, *, low: float, high: float
) -> Tensor:
    """Monotone bounded analytic confidence with exact zero/one plateaus."""

    if not high > low >= 0.0:
        raise ValueError("risk thresholds must satisfy 0 <= low < high")
    normalized = ((score - low) / (high - low)).clamp(0.0, 1.0)
    return normalized.square() * (3.0 - 2.0 * normalized)


def multiscale_query_error(
    prediction: Tensor,
    target: Tensor,
    *,
    levels: int = 3,
    gradient_weight: float = 0.20,
    epsilon: float = 1.0e-3,
) -> Tensor:
    """Per-sample robust RGB/gradient error over hand-built image scales."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share [B,C,H,W]")
    if levels < 1:
        raise ValueError("levels must be positive")
    if gradient_weight < 0.0:
        raise ValueError("gradient_weight cannot be negative")
    current_prediction = prediction
    current_target = target
    total = prediction.new_zeros(prediction.shape[0])
    weight_sum = 0.0
    for level in range(levels):
        weight = 0.5**level
        residual = charbonnier(
            current_prediction - current_target, epsilon
        ).mean(dim=(1, 2, 3))
        pred_dx, pred_dy = finite_difference_xy(current_prediction)
        target_dx, target_dy = finite_difference_xy(current_target)
        gradient = 0.5 * (
            charbonnier(pred_dx - target_dx, epsilon).mean(dim=(1, 2, 3))
            + charbonnier(pred_dy - target_dy, epsilon).mean(dim=(1, 2, 3))
        )
        total = total + weight * (residual + gradient_weight * gradient)
        weight_sum += weight
        if level + 1 < levels:
            current_prediction = separable_binomial_blur(current_prediction)[
                ..., ::2, ::2
            ]
            current_target = separable_binomial_blur(current_target)[..., ::2, ::2]
    return total / weight_sum


def multiscale_update_energy(
    prediction: Tensor,
    reference: Tensor,
    *,
    levels: int = 2,
) -> Tensor:
    """Per-sample squared update energy with low-frequency coverage."""

    if prediction.shape != reference.shape or prediction.ndim != 4:
        raise ValueError("prediction and reference must share [B,C,H,W]")
    if levels < 1:
        raise ValueError("levels must be positive")
    current_prediction = prediction
    current_reference = reference
    total = prediction.new_zeros(prediction.shape[0])
    weight_sum = 0.0
    for level in range(levels):
        weight = 0.5**level
        total = total + weight * (current_prediction - current_reference).square().mean(
            dim=(1, 2, 3)
        )
        weight_sum += weight
        if level + 1 < levels:
            current_prediction = separable_binomial_blur(current_prediction)[
                ..., ::2, ::2
            ]
            current_reference = separable_binomial_blur(current_reference)[
                ..., ::2, ::2
            ]
    return total / weight_sum


__all__ = [
    "analytic_noise_risk_features",
    "charbonnier",
    "diagonal_mad_noise_sigma",
    "finite_difference_xy",
    "multiscale_query_error",
    "multiscale_update_energy",
    "separable_binomial_blur",
    "smoothstep_risk_confidence",
]
