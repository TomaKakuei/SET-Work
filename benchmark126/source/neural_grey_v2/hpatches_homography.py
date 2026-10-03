"""HPatches homography compiler for task-ID-free DA-DFS evaluation.

The learned optimizer only receives non-negative factor values and AutoDiff
directions.  SIFT and RANSAC are a fixed, non-learned observation compiler;
ground-truth homographies are used exclusively by the metric helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .factor_sketch import GridFactorSketchSpec, hierarchical_grid_sketch


HPATCHES_HIGH_RESOLUTION_EXCLUSIONS = frozenset(
    {
        "i_contruction",
        "i_crownnight",
        "i_dc",
        "i_pencils",
        "i_whitebuilding",
        "v_artisans",
        "v_astronautis",
        "v_talent",
    }
)

HOMOGRAPHY_BLOCK_ID = torch.tensor([0, 0, 1, 0, 0, 1, 2, 2], dtype=torch.long)
HOMOGRAPHY_PARAMETER_SCALES = torch.tensor(
    [0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.01, 0.01],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class HPatchesPair:
    sequence: str
    kind: str
    target_index: int
    reference_path: Path
    target_path: Path
    ground_truth_path: Path


@dataclass(frozen=True)
class SIFTCompilerConfig:
    maximum_features: int = 4096
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6
    ransac_threshold_px: float = 3.0
    ransac_max_iterations: int = 2000
    ransac_confidence: float = 0.995
    ransac_seed: int = 0
    maximum_refinement_matches: int = 256


@dataclass(frozen=True)
class CompiledHomographyObservation:
    status: str
    initial_homography: np.ndarray | None
    reference_points: np.ndarray
    target_points: np.ndarray
    detected_reference: int
    detected_target: int
    mutual_matches: int
    ransac_inliers: int

    @property
    def succeeded(self) -> bool:
        return self.initial_homography is not None and self.status == "ok"


def list_hpatches_pairs(
    root: Path | str,
    *,
    formal: bool,
) -> list[HPatchesPair]:
    """Return the frozen formal or excluded-development pair list."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"HPatches root does not exist: {root}")
    pairs: list[HPatchesPair] = []
    for sequence_path in sorted(path for path in root.iterdir() if path.is_dir()):
        sequence = sequence_path.name
        if not (sequence.startswith("i_") or sequence.startswith("v_")):
            continue
        excluded = sequence in HPATCHES_HIGH_RESOLUTION_EXCLUSIONS
        if formal == excluded:
            continue
        reference = sequence_path / "1.ppm"
        if not reference.is_file():
            raise FileNotFoundError(f"missing HPatches reference: {reference}")
        for target_index in range(2, 7):
            target = sequence_path / f"{target_index}.ppm"
            ground_truth = sequence_path / f"H_1_{target_index}"
            if not target.is_file() or not ground_truth.is_file():
                raise FileNotFoundError(
                    f"incomplete HPatches pair {sequence}:1-{target_index}"
                )
            pairs.append(
                HPatchesPair(
                    sequence=sequence,
                    kind="illumination" if sequence.startswith("i_") else "viewpoint",
                    target_index=target_index,
                    reference_path=reference,
                    target_path=target,
                    ground_truth_path=ground_truth,
                )
            )
    expected = 540 if formal else 40
    if len(pairs) != expected:
        raise RuntimeError(
            f"expected {expected} {'formal' if formal else 'development'} pairs, "
            f"found {len(pairs)}"
        )
    return pairs


def read_grayscale(path: Path | str) -> np.ndarray:
    # Read bytes first so OpenCV also accepts non-ASCII checkout directories
    # on Windows. The decoder and grayscale pixels are otherwise unchanged.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def read_homography(path: Path | str) -> np.ndarray:
    homography = np.loadtxt(path, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError(f"invalid homography: {path}")
    if abs(float(homography[2, 2])) < 1.0e-12:
        raise ValueError(f"homography has a zero projective scale: {path}")
    return homography / homography[2, 2]


def _rootsift(descriptors: np.ndarray) -> np.ndarray:
    descriptors = descriptors.astype(np.float32, copy=False)
    denominator = np.sum(np.abs(descriptors), axis=1, keepdims=True)
    return np.sqrt(descriptors / np.maximum(denominator, 1.0e-12))


class SIFTHomographyCompiler:
    """Compile an image pair into a deterministic RANSAC/inlier observation."""

    def __init__(self, config: SIFTCompilerConfig = SIFTCompilerConfig()) -> None:
        if config.maximum_features < 4:
            raise ValueError("maximum_features must be at least four")
        if config.maximum_refinement_matches < 4:
            raise ValueError("maximum_refinement_matches must be at least four")
        self.config = config
        self.detector = cv2.SIFT_create(
            nfeatures=config.maximum_features,
            contrastThreshold=config.contrast_threshold,
            edgeThreshold=config.edge_threshold,
            sigma=config.sigma,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim != 2:
            raise ValueError("SIFT compiler expects a grayscale image")
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        if descriptors is None or len(keypoints) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 128), dtype=np.float32),
            )
        if len(keypoints) > self.config.maximum_features:
            # OpenCV may retain one or more tied responses beyond `nfeatures`.
            # Enforce the protocol cap with a complete deterministic ordering.
            order = sorted(
                range(len(keypoints)),
                key=lambda index: (
                    -float(keypoints[index].response),
                    int(keypoints[index].octave),
                    float(keypoints[index].pt[0]),
                    float(keypoints[index].pt[1]),
                    float(keypoints[index].size),
                    float(keypoints[index].angle),
                ),
            )[: self.config.maximum_features]
            keypoints = [keypoints[index] for index in order]
            descriptors = descriptors[np.asarray(order)]
        points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
        return points, _rootsift(descriptors)

    @staticmethod
    def _failure(
        status: str,
        detected_reference: int,
        detected_target: int,
        mutual_matches: int,
    ) -> CompiledHomographyObservation:
        empty = np.empty((0, 2), dtype=np.float32)
        return CompiledHomographyObservation(
            status=status,
            initial_homography=None,
            reference_points=empty,
            target_points=empty,
            detected_reference=detected_reference,
            detected_target=detected_target,
            mutual_matches=mutual_matches,
            ransac_inliers=0,
        )

    def compile_features(
        self,
        reference_features: tuple[np.ndarray, np.ndarray],
        target_features: tuple[np.ndarray, np.ndarray],
    ) -> CompiledHomographyObservation:
        reference_points, reference_descriptors = reference_features
        target_points, target_descriptors = target_features
        detected_reference = int(reference_points.shape[0])
        detected_target = int(target_points.shape[0])
        if detected_reference < 4 or detected_target < 4:
            return self._failure(
                "insufficient_features", detected_reference, detected_target, 0
            )
        matches = sorted(
            self.matcher.match(reference_descriptors, target_descriptors),
            key=lambda match: (float(match.distance), match.queryIdx, match.trainIdx),
        )
        if len(matches) < 4:
            return self._failure(
                "insufficient_matches",
                detected_reference,
                detected_target,
                len(matches),
            )
        source = np.asarray(
            [reference_points[match.queryIdx] for match in matches], dtype=np.float32
        )
        target = np.asarray(
            [target_points[match.trainIdx] for match in matches], dtype=np.float32
        )
        cv2.setRNGSeed(self.config.ransac_seed)
        homography, inlier_mask = cv2.findHomography(
            source,
            target,
            cv2.RANSAC,
            self.config.ransac_threshold_px,
            maxIters=self.config.ransac_max_iterations,
            confidence=self.config.ransac_confidence,
        )
        if (
            homography is None
            or inlier_mask is None
            or homography.shape != (3, 3)
            or not np.isfinite(homography).all()
            or abs(float(homography[2, 2])) < 1.0e-12
        ):
            return self._failure(
                "ransac_failed",
                detected_reference,
                detected_target,
                len(matches),
            )
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_indices = np.flatnonzero(inliers)[
            : self.config.maximum_refinement_matches
        ]
        if inlier_indices.size < 4:
            return self._failure(
                "insufficient_inliers",
                detected_reference,
                detected_target,
                len(matches),
            )
        homography = homography.astype(np.float64)
        homography = homography / homography[2, 2]
        return CompiledHomographyObservation(
            status="ok",
            initial_homography=homography,
            reference_points=source[inlier_indices],
            target_points=target[inlier_indices],
            detected_reference=detected_reference,
            detected_target=detected_target,
            mutual_matches=len(matches),
            ransac_inliers=int(inliers.sum()),
        )

    def compile(
        self, reference: np.ndarray, target: np.ndarray
    ) -> CompiledHomographyObservation:
        return self.compile_features(self.extract(reference), self.extract(target))


def _pixel_normalization(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if height < 2 or width < 2:
        raise ValueError("images must have height and width of at least two")
    return torch.tensor(
        [
            [2.0 / (width - 1), 0.0, -1.0],
            [0.0, 2.0 / (height - 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _safe_project(points: Tensor, homography: Tensor) -> Tensor:
    """Project shared `[N,2]` points through batched `[B,3,3]` matrices."""

    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
    projected = torch.matmul(
        homography, homogeneous.transpose(0, 1)[None]
    ).transpose(1, 2)
    denominator = projected[..., 2:3]
    safe_denominator = torch.where(
        denominator >= 0.0,
        denominator.clamp_min(1.0e-6),
        denominator.clamp_max(-1.0e-6),
    )
    return projected[..., :2] / safe_denominator


def homography_from_theta(
    theta: Tensor,
    initial_homography: Tensor | np.ndarray,
    reference_shape: tuple[int, int],
    target_shape: tuple[int, int],
    *,
    parameter_scales: Tensor = HOMOGRAPHY_PARAMETER_SCALES,
) -> Tensor:
    """Decode an eight-coordinate normalized delta around an initializer."""

    if theta.ndim != 2 or theta.shape[1] != 8:
        raise ValueError("theta must have shape [B,8]")
    batch = theta.shape[0]
    scales = parameter_scales.to(device=theta.device, dtype=theta.dtype)
    if scales.shape != (8,):
        raise ValueError("parameter_scales must have shape [8]")
    values = theta * scales[None]
    delta = torch.eye(3, device=theta.device, dtype=theta.dtype)[None].repeat(
        batch, 1, 1
    )
    delta[:, 0, 0] = delta[:, 0, 0] + values[:, 0]
    delta[:, 0, 1] = values[:, 1]
    delta[:, 0, 2] = values[:, 2]
    delta[:, 1, 0] = values[:, 3]
    delta[:, 1, 1] = delta[:, 1, 1] + values[:, 4]
    delta[:, 1, 2] = values[:, 5]
    delta[:, 2, 0] = values[:, 6]
    delta[:, 2, 1] = values[:, 7]

    initial = torch.as_tensor(
        initial_homography, device=theta.device, dtype=theta.dtype
    )
    if initial.shape == (3, 3):
        initial = initial[None].expand(batch, -1, -1)
    if initial.shape != (batch, 3, 3):
        raise ValueError("initial_homography must have shape [3,3] or [B,3,3]")
    reference_normalization = _pixel_normalization(
        *reference_shape, device=theta.device, dtype=theta.dtype
    )
    target_normalization = _pixel_normalization(
        *target_shape, device=theta.device, dtype=theta.dtype
    )
    normalized_initial = (
        target_normalization[None]
        @ initial
        @ torch.linalg.inv(reference_normalization)[None]
    )
    normalized_output = delta @ normalized_initial
    output = (
        torch.linalg.inv(target_normalization)[None]
        @ normalized_output
        @ reference_normalization[None]
    )
    return output / output[:, 2:3, 2:3].clamp_min(1.0e-8)


def homography_factor_graph(
    observation: CompiledHomographyObservation,
    reference_shape: tuple[int, int],
    target_shape: tuple[int, int],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    residual_scale_px: float = 3.0,
) -> tuple[Callable[[Tensor], Tensor], Tensor, Tensor, Callable[[Tensor], Tensor]]:
    """Build the fixed inlier reprojection graph and homography decoder."""

    if not observation.succeeded or observation.initial_homography is None:
        raise ValueError("cannot build a graph from a failed observation")
    if residual_scale_px <= 0.0:
        raise ValueError("residual_scale_px must be positive")
    device = torch.device(device)
    source = torch.as_tensor(
        observation.reference_points, device=device, dtype=dtype
    )
    target = torch.as_tensor(observation.target_points, device=device, dtype=dtype)
    initial_homography = torch.as_tensor(
        observation.initial_homography, device=device, dtype=dtype
    )

    def decode(theta: Tensor) -> Tensor:
        return homography_from_theta(
            theta, initial_homography, reference_shape, target_shape
        )

    def factor_function(theta: Tensor) -> Tensor:
        prediction = _safe_project(source, decode(theta))
        residual = (prediction - target[None]) / residual_scale_px
        return 0.5 * residual.square().flatten(1)

    initial_theta = torch.zeros(1, 8, device=device, dtype=dtype)
    return (
        factor_function,
        initial_theta,
        HOMOGRAPHY_BLOCK_ID.to(device),
        decode,
    )


def _as_image_tensor(
    image: np.ndarray | Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    tensor = torch.as_tensor(image, device=device, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[None]
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError("photometric images must be grayscale [H,W] or [1,1,H,W]")
    if float(tensor.detach().max()) > 1.5:
        tensor = tensor / 255.0
    return tensor


def _local_contrast_normalize(image: Tensor, window: int = 15) -> Tensor:
    mean = F.avg_pool2d(image, window, stride=1, padding=window // 2)
    second = F.avg_pool2d(image.square(), window, stride=1, padding=window // 2)
    standard_deviation = (second - mean.square()).clamp_min(1.0e-4).sqrt()
    return ((image - mean) / standard_deviation).clamp(-3.0, 3.0) / 3.0


def homography_photometric_factor_graph(
    observation: CompiledHomographyObservation,
    reference_image: np.ndarray | Tensor,
    target_image: np.ndarray | Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    sample_side: int = 24,
    blur_kernels: tuple[int, ...] = (15, 7, 1),
    level_weights: tuple[float, ...] = (0.75, 0.20, 0.05),
) -> tuple[Callable[[Tensor], Tensor], Tensor, Tensor, Callable[[Tensor], Tensor]]:
    """Compile illumination-normalized dense alignment into 63 factors.

    Its topology mirrors the natural-alignment graph used to train M3, while
    the unknown is a full eight-coordinate homography delta.  The RANSAC
    initializer is used only to define the local parameterization and a frozen
    valid sampling mask.
    """

    if not observation.succeeded or observation.initial_homography is None:
        raise ValueError("cannot build a graph from a failed observation")
    if sample_side < 4:
        raise ValueError("sample_side must be at least four")
    if (
        len(blur_kernels) != len(level_weights)
        or any(kernel < 1 or kernel % 2 == 0 for kernel in blur_kernels)
        or any(weight <= 0.0 for weight in level_weights)
        or not np.isclose(sum(level_weights), 1.0)
    ):
        raise ValueError("invalid photometric pyramid")
    device = torch.device(device)
    reference = _local_contrast_normalize(
        _as_image_tensor(reference_image, device=device, dtype=dtype)
    )
    target = _local_contrast_normalize(
        _as_image_tensor(target_image, device=device, dtype=dtype)
    )
    reference_shape = (int(reference.shape[-2]), int(reference.shape[-1]))
    target_shape = (int(target.shape[-2]), int(target.shape[-1]))
    initial_homography = torch.as_tensor(
        observation.initial_homography, device=device, dtype=dtype
    )
    axis = torch.linspace(0.08, 0.92, sample_side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    source = torch.stack(
        (
            xx.flatten() * (reference_shape[1] - 1),
            yy.flatten() * (reference_shape[0] - 1),
        ),
        dim=-1,
    )
    reference_grid = torch.stack((2.0 * xx - 1.0, 2.0 * yy - 1.0), dim=-1)[None]
    spec = GridFactorSketchSpec(
        sample_side, sample_side, ((1, 1), (2, 2), (4, 4))
    )
    reference_pyramid = tuple(
        reference
        if kernel == 1
        else F.avg_pool2d(reference, kernel, stride=1, padding=kernel // 2)
        for kernel in blur_kernels
    )
    target_pyramid = tuple(
        target
        if kernel == 1
        else F.avg_pool2d(target, kernel, stride=1, padding=kernel // 2)
        for kernel in blur_kernels
    )
    reference_observations = tuple(
        F.grid_sample(
            level,
            reference_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).detach()
        for level in reference_pyramid
    )

    def decode(theta: Tensor) -> Tensor:
        return homography_from_theta(
            theta, initial_homography, reference_shape, target_shape
        )

    with torch.no_grad():
        initial_projection = _safe_project(
            source, decode(torch.zeros(1, 8, device=device, dtype=dtype))
        )
        initial_grid = torch.stack(
            (
                2.0 * initial_projection[..., 0] / (target_shape[1] - 1) - 1.0,
                2.0 * initial_projection[..., 1] / (target_shape[0] - 1) - 1.0,
            ),
            dim=-1,
        )
        valid = (
            (initial_grid[..., 0].abs() <= 0.98)
            & (initial_grid[..., 1].abs() <= 0.98)
        ).reshape(1, 1, sample_side, sample_side)
        valid_float = valid.to(dtype)

    def factor_function(theta: Tensor) -> Tensor:
        projection = _safe_project(source, decode(theta))
        target_grid = torch.stack(
            (
                2.0 * projection[..., 0] / (target_shape[1] - 1) - 1.0,
                2.0 * projection[..., 1] / (target_shape[0] - 1) - 1.0,
            ),
            dim=-1,
        ).reshape(theta.shape[0], sample_side, sample_side, 2)
        level_factors = []
        for level, reference_observation, weight in zip(
            target_pyramid, reference_observations, level_weights
        ):
            sampled_target = F.grid_sample(
                level.expand(theta.shape[0], -1, -1, -1),
                target_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            residual_cost = (
                0.5
                * (sampled_target - reference_observation).square()
                * valid_float
            )
            level_factors.append(
                float(weight) * hierarchical_grid_sketch(residual_cost, spec)
            )
        return torch.cat(level_factors, dim=-1)

    initial_theta = torch.zeros(1, 8, device=device, dtype=dtype)
    return (
        factor_function,
        initial_theta,
        HOMOGRAPHY_BLOCK_ID.to(device),
        decode,
    )


def warp_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homography = np.asarray(homography, dtype=np.float64)
    homogeneous = np.concatenate(
        (points, np.ones((points.shape[0], 1), dtype=np.float64)), axis=1
    )
    projected = (homography @ homogeneous.T).T
    denominator = projected[:, 2:3]
    if np.any(np.abs(denominator) < 1.0e-12):
        return np.full_like(points, np.inf)
    return projected[:, :2] / denominator


def homography_corner_error(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    reference_shape: tuple[int, int],
) -> float:
    height, width = reference_shape
    corners = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]],
        dtype=np.float64,
    )
    predicted = warp_points(corners, estimate)
    truth = warp_points(corners, ground_truth)
    error = np.linalg.norm(predicted - truth, axis=1).mean()
    return float(error) if np.isfinite(error) else float("inf")


def summarize_mha(
    rows: Iterable[dict[str, object]],
    method: str,
    thresholds: tuple[int, ...] = (3, 5, 7),
) -> dict[str, float | int]:
    selected = list(rows)
    errors = np.asarray(
        [float(row[f"{method}_corner_error_px"]) for row in selected],
        dtype=np.float64,
    )
    if errors.size == 0:
        raise ValueError("cannot summarize an empty row collection")
    finite = np.isfinite(errors)
    summary: dict[str, float | int] = {
        "pairs": int(errors.size),
        "failures": int((~finite).sum()),
        "mean_corner_error_px": float(errors[finite].mean()) if finite.any() else float("inf"),
        "median_corner_error_px": float(np.median(errors[finite])) if finite.any() else float("inf"),
    }
    for threshold in thresholds:
        summary[f"mha_at_{threshold}px"] = float(np.mean(errors <= threshold))
    return summary


__all__ = [
    "CompiledHomographyObservation",
    "HOMOGRAPHY_BLOCK_ID",
    "HOMOGRAPHY_PARAMETER_SCALES",
    "HPATCHES_HIGH_RESOLUTION_EXCLUSIONS",
    "HPatchesPair",
    "SIFTCompilerConfig",
    "SIFTHomographyCompiler",
    "homography_corner_error",
    "homography_factor_graph",
    "homography_photometric_factor_graph",
    "homography_from_theta",
    "list_hpatches_pairs",
    "read_grayscale",
    "read_homography",
    "summarize_mha",
    "warp_points",
]
