"""Variable-size real HPatches multi-frame alignment factor graphs.

Each free block is an eight-coordinate homography from the fixed reference
frame to one target frame.  Dense photometric factors retain the established
pairwise Adapter, while SIFT matches between target frames create genuine
cross-block cycle factors.  Ground-truth homographies are intentionally absent
from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from torch import Tensor

from .hpatches_homography import (
    CompiledHomographyObservation,
    HOMOGRAPHY_BLOCK_ID,
    SIFTCompilerConfig,
    SIFTHomographyCompiler,
    homography_photometric_factor_graph,
    homography_from_theta,
    read_grayscale,
)


MultiframeSupportMode = Literal["photometric", "geometric", "hybrid"]


@dataclass(frozen=True)
class MultiframeCompileReport:
    sequence: str
    target_indices: tuple[int, ...]
    status: str
    support_mode: str
    symmetric_reprojection: bool
    reference_inliers: tuple[int, ...]
    cross_inliers: tuple[int, ...]
    parameter_count: int
    factor_count: int
    cross_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CompiledMultiframeGraph:
    factor_function: Callable[[Tensor], Tensor]
    initial_theta: Tensor
    block_id: Tensor
    decode: Callable[[Tensor], Tensor]
    reference_shape: tuple[int, int]
    target_indices: tuple[int, ...]
    report: MultiframeCompileReport


def _safe_project(points: Tensor, homography: Tensor) -> Tensor:
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
    projected = torch.matmul(homography, homogeneous.t()[None]).transpose(1, 2)
    denominator = projected[..., 2:3]
    safe = torch.where(
        denominator >= 0.0,
        denominator.clamp_min(1.0e-6),
        denominator.clamp_max(-1.0e-6),
    )
    return projected[..., :2] / safe


def _cross_edge_pairs(count: int) -> tuple[tuple[int, int], ...]:
    if count < 2:
        return ()
    edges = [(index, index + 1) for index in range(count - 1)]
    if count >= 3:
        edges.append((0, count - 1))
    return tuple(edges)


def _make_chunks(count: int, chunks: int, device: torch.device) -> tuple[Tensor, ...]:
    return tuple(
        index
        for index in torch.tensor_split(
            torch.arange(count, device=device),
            min(int(chunks), int(count)),
        )
        if index.numel() > 0
    )


def _chunked_reprojection_factors(
    source: Tensor,
    target: Tensor,
    homography: Tensor,
    chunks: tuple[Tensor, ...],
    *,
    residual_scale_px: float,
    symmetric: bool,
) -> Tensor:
    prediction = _safe_project(source, homography)
    residual = (prediction - target[None]) / float(residual_scale_px)
    robust = torch.sqrt(1.0 + residual.square().sum(dim=-1)) - 1.0
    factors = [
        torch.stack([robust[:, index].mean(dim=1) for index in chunks], dim=1)
    ]
    if symmetric:
        inverse_prediction = _safe_project(target, torch.linalg.inv(homography))
        inverse_residual = (inverse_prediction - source[None]) / float(
            residual_scale_px
        )
        inverse_robust = torch.sqrt(1.0 + inverse_residual.square().sum(dim=-1)) - 1.0
        factors.append(
            torch.stack(
                [inverse_robust[:, index].mean(dim=1) for index in chunks], dim=1
            )
        )
    return torch.cat(factors, dim=1)


def compile_hpatches_multiframe_graph(
    sequence_root: Path | str,
    target_indices: Sequence[int],
    *,
    device: torch.device | str,
    compiler: SIFTHomographyCompiler | None = None,
    dtype: torch.dtype = torch.float32,
    cross_factor_chunks: int = 16,
    residual_scale_px: float = 3.0,
    aggregate_photometric_levels: bool = True,
    support_mode: MultiframeSupportMode = "photometric",
    symmetric_reprojection: bool = True,
) -> CompiledMultiframeGraph:
    """Compile one variable-size multi-frame graph without opening GT files."""

    root = Path(sequence_root)
    targets = tuple(int(value) for value in target_indices)
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise ValueError("target_indices must contain at least two unique frames")
    if any(value < 2 or value > 6 for value in targets):
        raise ValueError("HPatches target indices must lie in [2, 6]")
    if cross_factor_chunks < 1 or residual_scale_px <= 0.0:
        raise ValueError("invalid cross-factor configuration")
    if support_mode not in ("photometric", "geometric", "hybrid"):
        raise ValueError("support_mode must be 'photometric', 'geometric', or 'hybrid'")
    compiler = compiler or SIFTHomographyCompiler(SIFTCompilerConfig())
    device = torch.device(device)

    reference = read_grayscale(root / "1.ppm")
    target_images = [read_grayscale(root / f"{index}.ppm") for index in targets]
    reference_shape = tuple(int(value) for value in reference.shape)
    target_shapes = [tuple(int(value) for value in image.shape) for image in target_images]
    reference_features = compiler.extract(reference)
    target_features = [compiler.extract(image) for image in target_images]
    observations: list[CompiledHomographyObservation] = [
        compiler.compile_features(reference_features, features)
        for features in target_features
    ]
    if not all(item.succeeded for item in observations):
        failed = [targets[i] for i, item in enumerate(observations) if not item.succeeded]
        raise RuntimeError(f"reference compilation failed for frames {failed}")

    edge_indices = _cross_edge_pairs(len(targets))
    cross_observations = [
        compiler.compile_features(target_features[left], target_features[right])
        for left, right in edge_indices
    ]
    if not all(item.succeeded for item in cross_observations):
        failed = [
            (targets[left], targets[right])
            for (left, right), item in zip(edge_indices, cross_observations)
            if not item.succeeded
        ]
        raise RuntimeError(f"cross-frame compilation failed for edges {failed}")

    decoders = []
    for observation, target_shape in zip(observations, target_shapes):
        initial_homography = torch.as_tensor(
            observation.initial_homography, device=device, dtype=dtype
        )

        def decode_single(
            theta: Tensor,
            *,
            initial_homography: Tensor = initial_homography,
            target_shape: tuple[int, int] = target_shape,
        ) -> Tensor:
            return homography_from_theta(
                theta, initial_homography, reference_shape, target_shape
            )

        decoders.append(decode_single)

    photometric_functions = []
    if support_mode in ("photometric", "hybrid"):
        for observation, target_image in zip(observations, target_images):
            function, _, _, _ = homography_photometric_factor_graph(
                observation,
                reference,
                target_image,
                device=device,
                dtype=dtype,
            )
            photometric_functions.append(function)

    reference_data = []
    if support_mode in ("geometric", "hybrid"):
        for index, observation in enumerate(observations):
            source = torch.as_tensor(
                observation.reference_points, device=device, dtype=dtype
            )
            target = torch.as_tensor(
                observation.target_points, device=device, dtype=dtype
            )
            reference_data.append(
                (index, source, target, _make_chunks(source.shape[0], cross_factor_chunks, device))
            )

    cross_data = []
    for (left, right), observation in zip(edge_indices, cross_observations):
        source = torch.as_tensor(
            observation.reference_points, device=device, dtype=dtype
        )
        target = torch.as_tensor(observation.target_points, device=device, dtype=dtype)
        chunks = _make_chunks(source.shape[0], cross_factor_chunks, device)
        cross_data.append((left, right, source, target, chunks))

    def decode(theta: Tensor) -> Tensor:
        if theta.ndim != 2 or theta.shape[1] != 8 * len(targets):
            raise ValueError("theta has the wrong multi-frame shape")
        return torch.stack(
            [
                decoder(theta[:, 8 * index : 8 * (index + 1)])
                for index, decoder in enumerate(decoders)
            ],
            dim=1,
        )

    def factor_function(theta: Tensor) -> Tensor:
        factors = []
        for index, function in enumerate(photometric_functions):
            values = function(theta[:, 8 * index : 8 * (index + 1)])
            # The established pair graph emits 3 pyramid levels x 21 spatial
            # cells.  Summing levels per cell preserves the exact scalar
            # objective and its gradient while avoiding a 3x token expansion
            # (roughly 9x dense attention work) in multi-frame graphs.
            if aggregate_photometric_levels:
                if values.shape[1] % 3 != 0:
                    raise RuntimeError("photometric factors are not a three-level sketch")
                values = values.reshape(values.shape[0], 3, -1).sum(dim=1)
            factors.append(values)
        homographies = decode(theta)
        for index, source, target, chunks in reference_data:
            factors.append(
                _chunked_reprojection_factors(
                    source,
                    target,
                    homographies[:, index],
                    chunks,
                    residual_scale_px=residual_scale_px,
                    symmetric=symmetric_reprojection,
                )
            )
        for left, right, source, target, chunks in cross_data:
            relative = homographies[:, right] @ torch.linalg.inv(homographies[:, left])
            factors.append(
                _chunked_reprojection_factors(
                    source,
                    target,
                    relative,
                    chunks,
                    residual_scale_px=residual_scale_px,
                    symmetric=symmetric_reprojection,
                )
            )
        values = torch.cat(factors, dim=1)
        if not torch.isfinite(values).all():
            raise RuntimeError("multi-frame factor graph produced non-finite values")
        return values

    block_id = torch.cat(
        [HOMOGRAPHY_BLOCK_ID + 3 * index for index in range(len(targets))]
    ).to(device)
    initial = torch.zeros(1, 8 * len(targets), device=device, dtype=dtype)
    with torch.no_grad():
        factor_count = int(factor_function(initial).shape[1])
    report = MultiframeCompileReport(
        sequence=root.name,
        target_indices=targets,
        status="ok",
        support_mode=support_mode,
        symmetric_reprojection=bool(symmetric_reprojection),
        reference_inliers=tuple(int(item.ransac_inliers) for item in observations),
        cross_inliers=tuple(int(item.ransac_inliers) for item in cross_observations),
        parameter_count=int(initial.shape[1]),
        factor_count=factor_count,
        cross_edges=tuple((targets[left], targets[right]) for left, right in edge_indices),
    )
    return CompiledMultiframeGraph(
        factor_function=factor_function,
        initial_theta=initial,
        block_id=block_id,
        decode=decode,
        reference_shape=reference.shape,
        target_indices=targets,
        report=report,
    )


__all__ = [
    "CompiledMultiframeGraph",
    "MultiframeSupportMode",
    "MultiframeCompileReport",
    "compile_hpatches_multiframe_graph",
]
