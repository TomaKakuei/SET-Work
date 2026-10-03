"""Hand-written variable-size SE(3) RGB-D pose-graph Adapter.

The graph stores raw 3D--3D correspondences on every edge.  Node states are
camera-to-anchor poses; the first camera is fixed to remove gauge freedom.
Neither edge compilation nor initialization reads a trajectory target.  A
deterministic robust-SVD transform on chain edges supplies the odometry
initializer, while every optimization method sees the same raw edge evidence.

The learned interface uses one scalar robust energy per edge.  A separate
per-correspondence energy view is exposed for a stronger classical LM control;
these have the same summed objective but different Gauss--Newton models.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from torch import Tensor

from .tum_rgbd_se3 import ROTATION_SCALE_RAD, TRANSLATION_SCALE_M, so3_exp


FactorGranularity = Literal["node", "edge", "correspondence"]
FactorView = Literal["even", "odd", "all"]


def _so3_log(rotation: Tensor) -> Tensor:
    """Stable logarithm for rotations away from the pi singularity."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in [3,3]")
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    half_vee = 0.5 * vee
    sine = torch.linalg.vector_norm(half_vee, dim=-1)
    cosine = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
    cosine = cosine.clamp(-1.0, 1.0)
    angle = torch.atan2(sine, cosine)
    exact = angle / sine.clamp_min(1.0e-7)
    angle2 = angle.square()
    series = 1.0 + angle2 / 6.0 + 7.0 * angle2.square() / 360.0
    scale = torch.where(sine < 1.0e-4, series, exact)
    return scale[..., None] * half_vee


def _rotation_matrix_to_vector_numpy(rotation: np.ndarray) -> np.ndarray:
    value = torch.as_tensor(rotation, dtype=torch.float64)
    return _so3_log(value).cpu().numpy()


def _pseudo_huber_norm(error: Tensor, delta: float) -> Tensor:
    norm2 = error.square().sum(dim=-1)
    scaled = norm2 / float(delta * delta)
    return delta * delta * (torch.sqrt(1.0 + scaled) - 1.0)


def _weighted_rigid_transform(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 1.0e-12:
        raise ValueError("rigid-transform weights are degenerate")
    normalized = weights / total
    source_center = (normalized[:, None] * source).sum(axis=0)
    target_center = (normalized[:, None] * target).sum(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ (normalized[:, None] * target_zero)
    left, _, right_transpose = np.linalg.svd(covariance, full_matrices=False)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transpose = right_transpose.copy()
        right_transpose[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation.astype(np.float64), translation.astype(np.float64)


def robust_rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    rounds: int = 6,
    minimum_scale_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Deterministic IRLS rigid transform without RANSAC or external solvers."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have shape [matches,3]")
    if source.shape[0] < 6 or rounds < 1:
        raise ValueError("at least six correspondences and one round are required")
    weights = np.ones(source.shape[0], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for _ in range(rounds):
        rotation, translation = _weighted_rigid_transform(source, target, weights)
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = max(minimum_scale_m, median + 2.5 * 1.4826 * mad)
        weights = np.minimum(1.0, scale / np.maximum(residual, 1.0e-12))
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return rotation, translation, {
        "median_residual_m": float(np.median(residual)),
        "p90_residual_m": float(np.quantile(residual, 0.90)),
        "effective_correspondences": float(weights.sum()),
    }


@dataclass(frozen=True)
class SE3EdgeObservation:
    source_index: int
    target_index: int
    source_points: np.ndarray
    target_points: np.ndarray

    def __post_init__(self) -> None:
        source = np.asarray(self.source_points)
        target = np.asarray(self.target_points)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("edge points must both have shape [matches,3]")
        if source.shape[0] < 6:
            raise ValueError("each edge requires at least six correspondences")
        if self.source_index < 0 or self.target_index <= self.source_index:
            raise ValueError("edges must be directed from a lower to a higher node")


@dataclass(frozen=True)
class VariableSE3PoseGraph:
    node_timestamps: tuple[float, ...]
    edges: Tensor
    source_points: Tensor
    target_points: Tensor
    correspondence_mask: Tensor
    measurement_rotations: Tensor
    measurement_translations: Tensor
    initial_theta: Tensor
    block_id: Tensor
    initialization_diagnostics: tuple[dict[str, float], ...]
    huber_delta_m: float = 0.05

    @property
    def node_count(self) -> int:
        return len(self.node_timestamps)

    @property
    def edge_count(self) -> int:
        return int(self.edges.shape[0])

    @property
    def parameter_count(self) -> int:
        return 6 * (self.node_count - 1)

    @property
    def correspondence_count(self) -> int:
        return int(self.correspondence_mask.sum().item())

    def decode(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        if theta.ndim != 2 or theta.shape[1] != self.parameter_count:
            raise ValueError("theta must have shape [batch, 6 * (nodes - 1)]")
        free = theta.reshape(theta.shape[0], self.node_count - 1, 6)
        translation = TRANSLATION_SCALE_M * free[..., :3]
        rotation = so3_exp(ROTATION_SCALE_RAD * free[..., 3:])
        anchor_rotation = torch.eye(
            3, device=theta.device, dtype=theta.dtype
        ).reshape(1, 1, 3, 3).expand(theta.shape[0], 1, 3, 3)
        anchor_translation = theta.new_zeros(theta.shape[0], 1, 3)
        return (
            torch.cat((anchor_rotation, rotation), dim=1),
            torch.cat((anchor_translation, translation), dim=1),
        )

    def _costs(self, theta: Tensor) -> Tensor:
        rotations, translations = self.decode(theta)
        edges = self.edges.to(theta.device)
        source_rotation = rotations[:, edges[:, 0]]
        target_rotation = rotations[:, edges[:, 1]]
        source_translation = translations[:, edges[:, 0]]
        target_translation = translations[:, edges[:, 1]]
        source = self.source_points.to(theta)[None]
        target = self.target_points.to(theta)[None]
        in_anchor = torch.einsum("beij,bekj->beki", source_rotation, source)
        in_anchor = in_anchor + source_translation[:, :, None]
        predicted = torch.einsum(
            "beji,bekj->beki",
            target_rotation,
            in_anchor - target_translation[:, :, None],
        )
        costs = _pseudo_huber_norm(predicted - target, self.huber_delta_m)
        return costs * self.correspondence_mask.to(theta)[None]

    def factor_function(
        self,
        *,
        view: FactorView = "all",
        granularity: FactorGranularity = "edge",
    ) -> Callable[[Tensor], Tensor]:
        if view not in ("even", "odd", "all"):
            raise ValueError("view must be even, odd, or all")
        if granularity not in ("node", "edge", "correspondence"):
            raise ValueError("granularity must be node, edge, or correspondence")
        indices = torch.arange(self.source_points.shape[1])
        if view == "even":
            selected = (indices % 2) == 0
        elif view == "odd":
            selected = (indices % 2) == 1
        else:
            selected = torch.ones_like(indices, dtype=torch.bool)
        selected_mask = self.correspondence_mask & selected[None]

        def factors(theta: Tensor) -> Tensor:
            costs = self._costs(theta) * selected_mask.to(theta)[None]
            edge_cost = costs.sum(dim=-1)
            if granularity == "node":
                # Edges are directed from lower to higher temporal index.  A
                # segmented reduction over target nodes preserves the exact
                # scalar objective while avoiding dense attention over O(E)
                # separate factors on long loopy graphs.
                target_bucket = torch.maximum(
                    self.edges[:, 0], self.edges[:, 1]
                ).to(theta.device) - 1
                assignment = torch.nn.functional.one_hot(
                    target_bucket, num_classes=self.node_count - 1
                ).to(theta)
                return edge_cost @ assignment
            if granularity == "edge":
                return edge_cost
            return costs.reshape(theta.shape[0], -1)

        return factors

    def measurement_factor_function(
        self,
        *,
        reverse: bool = False,
        granularity: FactorGranularity = "correspondence",
        translation_scale_m: float = 0.05,
        rotation_scale_rad: float = 0.05,
    ) -> Callable[[Tensor], Tensor]:
        """Classical pose-graph residuals from robustly summarized edge poses.

        This is intentionally exposed as a stronger task-specific control, not
        as the same objective as the raw-correspondence SETSUNET Adapter.
        """

        if granularity not in ("node", "edge", "correspondence"):
            raise ValueError("granularity must be node, edge, or correspondence")
        if translation_scale_m <= 0.0 or rotation_scale_rad <= 0.0:
            raise ValueError("measurement residual scales must be positive")

        def factors(theta: Tensor) -> Tensor:
            rotations, translations = self.decode(theta)
            edges = self.edges.to(theta.device)
            measured_rotation = self.measurement_rotations.to(theta)
            measured_translation = self.measurement_translations.to(theta)
            if reverse:
                edges = torch.flip(edges, dims=(1,))
                measured_rotation = measured_rotation.transpose(-1, -2)
                measured_translation = -torch.einsum(
                    "eij,ej->ei", measured_rotation, measured_translation
                )
            source_rotation = rotations[:, edges[:, 0]]
            target_rotation = rotations[:, edges[:, 1]]
            predicted_rotation = target_rotation.transpose(-1, -2) @ source_rotation
            predicted_translation = torch.einsum(
                "beji,bej->bei",
                target_rotation,
                translations[:, edges[:, 0]] - translations[:, edges[:, 1]],
            )
            translation_error = (
                predicted_translation - measured_translation[None]
            ) / float(translation_scale_m)
            rotation_error = _so3_log(
                measured_rotation.transpose(-1, -2)[None] @ predicted_rotation
            ) / float(rotation_scale_rad)
            coordinate_cost = 0.5 * torch.cat(
                (translation_error, rotation_error), dim=-1
            ).square()
            edge_cost = coordinate_cost.sum(dim=-1)
            if granularity == "node":
                target_bucket = torch.maximum(edges[:, 0], edges[:, 1]) - 1
                assignment = torch.nn.functional.one_hot(
                    target_bucket, num_classes=self.node_count - 1
                ).to(theta)
                return edge_cost @ assignment
            if granularity == "edge":
                return edge_cost
            return coordinate_cost.reshape(theta.shape[0], -1)

        return factors


def build_variable_se3_pose_graph(
    node_timestamps: Sequence[float],
    observations: Sequence[SE3EdgeObservation],
    *,
    huber_delta_m: float = 0.05,
) -> VariableSE3PoseGraph:
    timestamps = tuple(float(value) for value in node_timestamps)
    node_count = len(timestamps)
    if node_count < 3 or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("at least three strictly ordered node timestamps are required")
    if huber_delta_m <= 0.0 or not observations:
        raise ValueError("positive Huber delta and non-empty observations required")
    if any(item.target_index >= node_count for item in observations):
        raise ValueError("edge index exceeds node count")
    ordered = sorted(observations, key=lambda item: (item.source_index, item.target_index))
    maximum = max(item.source_points.shape[0] for item in ordered)
    source = torch.zeros(len(ordered), maximum, 3, dtype=torch.float32)
    target = torch.zeros_like(source)
    mask = torch.zeros(len(ordered), maximum, dtype=torch.bool)
    edges = torch.empty(len(ordered), 2, dtype=torch.long)
    for edge_index, item in enumerate(ordered):
        count = item.source_points.shape[0]
        source[edge_index, :count] = torch.as_tensor(item.source_points)
        target[edge_index, :count] = torch.as_tensor(item.target_points)
        mask[edge_index, :count] = True
        edges[edge_index] = torch.tensor((item.source_index, item.target_index))

    measurements: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
    for item in ordered:
        measurements[(item.source_index, item.target_index)] = robust_rigid_transform(
            item.source_points, item.target_points
        )
    adjacency: list[list[tuple[int, np.ndarray, np.ndarray, dict[str, float]]]] = [
        [] for _ in range(node_count)
    ]
    for item in ordered:
        relative_rotation, relative_translation, diagnostic = measurements[
            (item.source_index, item.target_index)
        ]
        adjacency[item.source_index].append(
            (item.target_index, relative_rotation, relative_translation, diagnostic)
        )
        inverse_rotation = relative_rotation.T
        inverse_translation = -inverse_rotation @ relative_translation
        adjacency[item.target_index].append(
            (item.source_index, inverse_rotation, inverse_translation, diagnostic)
        )
    # Dijkstra with squared temporal-gap cost prefers consecutive odometry and
    # uses skip edges only to bridge missing links.  Plain BFS would attach all
    # long edges incident to node zero and produce a poor large-graph initial
    # trajectory even when a high-quality local path exists.
    distance = [math.inf] * node_count
    distance[0] = 0.0
    parent: list[tuple[int, np.ndarray, np.ndarray, dict[str, float]] | None] = [
        None
    ] * node_count
    queue: list[tuple[float, int]] = [(0.0, 0)]
    while queue:
        current_distance, source_index = heapq.heappop(queue)
        if current_distance != distance[source_index]:
            continue
        for target_index, relative_rotation, relative_translation, diagnostic in sorted(
            adjacency[source_index],
            key=lambda item: (abs(item[0] - source_index), item[0]),
        ):
            gap = abs(target_index - source_index)
            candidate = current_distance + float(gap * gap)
            if candidate + 1.0e-12 < distance[target_index]:
                distance[target_index] = candidate
                parent[target_index] = (
                    source_index,
                    relative_rotation,
                    relative_translation,
                    diagnostic,
                )
                heapq.heappush(queue, (candidate, target_index))
    if any(not math.isfinite(value) for value in distance):
        raise ValueError("successful observation graph is disconnected")
    rotations: list[np.ndarray | None] = [None] * node_count
    translations: list[np.ndarray | None] = [None] * node_count
    rotations[0] = np.eye(3, dtype=np.float64)
    translations[0] = np.zeros(3, dtype=np.float64)
    diagnostics: list[dict[str, float]] = []
    for target_index in sorted(range(1, node_count), key=lambda index: (distance[index], index)):
        selected = parent[target_index]
        if selected is None:
            raise ValueError("connected graph lacks a deterministic parent")
        source_index, relative_rotation, relative_translation, diagnostic = selected
        source_rotation = rotations[source_index]
        source_translation = translations[source_index]
        assert source_rotation is not None and source_translation is not None
        next_rotation = source_rotation @ relative_rotation.T
        next_translation = source_translation - next_rotation @ relative_translation
        rotations[target_index] = next_rotation
        translations[target_index] = next_translation
        diagnostics.append(
            {
                **diagnostic,
                "tree_source": float(source_index),
                "tree_target": float(target_index),
                "tree_path_cost": float(distance[target_index]),
            }
        )
    free = []
    for rotation, translation in zip(rotations[1:], translations[1:]):
        assert rotation is not None and translation is not None
        rotation_vector = _rotation_matrix_to_vector_numpy(rotation)
        free.extend(
            np.concatenate(
                (
                    translation / TRANSLATION_SCALE_M,
                    rotation_vector / ROTATION_SCALE_RAD,
                )
            ).tolist()
        )
    initial = torch.tensor(free, dtype=torch.float32).reshape(1, -1)
    measurement_rotations = torch.as_tensor(
        np.stack(
            [measurements[(item.source_index, item.target_index)][0] for item in ordered]
        ),
        dtype=torch.float32,
    )
    measurement_translations = torch.as_tensor(
        np.stack(
            [measurements[(item.source_index, item.target_index)][1] for item in ordered]
        ),
        dtype=torch.float32,
    )
    block_ids = []
    for pose_index in range(node_count - 1):
        block_ids.extend((2 * pose_index,) * 3)
        block_ids.extend((2 * pose_index + 1,) * 3)
    graph = VariableSE3PoseGraph(
        node_timestamps=timestamps,
        edges=edges,
        source_points=source,
        target_points=target,
        correspondence_mask=mask,
        measurement_rotations=measurement_rotations,
        measurement_translations=measurement_translations,
        initial_theta=initial,
        block_id=torch.tensor(block_ids, dtype=torch.long),
        initialization_diagnostics=tuple(diagnostics),
        huber_delta_m=float(huber_delta_m),
    )
    # Every parity view must contain evidence from every edge; fail closed rather
    # than allowing padding or an empty factor to become a topology cue.
    for view in ("even", "odd"):
        values = graph.factor_function(view=view)(initial)
        if values.shape != (1, len(ordered)) or not torch.isfinite(values).all():
            raise RuntimeError("invalid parity factor view")
    return graph


def encode_se3_poses(rotations: np.ndarray, translations: np.ndarray) -> Tensor:
    """Encode anchored camera-to-frame poses into the dimensionless state."""

    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape [nodes,3,3]")
    if translations.shape != (rotations.shape[0], 3):
        raise ValueError("translations must have shape [nodes,3]")
    if not np.allclose(rotations[0], np.eye(3), atol=1.0e-7) or not np.allclose(
        translations[0], 0.0, atol=1.0e-7
    ):
        raise ValueError("the first pose must be the identity anchor")
    free = []
    for rotation, translation in zip(rotations[1:], translations[1:]):
        free.extend(
            np.concatenate(
                (
                    translation / TRANSLATION_SCALE_M,
                    _rotation_matrix_to_vector_numpy(rotation) / ROTATION_SCALE_RAD,
                )
            ).tolist()
        )
    return torch.tensor(free, dtype=torch.float32).reshape(1, -1)


def anchored_ground_truth(
    rotations_world_camera: np.ndarray, translations_world_camera: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Express camera-to-world poses in the first camera's anchor frame."""

    rotations = np.asarray(rotations_world_camera, dtype=np.float64)
    translations = np.asarray(translations_world_camera, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape [nodes,3,3]")
    if translations.shape != (rotations.shape[0], 3):
        raise ValueError("translations must have shape [nodes,3]")
    anchor_inverse = rotations[0].T
    anchored_rotation = np.einsum("ij,njk->nik", anchor_inverse, rotations)
    anchored_translation = (translations - translations[0]) @ anchor_inverse.T
    return anchored_rotation, anchored_translation


def se3_pose_graph_metrics(
    estimated_rotations: np.ndarray,
    estimated_translations: np.ndarray,
    truth_rotations: np.ndarray,
    truth_translations: np.ndarray,
) -> dict[str, float]:
    estimated_rotations = np.asarray(estimated_rotations, dtype=np.float64)
    estimated_translations = np.asarray(estimated_translations, dtype=np.float64)
    truth_rotations = np.asarray(truth_rotations, dtype=np.float64)
    truth_translations = np.asarray(truth_translations, dtype=np.float64)
    if estimated_rotations.shape != truth_rotations.shape:
        raise ValueError("estimated and truth rotations must match")
    if estimated_translations.shape != truth_translations.shape:
        raise ValueError("estimated and truth translations must match")

    def angle_deg(relative: np.ndarray) -> np.ndarray:
        cosine = np.clip(
            (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0
        )
        return np.degrees(np.arccos(cosine))

    translation_error = np.linalg.norm(
        estimated_translations - truth_translations, axis=1
    )
    rotation_error = angle_deg(
        np.einsum("nji,njk->nik", estimated_rotations, truth_rotations)
    )
    estimated_relative_rotation = np.einsum(
        "nji,njk->nik", estimated_rotations[1:], estimated_rotations[:-1]
    )
    truth_relative_rotation = np.einsum(
        "nji,njk->nik", truth_rotations[1:], truth_rotations[:-1]
    )
    estimated_relative_translation = np.einsum(
        "nji,nj->ni",
        estimated_rotations[1:],
        estimated_translations[:-1] - estimated_translations[1:],
    )
    truth_relative_translation = np.einsum(
        "nji,nj->ni",
        truth_rotations[1:],
        truth_translations[:-1] - truth_translations[1:],
    )
    rpe_rotation = angle_deg(
        np.einsum(
            "nji,njk->nik", estimated_relative_rotation, truth_relative_rotation
        )
    )
    rpe_translation = np.linalg.norm(
        estimated_relative_translation - truth_relative_translation, axis=1
    )
    return {
        "ate_translation_rmse_m": float(np.sqrt(np.mean(translation_error**2))),
        "rotation_mae_deg": float(np.mean(rotation_error)),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(rpe_translation**2))),
        "rpe_rotation_mae_deg": float(np.mean(rpe_rotation)),
    }


def align_estimate_to_truth(
    estimated_rotations: np.ndarray,
    estimated_translations: np.ndarray,
    truth_translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly align an estimated trajectory to truth, without scale fitting."""

    estimated_rotations = np.asarray(estimated_rotations, dtype=np.float64)
    estimated_translations = np.asarray(estimated_translations, dtype=np.float64)
    truth_translations = np.asarray(truth_translations, dtype=np.float64)
    if estimated_rotations.shape != (estimated_translations.shape[0], 3, 3):
        raise ValueError("estimated rotations and translations disagree")
    if truth_translations.shape != estimated_translations.shape:
        raise ValueError("truth translations must match estimated translations")
    alignment_rotation, alignment_translation = _weighted_rigid_transform(
        estimated_translations,
        truth_translations,
        np.ones(estimated_translations.shape[0], dtype=np.float64),
    )
    return (
        np.einsum("ij,njk->nik", alignment_rotation, estimated_rotations),
        estimated_translations @ alignment_rotation.T + alignment_translation,
    )


__all__ = [
    "SE3EdgeObservation",
    "VariableSE3PoseGraph",
    "align_estimate_to_truth",
    "anchored_ground_truth",
    "build_variable_se3_pose_graph",
    "encode_se3_poses",
    "robust_rigid_transform",
    "se3_pose_graph_metrics",
]
