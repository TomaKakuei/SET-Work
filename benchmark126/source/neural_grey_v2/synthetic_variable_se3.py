"""Synthetic variable-size SE(3) graphs for learned-rule fitting only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .tum_rgbd_se3 import so3_exp
from .variable_se3_pose_graph import (
    SE3EdgeObservation,
    VariableSE3PoseGraph,
    build_variable_se3_pose_graph,
    encode_se3_poses,
)


@dataclass(frozen=True)
class SyntheticSE3Episode:
    graph: VariableSE3PoseGraph
    truth_theta: Tensor
    truth_rotations: np.ndarray
    truth_translations: np.ndarray


def generate_synthetic_variable_se3(
    node_count: int,
    *,
    seed: int,
    edge_gaps: tuple[int, ...] = (1, 2, 4, 8),
    correspondences: int = 48,
    observation_noise_m: float = 0.012,
    outlier_fraction: float = 0.08,
    rotation_step_std_rad: float = 0.035,
    forward_translation_mean_m: float = 0.045,
    forward_translation_std_m: float = 0.012,
    lateral_translation_std_m: float = 0.012,
    vertical_translation_std_m: float = 0.008,
) -> SyntheticSE3Episode:
    if node_count < 4 or correspondences < 16:
        raise ValueError("synthetic graph needs at least four nodes and 16 matches")
    if edge_gaps[0] != 1 or observation_noise_m <= 0.0:
        raise ValueError("edge gaps must start at one and noise must be positive")
    if not 0.0 <= outlier_fraction < 0.5:
        raise ValueError("outlier fraction must lie in [0,0.5)")
    if rotation_step_std_rad <= 0.0 or forward_translation_mean_m <= 0.0:
        raise ValueError("motion scales must be positive")
    if any(
        value < 0.0
        for value in (
            forward_translation_std_m,
            lateral_translation_std_m,
            vertical_translation_std_m,
        )
    ):
        raise ValueError("motion standard deviations must be non-negative")
    generator = np.random.default_rng(int(seed))
    rotations = [np.eye(3, dtype=np.float64)]
    translations = [np.zeros(3, dtype=np.float64)]
    for _ in range(node_count - 1):
        rotation_vector = generator.normal(0.0, rotation_step_std_rad, size=3)
        relative_rotation = (
            so3_exp(torch.tensor(rotation_vector, dtype=torch.float64)[None])[0]
            .numpy()
        )
        relative_translation = np.asarray(
            (
                generator.normal(
                    -forward_translation_mean_m, forward_translation_std_m
                ),
                generator.normal(0.0, lateral_translation_std_m),
                generator.normal(0.0, vertical_translation_std_m),
            ),
            dtype=np.float64,
        )
        next_rotation = rotations[-1] @ relative_rotation.T
        next_translation = translations[-1] - next_rotation @ relative_translation
        rotations.append(next_rotation)
        translations.append(next_translation)
    rotation_array = np.stack(rotations)
    translation_array = np.stack(translations)
    observations = []
    for gap in edge_gaps:
        if gap >= node_count:
            continue
        for source_index in range(node_count - gap):
            target_index = source_index + gap
            source_rotation = rotation_array[source_index]
            target_rotation = rotation_array[target_index]
            relative_rotation = target_rotation.T @ source_rotation
            relative_translation = target_rotation.T @ (
                translation_array[source_index] - translation_array[target_index]
            )
            source = generator.normal(0.0, 0.65, size=(correspondences, 3))
            source[:, 2] = generator.uniform(1.0, 4.5, size=correspondences)
            target = source @ relative_rotation.T + relative_translation
            target += generator.normal(
                0.0, observation_noise_m, size=target.shape
            )
            outliers = int(round(outlier_fraction * correspondences))
            if outliers:
                selected = generator.choice(correspondences, outliers, replace=False)
                target[selected] += generator.normal(0.0, 0.18, size=(outliers, 3))
            observations.append(
                SE3EdgeObservation(
                    source_index,
                    target_index,
                    source.astype(np.float32),
                    target.astype(np.float32),
                )
            )
    graph = build_variable_se3_pose_graph(
        tuple(float(index) for index in range(node_count)), observations
    )
    truth_theta = encode_se3_poses(rotation_array, translation_array)
    return SyntheticSE3Episode(
        graph=graph,
        truth_theta=truth_theta,
        truth_rotations=rotation_array,
        truth_translations=translation_array,
    )


__all__ = ["SyntheticSE3Episode", "generate_synthetic_variable_se3"]
