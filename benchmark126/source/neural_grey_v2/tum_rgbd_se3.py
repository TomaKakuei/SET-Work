"""Query-blind TUM RGB-D two-frame SE(3) factor-graph compiler.

The geometry in this module is deliberately implemented from tensor algebra.
OpenCV is used only for the frozen ORB observation compiler; it never estimates
a pose.  Ground truth utilities are separate from the compiler and factor graph
so evaluation code can keep supervision out of inference by construction.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor


DEPTH_SCALE = 5000.0
FREIBURG1_INTRINSICS = (517.3, 516.5, 318.6, 255.3)
FREIBURG2_INTRINSICS = (520.9, 521.0, 325.1, 249.7)
FREIBURG3_INTRINSICS = (535.4, 539.2, 320.1, 247.6)
TUM_RGB_INTRINSICS = {
    "freiburg1": FREIBURG1_INTRINSICS,
    "freiburg2": FREIBURG2_INTRINSICS,
    "freiburg3": FREIBURG3_INTRINSICS,
}
TRANSLATION_SCALE_M = 0.10
ROTATION_SCALE_RAD = 0.10
HUBER_DELTA_M = 0.05

# A tangent vector is not six unrelated scalars. Translation and rotation
# each form a coordinate-equivariant three-vector with their own physical
# type. Keeping the two types separate lets the generic curvature anchor
# retain within-vector coupling without mixing metres and radians. The
# singleton declaration is retained only for frozen legacy comparisons.
SE3_SINGLETON_BLOCK_ID = torch.arange(6, dtype=torch.long)
SE3_BLOCK_ID = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)


@dataclass(frozen=True)
class TimestampedPath:
    timestamp: float
    relative_path: str


@dataclass(frozen=True)
class SynchronizedObservation:
    rgb_timestamp: float
    rgb_path: Path
    depth_timestamp: float
    depth_path: Path


@dataclass(frozen=True)
class CandidatePair:
    candidate_index: int
    first_index: int
    second_index: int
    first: SynchronizedObservation | None
    second: SynchronizedObservation | None

    @property
    def status(self) -> str:
        return "ok" if self.first is not None and self.second is not None else "missing_observation"


@dataclass(frozen=True)
class GroundTruthPose:
    timestamp: float
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class CompiledSE3Observation:
    status: str
    source_points: np.ndarray
    target_points: np.ndarray
    detected_source: int
    detected_target: int
    crosscheck_matches: int
    valid_depth_matches: int
    intrinsics: tuple[float, float, float, float] = FREIBURG1_INTRINSICS

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"


def _data_lines(path: Path) -> Iterable[list[str]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield line.split()


def tum_rgb_intrinsics_from_sequence(
    sequence: Path | str,
) -> tuple[float, float, float, float]:
    """Resolve the official RGB calibration from a TUM sequence name.

    Calibration is physical Adapter metadata.  It is never exposed to the
    recurrent Solver, and unknown sensors fail closed instead of silently
    inheriting Freiburg1 geometry.
    """

    name = Path(sequence).name.lower()
    matches = [key for key in TUM_RGB_INTRINSICS if key in name]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve a unique TUM camera calibration from {name!r}")
    return TUM_RGB_INTRINSICS[matches[0]]


def read_timestamped_paths(index_path: Path) -> list[TimestampedPath]:
    rows: list[TimestampedPath] = []
    for fields in _data_lines(index_path):
        if len(fields) < 2:
            raise ValueError(f"malformed timestamped path in {index_path}: {fields}")
        rows.append(TimestampedPath(float(fields[0]), fields[1]))
    if any(b.timestamp < a.timestamp for a, b in zip(rows, rows[1:])):
        raise ValueError(f"timestamps are not sorted in {index_path}")
    return rows


def synchronize_rgb_depth(
    sequence_root: Path,
    *,
    maximum_delta_seconds: float = 0.020,
) -> list[SynchronizedObservation]:
    """Greedily associate each RGB frame with its nearest unused depth frame."""

    rgb = read_timestamped_paths(sequence_root / "rgb.txt")
    depth = read_timestamped_paths(sequence_root / "depth.txt")
    depth_times = [row.timestamp for row in depth]
    used: set[int] = set()
    synchronized: list[SynchronizedObservation] = []
    for rgb_row in rgb:
        left = bisect_left(depth_times, rgb_row.timestamp - maximum_delta_seconds)
        right = bisect_right(depth_times, rgb_row.timestamp + maximum_delta_seconds)
        candidates = sorted(
            (
                (abs(depth[index].timestamp - rgb_row.timestamp), depth[index].timestamp, index)
                for index in range(left, right)
                if index not in used
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        if not candidates:
            continue
        _, _, selected = candidates[0]
        used.add(selected)
        synchronized.append(
            SynchronizedObservation(
                rgb_timestamp=rgb_row.timestamp,
                rgb_path=sequence_root / rgb_row.relative_path,
                depth_timestamp=depth[selected].timestamp,
                depth_path=sequence_root / depth[selected].relative_path,
            )
        )
    return synchronized


def enumerate_candidate_pairs(
    observations: Sequence[SynchronizedObservation],
    *,
    count: int,
    stride: int = 8,
    gap: int = 30,
) -> list[CandidatePair]:
    output: list[CandidatePair] = []
    for candidate_index in range(count):
        first_index = stride * candidate_index
        second_index = first_index + gap
        first = observations[first_index] if first_index < len(observations) else None
        second = observations[second_index] if second_index < len(observations) else None
        output.append(
            CandidatePair(
                candidate_index=candidate_index,
                first_index=first_index,
                second_index=second_index,
                first=first,
                second=second,
            )
        )
    return output


def _imread(path: Path, flags: int) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    return image


def _point_from_depth(
    keypoint: cv2.KeyPoint,
    depth: np.ndarray,
    *,
    intrinsics: tuple[float, float, float, float],
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> np.ndarray | None:
    u = int(round(float(keypoint.pt[0])))
    v = int(round(float(keypoint.pt[1])))
    if u < 0 or v < 0 or u >= depth.shape[1] or v >= depth.shape[0]:
        return None
    z = float(depth[v, u]) / DEPTH_SCALE
    if z < minimum_depth_m or z > maximum_depth_m:
        return None
    fx, fy, cx, cy = intrinsics
    return np.asarray(((u - cx) * z / fx, (v - cy) * z / fy, z), dtype=np.float32)


class FrozenORB3DCompiler:
    """Compile fixed mutual ORB matches into 3D--3D observations, without pose fitting."""

    def __init__(
        self,
        *,
        maximum_correspondences: int = 256,
        minimum_correspondences: int = 32,
        minimum_depth_m: float = 0.30,
        maximum_depth_m: float = 5.00,
        intrinsics: tuple[float, float, float, float] = FREIBURG1_INTRINSICS,
    ) -> None:
        if minimum_correspondences < 3 or maximum_correspondences < minimum_correspondences:
            raise ValueError("invalid correspondence limits")
        self.maximum_correspondences = int(maximum_correspondences)
        self.minimum_correspondences = int(minimum_correspondences)
        self.minimum_depth_m = float(minimum_depth_m)
        self.maximum_depth_m = float(maximum_depth_m)
        self.intrinsics = tuple(float(value) for value in intrinsics)
        cv2.setNumThreads(1)
        cv2.setRNGSeed(4101)
        self.orb = cv2.ORB_create(
            nfeatures=1500,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def compile(
        self,
        first: SynchronizedObservation,
        second: SynchronizedObservation,
    ) -> CompiledSE3Observation:
        first_rgb = _imread(first.rgb_path, cv2.IMREAD_GRAYSCALE)
        second_rgb = _imread(second.rgb_path, cv2.IMREAD_GRAYSCALE)
        first_depth = _imread(first.depth_path, cv2.IMREAD_UNCHANGED)
        second_depth = _imread(second.depth_path, cv2.IMREAD_UNCHANGED)
        if first_depth.dtype != np.uint16 or second_depth.dtype != np.uint16:
            raise ValueError("TUM depth PNGs must decode as uint16")
        first_keypoints, first_descriptors = self.orb.detectAndCompute(first_rgb, None)
        second_keypoints, second_descriptors = self.orb.detectAndCompute(second_rgb, None)
        detected_first = len(first_keypoints)
        detected_second = len(second_keypoints)
        if first_descriptors is None or second_descriptors is None:
            return CompiledSE3Observation(
                "missing_descriptors",
                np.empty((0, 3), np.float32),
                np.empty((0, 3), np.float32),
                detected_first,
                detected_second,
                0,
                0,
                self.intrinsics,
            )
        matches = sorted(
            self.matcher.match(first_descriptors, second_descriptors),
            key=lambda match: (float(match.distance), int(match.queryIdx), int(match.trainIdx)),
        )
        source: list[np.ndarray] = []
        target: list[np.ndarray] = []
        for match in matches:
            first_point = _point_from_depth(
                first_keypoints[match.queryIdx],
                first_depth,
                intrinsics=self.intrinsics,
                minimum_depth_m=self.minimum_depth_m,
                maximum_depth_m=self.maximum_depth_m,
            )
            second_point = _point_from_depth(
                second_keypoints[match.trainIdx],
                second_depth,
                intrinsics=self.intrinsics,
                minimum_depth_m=self.minimum_depth_m,
                maximum_depth_m=self.maximum_depth_m,
            )
            if first_point is None or second_point is None:
                continue
            source.append(first_point)
            target.append(second_point)
            if len(source) == self.maximum_correspondences:
                break
        valid = len(source)
        status = "ok" if valid >= self.minimum_correspondences else "insufficient_correspondences"
        return CompiledSE3Observation(
            status=status,
            source_points=np.asarray(source, dtype=np.float32).reshape(-1, 3),
            target_points=np.asarray(target, dtype=np.float32).reshape(-1, 3),
            detected_source=detected_first,
            detected_target=detected_second,
            crosscheck_matches=len(matches),
            valid_depth_matches=valid,
            intrinsics=self.intrinsics,
        )


def skew_symmetric(vector: Tensor) -> Tensor:
    if vector.shape[-1] != 3:
        raise ValueError("vector must end in dimension three")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(rotation_vector: Tensor) -> Tensor:
    """Differentiable Rodrigues exponential with a finite small-angle branch."""

    if rotation_vector.shape[-1] != 3:
        raise ValueError("rotation vector must end in dimension three")
    squared_angle = rotation_vector.square().sum(dim=-1, keepdim=True)
    safe_squared = squared_angle.clamp_min(torch.finfo(rotation_vector.dtype).eps)
    safe_angle = safe_squared.sqrt()
    series_a = 1.0 - squared_angle / 6.0 + squared_angle.square() / 120.0
    series_b = 0.5 - squared_angle / 24.0 + squared_angle.square() / 720.0
    exact_a = safe_angle.sin() / safe_angle
    exact_b = (1.0 - safe_angle.cos()) / safe_squared
    small = squared_angle < 1.0e-8
    a = torch.where(small, series_a, exact_a)[..., None]
    b = torch.where(small, series_b, exact_b)[..., None]
    hat = skew_symmetric(rotation_vector)
    identity = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    identity = identity.expand(*rotation_vector.shape[:-1], 3, 3)
    return identity + a * hat + b * (hat @ hat)


def decode_se3(theta: Tensor) -> tuple[Tensor, Tensor]:
    if theta.ndim != 2 or theta.shape[1] != 6:
        raise ValueError("theta must have shape [batch, 6]")
    translation = TRANSLATION_SCALE_M * theta[:, :3]
    rotation = so3_exp(ROTATION_SCALE_RAD * theta[:, 3:])
    return rotation, translation


def se3_factor_graph(
    observation: CompiledSE3Observation,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    huber_delta_m: float = HUBER_DELTA_M,
) -> tuple[Callable[[Tensor], Tensor], Tensor, Tensor, Callable[[Tensor], tuple[Tensor, Tensor]]]:
    if not observation.succeeded:
        raise ValueError("cannot build a factor graph from a failed observation")
    device = torch.device(device)
    source = torch.as_tensor(observation.source_points, device=device, dtype=dtype)
    target = torch.as_tensor(observation.target_points, device=device, dtype=dtype)
    delta = float(huber_delta_m)
    if delta <= 0.0:
        raise ValueError("Huber delta must be positive")

    def decode(theta: Tensor) -> tuple[Tensor, Tensor]:
        return decode_se3(theta)

    def factor_function(theta: Tensor) -> Tensor:
        rotation, translation = decode(theta)
        predicted = torch.einsum("bij,nj->bni", rotation, source) + translation[:, None, :]
        squared_norm = (predicted - target[None]).square().sum(dim=-1)
        norm = squared_norm.clamp_min(1.0e-24).sqrt()
        return torch.where(
            squared_norm <= delta * delta,
            0.5 * squared_norm,
            delta * (norm - 0.5 * delta),
        )

    initial = torch.zeros(1, 6, device=device, dtype=dtype)
    return factor_function, initial, SE3_BLOCK_ID.to(device), decode


def stacked_se3_factor_graph(
    observations: Sequence[CompiledSE3Observation],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    huber_delta_m: float = HUBER_DELTA_M,
) -> tuple[
    Callable[[Tensor], Tensor],
    Tensor,
    Tensor,
    Callable[[Tensor], tuple[Tensor, Tensor]],
]:
    """Stack equal-cardinality SE(3) graphs without cross-example coupling.

    Equal factor cardinality preserves the exact per-graph typed interface and
    avoids padding statistics becoming an implicit topology cue.  This is a
    throughput operator; each row remains an independent graph.
    """

    if not observations:
        raise ValueError("at least one observation is required")
    if any(not observation.succeeded for observation in observations):
        raise ValueError("all stacked observations must have succeeded")
    factor_counts = {int(observation.source_points.shape[0]) for observation in observations}
    if len(factor_counts) != 1:
        raise ValueError("stacked observations must have equal factor cardinality")
    if any(
        observation.source_points.shape != observation.target_points.shape
        or observation.source_points.ndim != 2
        or observation.source_points.shape[1] != 3
        for observation in observations
    ):
        raise ValueError("all source/target point arrays must have shape [F,3]")

    device = torch.device(device)
    source = torch.as_tensor(
        np.stack([observation.source_points for observation in observations]),
        device=device,
        dtype=dtype,
    )
    target = torch.as_tensor(
        np.stack([observation.target_points for observation in observations]),
        device=device,
        dtype=dtype,
    )
    delta = float(huber_delta_m)
    if delta <= 0.0:
        raise ValueError("Huber delta must be positive")

    def decode(theta: Tensor) -> tuple[Tensor, Tensor]:
        return decode_se3(theta)

    def factor_function(theta: Tensor) -> Tensor:
        if theta.shape[0] != source.shape[0]:
            raise ValueError("theta batch must match stacked observation count")
        rotation, translation = decode(theta)
        predicted = torch.einsum("bij,bnj->bni", rotation, source) + translation[:, None, :]
        squared_norm = (predicted - target).square().sum(dim=-1)
        norm = squared_norm.clamp_min(1.0e-24).sqrt()
        return torch.where(
            squared_norm <= delta * delta,
            0.5 * squared_norm,
            delta * (norm - 0.5 * delta),
        )

    initial = torch.zeros(len(observations), 6, device=device, dtype=dtype)
    return factor_function, initial, SE3_BLOCK_ID.to(device), decode


def so3_right_jacobian(rotation_vector: Tensor) -> Tensor:
    """Right Jacobian of SO(3), including a finite small-angle branch."""

    if rotation_vector.shape[-1] != 3:
        raise ValueError("rotation vector must end in dimension three")
    squared_angle = rotation_vector.square().sum(dim=-1, keepdim=True)
    angle = squared_angle.clamp_min(torch.finfo(rotation_vector.dtype).eps).sqrt()
    angle_four = squared_angle.square()
    series_a = 0.5 - squared_angle / 24.0 + angle_four / 720.0
    series_b = 1.0 / 6.0 - squared_angle / 120.0 + angle_four / 5040.0
    exact_a = (1.0 - angle.cos()) / squared_angle.clamp_min(
        torch.finfo(rotation_vector.dtype).eps
    )
    exact_b = (angle - angle.sin()) / (
        squared_angle * angle
    ).clamp_min(torch.finfo(rotation_vector.dtype).eps)
    small = squared_angle < 1.0e-8
    a = torch.where(small, series_a, exact_a)[..., None]
    b = torch.where(small, series_b, exact_b)[..., None]
    hat = skew_symmetric(rotation_vector)
    identity = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    identity = identity.expand(*rotation_vector.shape[:-1], 3, 3)
    return identity - a * hat + b * (hat @ hat)


def stacked_se3_huber_residual_jacobian(
    theta: Tensor,
    source: Tensor,
    target: Tensor,
    *,
    huber_delta_m: float = HUBER_DELTA_M,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Analytic robust scalar residuals and Jacobians for stacked SE(3) graphs.

    Returns residuals ``[B,F]``, Jacobians ``[B,F,6]``, and factor costs
    ``[B,F]``.  The Jacobian is with respect to the dimensionless DA-DFS state.
    """

    if theta.ndim != 2 or theta.shape[1] != 6:
        raise ValueError("theta must have shape [B,6]")
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("source and target must have shape [B,F,3]")
    if source.shape[0] != theta.shape[0]:
        raise ValueError("point and state batches must match")
    delta = float(huber_delta_m)
    if delta <= 0.0 or epsilon <= 0.0:
        raise ValueError("Huber delta and epsilon must be positive")

    translation = TRANSLATION_SCALE_M * theta[:, :3]
    rotation_vector = ROTATION_SCALE_RAD * theta[:, 3:]
    rotation = so3_exp(rotation_vector)
    right_jacobian = so3_right_jacobian(rotation_vector)
    predicted = torch.einsum("bij,bfj->bfi", rotation, source) + translation[:, None, :]
    error = predicted - target
    squared_norm = error.square().sum(dim=-1)
    norm = squared_norm.clamp_min(1.0e-24).sqrt()
    inlier = squared_norm <= delta * delta
    costs = torch.where(
        inlier,
        0.5 * squared_norm,
        delta * (norm - 0.5 * delta),
    )
    residual = (2.0 * costs + epsilon).sqrt()
    cost_gradient_scale = torch.where(inlier, torch.ones_like(norm), delta / norm)
    residual_gradient_error = (
        cost_gradient_scale / residual
    )[..., None] * error

    identity = torch.eye(3, device=theta.device, dtype=theta.dtype)
    translation_jacobian = (
        TRANSLATION_SCALE_M
        * identity[None, None].expand(theta.shape[0], source.shape[1], 3, 3)
    )
    source_hat = skew_symmetric(source)
    rotation_jacobian = -ROTATION_SCALE_RAD * torch.matmul(
        torch.matmul(rotation[:, None], source_hat),
        right_jacobian[:, None],
    )
    point_jacobian = torch.cat((translation_jacobian, rotation_jacobian), dim=-1)
    jacobian = torch.einsum(
        "bfi,bfij->bfj", residual_gradient_error, point_jacobian
    )
    return residual, jacobian, costs


def run_stacked_se3_lm5(
    observations: Sequence[CompiledSE3Observation],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    initial_damping: float,
    trust_radius: float,
    rounds: int = 5,
    huber_delta_m: float = HUBER_DELTA_M,
    epsilon: float = 1.0e-12,
) -> tuple[Tensor, dict[str, object]]:
    """Five-round batched LM using explicit SE(3) and Huber Jacobians."""

    if initial_damping <= 0.0 or trust_radius <= 0.0 or rounds < 1:
        raise ValueError("damping, trust radius, and rounds must be positive")
    if not observations or any(not item.succeeded for item in observations):
        raise ValueError("all observations must be successful")
    counts = {int(item.source_points.shape[0]) for item in observations}
    if len(counts) != 1:
        raise ValueError("stacked observations must have equal factor cardinality")
    device = torch.device(device)
    source = torch.as_tensor(
        np.stack([item.source_points for item in observations]),
        device=device,
        dtype=dtype,
    )
    target = torch.as_tensor(
        np.stack([item.target_points for item in observations]),
        device=device,
        dtype=dtype,
    )
    batch = len(observations)
    theta = torch.zeros(batch, 6, device=device, dtype=dtype)
    damping = theta.new_full((batch,), float(initial_damping))
    identity = torch.eye(6, device=device, dtype=dtype)[None].expand(batch, 6, 6)

    def objective(candidate: Tensor) -> Tensor:
        _, _, costs = stacked_se3_huber_residual_jacobian(
            candidate,
            source,
            target,
            huber_delta_m=huber_delta_m,
            epsilon=epsilon,
        )
        return costs.sum(dim=-1)

    current_objective = objective(theta)
    objective_trace = [current_objective.detach().cpu().tolist()]
    accepted_trace: list[list[bool]] = []
    for _ in range(rounds):
        residual, jacobian, _ = stacked_se3_huber_residual_jacobian(
            theta,
            source,
            target,
            huber_delta_m=huber_delta_m,
            epsilon=epsilon,
        )
        normal = torch.matmul(jacobian.transpose(1, 2), jacobian)
        gradient = torch.matmul(jacobian.transpose(1, 2), residual[..., None])[..., 0]
        diagonal = torch.diagonal(normal, dim1=-2, dim2=-1).clamp_min(1.0e-6)
        system = normal + damping[:, None, None] * torch.diag_embed(diagonal) + 1.0e-7 * identity
        try:
            step = torch.linalg.solve(system, -gradient)
        except RuntimeError:
            step = torch.linalg.lstsq(system, -gradient[..., None]).solution[..., 0]
        finite = torch.isfinite(step).all(dim=-1)
        norm = torch.linalg.vector_norm(step, dim=-1)
        scale = torch.minimum(
            torch.ones_like(norm),
            theta.new_full(norm.shape, float(trust_radius)) / norm.clamp_min(1.0e-12),
        )
        step = step * scale[:, None]
        candidate = theta + step
        candidate_objective = objective(candidate)
        accepted = finite & torch.isfinite(candidate_objective) & (
            candidate_objective <= current_objective
        )
        theta = torch.where(accepted[:, None], candidate, theta)
        current_objective = torch.where(
            accepted, candidate_objective, current_objective
        )
        damping = torch.where(
            accepted,
            (0.3 * damping).clamp_min(1.0e-9),
            (10.0 * damping).clamp_max(1.0e9),
        )
        accepted_trace.append(accepted.detach().cpu().tolist())
        objective_trace.append(current_objective.detach().cpu().tolist())

    return theta.detach(), {
        "rounds": rounds,
        "factor_value_calls": 1 + 2 * rounds,
        "accepted_by_round": accepted_trace,
        "objective_trace": objective_trace,
    }


def read_ground_truth(path: Path) -> list[GroundTruthPose]:
    rows: list[GroundTruthPose] = []
    for fields in _data_lines(path):
        if len(fields) < 8:
            raise ValueError(f"malformed ground truth row in {path}: {fields}")
        values = np.asarray([float(value) for value in fields[:8]], dtype=np.float64)
        rows.append(
            GroundTruthPose(
                timestamp=float(values[0]),
                translation=values[1:4],
                quaternion_xyzw=values[4:8],
            )
        )
    if any(b.timestamp < a.timestamp for a, b in zip(rows, rows[1:])):
        raise ValueError(f"ground truth timestamps are not sorted in {path}")
    return rows


def associate_ground_truth(
    poses: Sequence[GroundTruthPose],
    timestamp: float,
    *,
    maximum_delta_seconds: float = 0.020,
) -> GroundTruthPose | None:
    times = [pose.timestamp for pose in poses]
    insertion = bisect_left(times, timestamp)
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(poses)]
    if not candidates:
        return None
    selected = min(candidates, key=lambda index: (abs(times[index] - timestamp), times[index]))
    return poses[selected] if abs(times[selected] - timestamp) <= maximum_delta_seconds else None


def quaternion_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quaternion must have shape [4]")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("invalid quaternion")
    x, y, z, w = q / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def relative_camera_transform(
    first: GroundTruthPose,
    second: GroundTruthPose,
) -> tuple[np.ndarray, np.ndarray]:
    """Return p_c2 = R p_c1 + t from camera-to-world TUM poses."""

    first_rotation = quaternion_xyzw_to_rotation(first.quaternion_xyzw)
    second_rotation = quaternion_xyzw_to_rotation(second.quaternion_xyzw)
    rotation = second_rotation.T @ first_rotation
    translation = second_rotation.T @ (first.translation - second.translation)
    return rotation, translation


def pose_errors(
    estimated_rotation: np.ndarray,
    estimated_translation: np.ndarray,
    ground_truth_rotation: np.ndarray,
    ground_truth_translation: np.ndarray,
) -> dict[str, float]:
    relative = np.asarray(estimated_rotation, np.float64).T @ np.asarray(
        ground_truth_rotation, np.float64
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    rotation_error_deg = float(np.degrees(np.arccos(cosine)))
    translation_error_m = float(
        np.linalg.norm(
            np.asarray(estimated_translation, np.float64)
            - np.asarray(ground_truth_translation, np.float64)
        )
    )
    return {
        "rotation_error_deg": rotation_error_deg,
        "translation_error_cm": 100.0 * translation_error_m,
        "se3_score": rotation_error_deg + 10.0 * translation_error_m,
    }


__all__ = [
    "CandidatePair",
    "CompiledSE3Observation",
    "FREIBURG1_INTRINSICS",
    "FREIBURG2_INTRINSICS",
    "FREIBURG3_INTRINSICS",
    "FrozenORB3DCompiler",
    "GroundTruthPose",
    "HUBER_DELTA_M",
    "ROTATION_SCALE_RAD",
    "SE3_BLOCK_ID",
    "SE3_SINGLETON_BLOCK_ID",
    "SynchronizedObservation",
    "TRANSLATION_SCALE_M",
    "TUM_RGB_INTRINSICS",
    "associate_ground_truth",
    "decode_se3",
    "enumerate_candidate_pairs",
    "pose_errors",
    "quaternion_xyzw_to_rotation",
    "read_ground_truth",
    "read_timestamped_paths",
    "relative_camera_transform",
    "se3_factor_graph",
    "skew_symmetric",
    "so3_exp",
    "so3_right_jacobian",
    "stacked_se3_factor_graph",
    "stacked_se3_huber_residual_jacobian",
    "run_stacked_se3_lm5",
    "synchronize_rgb_depth",
    "tum_rgb_intrinsics_from_sequence",
]
