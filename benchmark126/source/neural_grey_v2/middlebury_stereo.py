"""Query-blind dense disparity factor graphs for Middlebury Stereo v3.

All optimization operators are implemented with PyTorch tensor algebra.  The
only image-library operation is deterministic PNG decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .factor_sketch import GridFactorSketchSpec, hierarchical_grid_sketch


DOWNSAMPLE_SCALE = 0.5
CROP_HEIGHT = 32
CROP_WIDTH = 48
CROP_ROWS = 3
CROP_COLUMNS = 4
STEREO_BLOCK_ID = torch.zeros(CROP_HEIGHT * CROP_WIDTH, dtype=torch.long)


@dataclass(frozen=True)
class StereoSceneObservation:
    scene: str
    root: Path
    left_full: Tensor
    right_full: Tensor
    initial_disparity_full: Tensor
    crop_origins: tuple[tuple[int, int], ...]
    valid_crop: tuple[bool, ...]
    maximum_disparity: int

    @property
    def crop_count(self) -> int:
        return len(self.crop_origins)


def _decode_grayscale(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def parse_ndisp(path: Path | str) -> int:
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"(?m)^ndisp\s*=\s*(\d+)\s*$", text)
    if match is None:
        raise ValueError(f"missing ndisp in calibration: {path}")
    value = int(match.group(1))
    if value <= 0:
        raise ValueError("ndisp must be positive")
    return value


def read_pfm(path: Path | str) -> np.ndarray:
    """Read a single- or three-channel PFM without an external codec."""

    with Path(path).open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"invalid PFM header: {header}")
        dimensions = handle.readline().decode("ascii").strip().split()
        while dimensions and dimensions[0].startswith("#"):
            dimensions = handle.readline().decode("ascii").strip().split()
        if len(dimensions) != 2:
            raise ValueError("invalid PFM dimensions")
        width, height = (int(value) for value in dimensions)
        scale = float(handle.readline().decode("ascii").strip())
        little_endian = scale < 0.0
        channels = 1 if header == "Pf" else 3
        dtype = np.dtype("<f4" if little_endian else ">f4")
        data = np.frombuffer(handle.read(), dtype=dtype)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM payload has {data.size} values, expected {expected}")
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.flipud(data.reshape(shape)).astype(np.float32, copy=True)


def _census(image: Tensor) -> Tensor:
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("census expects [B,1,H,W]")
    patches = F.unfold(image, kernel_size=3, padding=1).reshape(
        image.shape[0], 9, image.shape[2], image.shape[3]
    )
    center = patches[:, 4:5]
    neighbors = torch.cat((patches[:, :4], patches[:, 5:]), dim=1)
    return (neighbors >= center).to(image.dtype)


def _shift_right_to_left(values: Tensor, disparity: int, fill: float) -> Tensor:
    """Return values sampled at x-disparity and aligned to left coordinates."""

    if disparity == 0:
        return values
    output = values.new_full(values.shape, fill)
    output[..., disparity:] = values[..., : values.shape[-1] - disparity]
    return output


def census_wta_disparity(
    left: Tensor,
    right: Tensor,
    *,
    maximum_disparity: int,
    window: int = 5,
) -> Tensor:
    """Hand-written census/absolute-difference cost volume and WTA."""

    if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 1:
        raise ValueError("stereo images must have the same [B,1,H,W] shape")
    if maximum_disparity < 1 or maximum_disparity >= left.shape[-1]:
        raise ValueError("maximum disparity must lie within the image width")
    left_census = _census(left)
    right_census = _census(right)
    costs = []
    for disparity in range(maximum_disparity + 1):
        shifted_census = _shift_right_to_left(right_census, disparity, 1.0)
        shifted_right = _shift_right_to_left(right, disparity, 1.0)
        census_cost = (left_census - shifted_census).abs().mean(dim=1, keepdim=True)
        intensity_cost = (left - shifted_right).abs()
        local = F.avg_pool2d(
            census_cost + 0.20 * intensity_cost,
            kernel_size=window,
            stride=1,
            padding=window // 2,
        )
        if disparity > 0:
            local[..., :disparity] = 10.0
        costs.append(local)
    volume = torch.cat(costs, dim=1)
    disparity = volume.argmin(dim=1, keepdim=True).to(left.dtype)
    # A package-free 3x3 median suppresses isolated WTA spikes.
    patches = F.unfold(disparity, kernel_size=3, padding=1).reshape(
        disparity.shape[0], 9, disparity.shape[2], disparity.shape[3]
    )
    return patches.median(dim=1, keepdim=True).values


def _fixed_crop_origins(
    height: int,
    width: int,
    maximum_disparity: int,
) -> tuple[tuple[int, int], ...]:
    if height < CROP_HEIGHT or width < CROP_WIDTH:
        raise ValueError("downsampled scene is smaller than the crop")
    y_centers = np.linspace(CROP_HEIGHT // 2, height - CROP_HEIGHT // 2, CROP_ROWS)
    left_margin = min(maximum_disparity + CROP_WIDTH // 2, width - CROP_WIDTH // 2)
    x_centers = np.linspace(left_margin, width - CROP_WIDTH // 2, CROP_COLUMNS)
    origins = []
    for y_center in y_centers:
        for x_center in x_centers:
            y0 = int(round(float(y_center))) - CROP_HEIGHT // 2
            x0 = int(round(float(x_center))) - CROP_WIDTH // 2
            y0 = min(max(y0, 0), height - CROP_HEIGHT)
            x0 = min(max(x0, 0), width - CROP_WIDTH)
            origins.append((y0, x0))
    return tuple(origins)


def compile_middlebury_scene(
    scene_root: Path | str,
    *,
    device: torch.device | str,
) -> StereoSceneObservation:
    root = Path(scene_root)
    device = torch.device(device)
    left_np = _decode_grayscale(root / "im0.png")
    right_np = _decode_grayscale(root / "im1.png")
    if left_np.shape != right_np.shape:
        raise ValueError("Middlebury stereo images have different shapes")
    left = torch.as_tensor(left_np, device=device, dtype=torch.float32)[None, None] / 255.0
    right = torch.as_tensor(right_np, device=device, dtype=torch.float32)[None, None] / 255.0
    left = F.interpolate(left, scale_factor=DOWNSAMPLE_SCALE, mode="area")
    right = F.interpolate(right, scale_factor=DOWNSAMPLE_SCALE, mode="area")
    maximum_disparity = max(1, int(math_ceil(parse_ndisp(root / "calib.txt") * DOWNSAMPLE_SCALE)))
    maximum_disparity = min(maximum_disparity, left.shape[-1] - 1)
    initial = census_wta_disparity(left, right, maximum_disparity=maximum_disparity)
    origins = _fixed_crop_origins(left.shape[-2], left.shape[-1], maximum_disparity)
    valid = []
    for y0, x0 in origins:
        disparity = initial[0, 0, y0 : y0 + CROP_HEIGHT, x0 : x0 + CROP_WIDTH]
        xx = torch.arange(x0, x0 + CROP_WIDTH, device=device)[None].expand(CROP_HEIGHT, -1)
        valid.append(float((xx - disparity >= 0.0).float().mean()) >= 0.60)
    return StereoSceneObservation(
        scene=root.name,
        root=root,
        left_full=left.detach(),
        right_full=right.detach(),
        initial_disparity_full=initial.detach(),
        crop_origins=origins,
        valid_crop=tuple(valid),
        maximum_disparity=maximum_disparity,
    )


def math_ceil(value: float) -> int:
    # Kept local to make the exact integer conversion explicit and testable.
    integer = int(value)
    return integer if value == integer else integer + 1


def horizontal_bilinear_sample(rows: Tensor, x_coordinates: Tensor) -> Tensor:
    """Sample `[B,1,H,W]` rows at differentiable `[B,H,K]` x coordinates."""

    if rows.ndim != 4 or rows.shape[1] != 1 or x_coordinates.ndim != 3:
        raise ValueError("invalid row sampler shapes")
    if rows.shape[0] != x_coordinates.shape[0] or rows.shape[2] != x_coordinates.shape[1]:
        raise ValueError("row sampler batch/height mismatch")
    width = rows.shape[-1]
    coordinates = x_coordinates.clamp(0.0, float(width - 1) - 1.0e-6)
    lower = coordinates.floor().long()
    upper = (lower + 1).clamp_max(width - 1)
    weight = coordinates - lower.to(coordinates.dtype)
    flat = rows[:, 0]
    lower_values = torch.gather(flat, 2, lower)
    upper_values = torch.gather(flat, 2, upper)
    return ((1.0 - weight) * lower_values + weight * upper_values)[:, None]


def stereo_scene_factor_graph(
    observation: StereoSceneObservation,
    *,
    update_scale_px: float = 1.0,
    tv_weight: float = 0.08,
    trust_weight: float = 0.002,
) -> tuple[Callable[[Tensor], Tensor], Tensor, Tensor, Callable[[Tensor], Tensor]]:
    if update_scale_px <= 0.0 or tv_weight <= 0.0 or trust_weight <= 0.0:
        raise ValueError("stereo graph weights/scales must be positive")
    device = observation.left_full.device
    left_crops = []
    right_rows = []
    initial_crops = []
    base_x = []
    for y0, x0 in observation.crop_origins:
        left_crops.append(
            observation.left_full[:, :, y0 : y0 + CROP_HEIGHT, x0 : x0 + CROP_WIDTH]
        )
        right_rows.append(
            observation.right_full[:, :, y0 : y0 + CROP_HEIGHT, :]
        )
        initial_crops.append(
            observation.initial_disparity_full[
                :, :, y0 : y0 + CROP_HEIGHT, x0 : x0 + CROP_WIDTH
            ]
        )
        base_x.append(
            torch.arange(x0, x0 + CROP_WIDTH, device=device, dtype=torch.float32)
            .reshape(1, 1, CROP_WIDTH)
            .expand(1, CROP_HEIGHT, -1)
        )
    left = torch.cat(left_crops, dim=0)
    right = torch.cat(right_rows, dim=0)
    initial_disparity = torch.cat(initial_crops, dim=0)
    x_grid = torch.cat(base_x, dim=0)
    valid_crop = torch.as_tensor(observation.valid_crop, device=device, dtype=left.dtype)
    spec = GridFactorSketchSpec(CROP_HEIGHT, CROP_WIDTH, ((1, 1), (2, 2), (4, 4)))

    left_dx = F.pad(left[..., 1:] - left[..., :-1], (0, 1, 0, 0))
    left_dy = F.pad(left[..., 1:, :] - left[..., :-1, :], (0, 0, 0, 1))
    horizontal_weight = torch.exp(-5.0 * left_dx.abs()).detach()
    vertical_weight = torch.exp(-5.0 * left_dy.abs()).detach()

    def decode(theta: Tensor) -> Tensor:
        if theta.ndim != 2 or theta.shape[1] != CROP_HEIGHT * CROP_WIDTH:
            raise ValueError("stereo theta has the wrong shape")
        return initial_disparity + update_scale_px * theta.reshape(
            theta.shape[0], 1, CROP_HEIGHT, CROP_WIDTH
        )

    def factor_function(theta: Tensor) -> Tensor:
        disparity = decode(theta)
        coordinates = x_grid[: theta.shape[0]] - disparity[:, 0]
        sampled = horizontal_bilinear_sample(right[: theta.shape[0]], coordinates)
        photo = torch.sqrt((left[: theta.shape[0]] - sampled).square() + 1.0e-4) - 0.01
        disparity_dx = F.pad(disparity[..., 1:] - disparity[..., :-1], (0, 1, 0, 0))
        disparity_dy = F.pad(disparity[..., 1:, :] - disparity[..., :-1, :], (0, 0, 0, 1))
        tv = horizontal_weight[: theta.shape[0]] * torch.sqrt(disparity_dx.square() + 1.0e-4)
        tv = tv + vertical_weight[: theta.shape[0]] * torch.sqrt(disparity_dy.square() + 1.0e-4)
        trust = theta.reshape(theta.shape[0], 1, CROP_HEIGHT, CROP_WIDTH).square()
        mask = valid_crop[: theta.shape[0], None, None, None]
        return torch.cat(
            (
                hierarchical_grid_sketch(0.5 * photo.square() * mask, spec),
                hierarchical_grid_sketch(tv_weight * tv * mask, spec),
                hierarchical_grid_sketch(trust_weight * trust * mask, spec),
            ),
            dim=1,
        )

    initial_theta = torch.zeros(
        observation.crop_count,
        CROP_HEIGHT * CROP_WIDTH,
        device=device,
        dtype=torch.float32,
    )
    return factor_function, initial_theta, STEREO_BLOCK_ID.to(device), decode


def load_middlebury_query(
    observation: StereoSceneObservation,
) -> tuple[Tensor, Tensor]:
    """Load GT only in the evaluator after every method estimate is frozen."""

    disparity_np = read_pfm(observation.root / "disp0GT.pfm")
    mask_np = _decode_grayscale(observation.root / "mask0nocc.png")
    disparity = torch.as_tensor(
        disparity_np, device=observation.left_full.device, dtype=torch.float32
    )[None, None]
    mask = torch.as_tensor(mask_np > 0, device=observation.left_full.device)[None, None]
    finite = torch.isfinite(disparity)
    disparity = torch.where(finite, disparity, torch.zeros_like(disparity))
    mask = mask & finite
    disparity = F.interpolate(disparity, scale_factor=DOWNSAMPLE_SCALE, mode="bilinear", align_corners=False)
    disparity = disparity * DOWNSAMPLE_SCALE
    mask = F.interpolate(mask.float(), size=disparity.shape[-2:], mode="nearest").bool()
    gt_crops = []
    mask_crops = []
    for y0, x0 in observation.crop_origins:
        gt_crops.append(disparity[:, :, y0 : y0 + CROP_HEIGHT, x0 : x0 + CROP_WIDTH])
        mask_crops.append(mask[:, :, y0 : y0 + CROP_HEIGHT, x0 : x0 + CROP_WIDTH])
    return torch.cat(gt_crops, dim=0), torch.cat(mask_crops, dim=0)


__all__ = [
    "CROP_COLUMNS",
    "CROP_HEIGHT",
    "CROP_ROWS",
    "CROP_WIDTH",
    "DOWNSAMPLE_SCALE",
    "STEREO_BLOCK_ID",
    "StereoSceneObservation",
    "census_wta_disparity",
    "compile_middlebury_scene",
    "horizontal_bilinear_sample",
    "load_middlebury_query",
    "parse_ndisp",
    "read_pfm",
    "stereo_scene_factor_graph",
]
