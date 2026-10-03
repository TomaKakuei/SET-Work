"""Structured problem adapters used by the stage-2 CSN experiments."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch
from torch import Tensor
import torch.nn.functional as F

from .problems import (
    ResidualProblem,
    ba_problem as _ba_problem,
    hpatches_problem as _hpatches_problem,
    se3_problem as _se3_problem,
)


def _edge_tensor(pairs: set[tuple[int, int]]) -> Tensor:
    ordered = sorted((min(a, b), max(a, b)) for a, b in pairs if a != b)
    ordered = sorted(set(ordered))
    return torch.tensor(ordered, dtype=torch.long).T if ordered else torch.empty(2, 0, dtype=torch.long)


def _attach(problem: ResidualProblem, edges: Tensor, **metadata) -> ResidualProblem:
    merged = dict(problem.metadata or {})
    merged.update(metadata)
    merged["block_edges"] = edges
    return replace(problem, metadata=merged)


def hpatches_problem(*args, **kwargs):
    problem, decode = _hpatches_problem(*args, **kwargs)
    return _attach(problem, torch.empty(2, 0, dtype=torch.long), structure="one homography block"), decode


def ba_problem(graph):
    problem = _ba_problem(graph)
    pairs: set[tuple[int, int]] = set()
    camera_blocks: dict[int, tuple[int, int]] = {}
    for camera in range(1, graph.camera_count):
        start = 6 * (camera - 1)
        blocks = (int(problem.block_id[start]), int(problem.block_id[start + 3]))
        camera_blocks[camera] = blocks
        pairs.add(blocks)
    landmark_start = 6 * (graph.camera_count - 1)
    landmark_blocks = {
        landmark: int(problem.block_id[landmark_start if landmark == 0 else landmark_start + 2 + 3 * (landmark - 1)])
        for landmark in range(graph.landmark_count)
    }
    for camera, landmark in zip(graph.observation_camera.tolist(), graph.observation_landmark.tolist()):
        for camera_block in camera_blocks.get(int(camera), ()):
            pairs.add((camera_block, landmark_blocks[int(landmark)]))
    return _attach(
        problem,
        _edge_tensor(pairs),
        structure="camera-landmark incidence",
        schur_split=landmark_start,
    )


def bal_problem(graph):
    """Native two-coordinate residual adapter for a compiled BAL subgraph."""
    initial = graph.initial_theta[0].double()
    selected = [graph.observation_view == view for view in (0, 1)]

    def raw(x: Tensor, view: int) -> Tensor:
        rotation, translation, landmarks = graph.decode(x[None].to(torch.float32))
        camera_index = graph.observation_camera[selected[view]].to(x.device)
        landmark_index = graph.observation_landmark[selected[view]].to(x.device)
        point = landmarks[0, landmark_index]
        camera_point = torch.einsum("oij,oj->oi", rotation[0, camera_index], point)
        camera_point = camera_point + translation[0, camera_index]
        # BAL cameras look down -Z: -P.xy/P.z = P.xy/(-P.z).
        normalized = camera_point[:, :2] / (-camera_point[:, 2:]).clamp_min(1.0e-3)
        intrinsics = graph.fixed_intrinsics.to(x)[camera_index]
        radius2 = normalized.square().sum(dim=-1)
        radial = 1.0 + intrinsics[:, 1] * radius2 + intrinsics[:, 2] * radius2.square()
        predicted = intrinsics[:, :1] * radial[:, None] * normalized
        observed = graph.observed_xy.to(x)[selected[view]]
        return (predicted - observed) / intrinsics[:, :1].abs().clamp_min(1.0)

    delta = float(graph.reprojection_delta)

    def cost(x: Tensor, view: int) -> Tensor:
        squared = raw(x, view).square().sum(dim=-1)
        return (delta * delta * (torch.sqrt(1.0 + squared / (delta * delta)) - 1.0)).sum()

    def weights(x: Tensor, view: int) -> Tensor:
        squared = raw(x, view).square().sum(dim=-1, keepdim=True)
        return torch.rsqrt(1.0 + squared / (delta * delta))

    legacy = tuple(
        (lambda x, fn=graph.factor_function(view=view): fn(x.to(torch.float32)).to(torch.float64))
        for view in ("even", "odd")
    )
    problem = ResidualProblem(
        initial,
        graph.block_id.clone(),
        raw,
        cost,
        weights,
        legacy,
        metadata={
            "family": "bal",
            "projection_schema": "bal-negative-z-v2",
            "gauge": "camera0 fixed; released intrinsics fixed",
            "robust": "frozen IRLS pseudo-Huber",
        },
    )
    pairs: set[tuple[int, int]] = set()
    camera_blocks = {}
    for camera in range(1, graph.camera_count):
        start = 6 * (camera - 1)
        camera_blocks[camera] = (int(problem.block_id[start]), int(problem.block_id[start + 3]))
        pairs.add(camera_blocks[camera])
    landmark_start = 6 * (graph.camera_count - 1)
    landmark_blocks = {
        landmark: int(problem.block_id[landmark_start + 3 * landmark])
        for landmark in range(graph.landmark_count)
    }
    for camera, landmark in zip(graph.observation_camera.tolist(), graph.observation_landmark.tolist()):
        for camera_block in camera_blocks.get(int(camera), ()):
            pairs.add((camera_block, landmark_blocks[int(landmark)]))
    return _attach(
        problem,
        _edge_tensor(pairs),
        structure="BAL camera-landmark incidence",
        schur_split=landmark_start,
    )


def se3_problem(graph):
    problem = _se3_problem(graph)
    pairs: set[tuple[int, int]] = set()
    node_blocks: dict[int, tuple[int, int]] = {}
    for node in range(1, graph.node_count):
        start = 6 * (node - 1)
        blocks = (int(problem.block_id[start]), int(problem.block_id[start + 3]))
        node_blocks[node] = blocks
        pairs.add(blocks)
    for source, target in graph.edges.tolist():
        for left in node_blocks.get(int(source), ()):
            for right in node_blocks.get(int(target), ()):
                pairs.add((left, right))
    return _attach(problem, _edge_tensor(pairs), structure="SE3 factor-variable incidence")


def _refined_blocks(original: Tensor, maximum_width: int = 32) -> Tensor:
    output = torch.empty_like(original)
    next_block = 0
    for block in torch.unique(original, sorted=True).tolist():
        indices = (original == block).nonzero().flatten()
        for chunk in indices.split(maximum_width):
            output[chunk] = next_block
            next_block += 1
    return output


def scalar_factor_problem(
    factor_graph,
    *,
    family: str,
    factor_function: Callable[[Tensor], Tensor] | None = None,
    initial: Tensor | None = None,
    block_id: Tensor | None = None,
) -> ResidualProblem:
    """Wrap legacy nonnegative factors for planned boundary/stress tasks.

    This adapter is intentionally labelled scalarized: it tests task coverage
    and support/query alignment, while native-vector claims remain restricted
    to the dedicated HPatches, BA, and SE3 adapters.
    """
    function = factor_function or factor_graph.factor_values
    if initial is None:
        value = factor_graph.initial_theta()
        initial = value[0]
    if block_id is None:
        candidate = factor_graph.block_id()
        block_id = _refined_blocks(candidate)
    else:
        _, block_id = torch.unique(block_id.long(), sorted=True, return_inverse=True)
    factor_count = int(function(initial[None]).shape[1])
    selected = [torch.arange(factor_count) % 2 == view for view in (0, 1)]

    def raw(x: Tensor, view: int) -> Tensor:
        values = function(x[None])[0, selected[view].to(x.device)]
        return torch.sqrt(2.0 * values.clamp_min(0.0) + 1.0e-18)[:, None]

    def cost(x: Tensor, view: int) -> Tensor:
        return function(x[None])[0, selected[view].to(x.device)].sum()

    def weights(x: Tensor, view: int) -> Tensor:
        return x.new_ones(int(selected[view].sum()), 1)

    blocks = int(block_id.max()) + 1
    edges = _edge_tensor({(index, index + 1) for index in range(blocks - 1)})
    return ResidualProblem(
        initial.detach().clone(), block_id.long(), raw, cost, weights,
        (
            lambda x: torch.stack([function(row[None])[0] for row in x]),
            lambda x: torch.stack([function(row[None])[0] for row in x]),
        ),
        metadata={
            "family": family,
            "geometry": "scalarized nonnegative factors",
            "block_edges": edges,
            "factor_count": factor_count,
        },
    )


def hpatches_multiframe_problem(graph) -> ResidualProblem:
    return scalar_factor_problem(
        graph,
        family="hpatches_multiframe",
        factor_function=graph.factor_function,
        initial=graph.initial_theta[0],
        block_id=graph.block_id,
    )


def known_blur_problem(observation: Tensor, kernel: Tensor, *, regularization: float = 2.0e-3) -> ResidualProblem:
    """Native linear residual graph for controlled nonblind restoration."""
    if observation.ndim != 2 or kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("observation and square kernel must be two dimensional")
    height, width = observation.shape
    radius = kernel.shape[0] // 2
    kernel4 = kernel.to(observation)[None, None]
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    checker = (xx + yy) % 2

    def decode(x: Tensor) -> Tensor:
        return x.reshape(1, 1, height, width)

    def components(x: Tensor):
        image = decode(x)
        prediction = F.conv2d(F.pad(image, (radius,) * 4, mode="reflect"), kernel4)[0, 0]
        data = prediction - observation
        dx = math_sqrt(regularization) * (image[0, 0, :, 1:] - image[0, 0, :, :-1])
        dy = math_sqrt(regularization) * (image[0, 0, 1:, :] - image[0, 0, :-1, :])
        return data, dx, dy

    def raw(x: Tensor, view: int) -> Tensor:
        data, dx, dy = components(x)
        pieces = [data[checker == view], dx[:, view::2].reshape(-1), dy[view::2, :].reshape(-1)]
        return torch.cat(pieces)[:, None]

    def cost(x: Tensor, view: int) -> Tensor:
        return 0.5 * raw(x, view).square().sum()

    lengths = [raw(observation.flatten(), view).shape[0] for view in (0, 1)]

    def weights(x: Tensor, view: int) -> Tensor:
        return x.new_ones(lengths[view], 1)

    tile = 8
    block_grid_width = (width + tile - 1) // tile
    block_id = ((yy // tile) * block_grid_width + xx // tile).flatten().long()
    rows = (height + tile - 1) // tile
    cols = block_grid_width
    pairs = set()
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if row + 1 < rows:
                pairs.add((index, index + cols))
            if col + 1 < cols:
                pairs.add((index, index + 1))
    return ResidualProblem(
        observation.flatten().detach().clone(), block_id, raw, cost, weights,
        (
            lambda x: torch.stack([cost(row, 0) for row in x])[:, None],
            lambda x: torch.stack([cost(row, 1) for row in x])[:, None],
        ),
        expand_function=lambda x: x.reshape(height, width),
        metadata={
            "family": "known_blur", "geometry": "native linear convolution residual",
            "block_edges": _edge_tensor(pairs), "shape": [height, width],
        },
    )


def math_sqrt(value: float) -> float:
    return float(value) ** 0.5


__all__ = [
    "ba_problem", "bal_problem", "hpatches_multiframe_problem", "hpatches_problem",
    "known_blur_problem", "scalar_factor_problem", "se3_problem",
]
