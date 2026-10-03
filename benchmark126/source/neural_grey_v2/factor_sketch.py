"""Task-agnostic, objective-preserving sketches for dense factor fields.

Dense visual objectives naturally expose one scalar factor per pixel.  Passing
all of them to DA-DFS makes the gradient bank ``[B, F, P]`` and the pairwise
factor geometry ``[B, K, F, F]`` unnecessarily large.  This module compiles a
dense non-negative factor field into a fixed multiscale set of regional sums.

Every level is an exact partition of the input grid.  Averaging across levels
therefore preserves the total scalar objective while exposing coarse and fine
spatial evidence to the same task-ID-free solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class GridFactorSketchSpec:
    """Description of a deterministic hierarchical grid sketch."""

    height: int
    width: int
    levels: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (4, 4))

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError("height and width must be positive")
        if not self.levels:
            raise ValueError("at least one grid level is required")
        for rows, columns in self.levels:
            if rows < 1 or columns < 1:
                raise ValueError("grid dimensions must be positive")
            if rows > self.height or columns > self.width:
                raise ValueError("a grid level cannot exceed the factor field")

    @property
    def factors(self) -> int:
        return sum(rows * columns for rows, columns in self.levels)


@lru_cache(maxsize=128)
def _cpu_region_ids(
    height: int,
    width: int,
    levels: tuple[tuple[int, int], ...],
) -> tuple[Tensor, ...]:
    """Build balanced integer partitions once on CPU for device-independent use."""

    y = torch.arange(height, dtype=torch.long)
    x = torch.arange(width, dtype=torch.long)
    outputs = []
    for rows, columns in levels:
        row_id = torch.div(y * rows, height, rounding_mode="floor")
        column_id = torch.div(x * columns, width, rounding_mode="floor")
        outputs.append((row_id[:, None] * columns + column_id[None, :]).flatten())
    return tuple(outputs)


def hierarchical_grid_sketch(
    values: Tensor,
    spec: GridFactorSketchSpec | None = None,
    *,
    levels: Iterable[tuple[int, int]] = ((1, 1), (2, 2), (4, 4)),
) -> Tensor:
    """Compress ``[B,H,W]`` or ``[B,C,H,W]`` factor maps into regional sums.

    Channels are treated as independent fields and concatenated.  Each level is
    divided by the number of levels, so summing the returned factors equals the
    sum of the input field (up to floating-point reduction order).
    """

    if values.ndim == 3:
        values = values[:, None]
    if values.ndim != 4:
        raise ValueError("values must have shape [B,H,W] or [B,C,H,W]")
    batch, channels, height, width = values.shape
    if spec is None:
        spec = GridFactorSketchSpec(height, width, tuple(levels))
    if (height, width) != (spec.height, spec.width):
        raise ValueError("values and sketch spec have different spatial shapes")

    flat = values.reshape(batch * channels, height * width)
    outputs = []
    level_count = len(spec.levels)
    for (rows, columns), cpu_ids in zip(
        spec.levels,
        _cpu_region_ids(height, width, spec.levels),
    ):
        region_ids = cpu_ids.to(device=values.device)
        index = region_ids[None].expand(batch * channels, -1)
        pooled = values.new_zeros(batch * channels, rows * columns)
        pooled.scatter_add_(1, index, flat)
        outputs.append(pooled / level_count)
    return torch.cat(outputs, dim=-1).reshape(batch, channels * spec.factors)


__all__ = ["GridFactorSketchSpec", "hierarchical_grid_sketch"]
