from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .math_utils import grid_coordinates, infer_grid
from .types import RouteResult


def _select(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    score: torch.Tensor,
    grid: tuple[int, int] | None,
) -> RouteResult:
    coordinates = grid_coordinates(
        len(tokens), grid or infer_grid(len(tokens)), tokens.device
    )
    indices = torch.sort(indices.long()).values
    assignment = F.one_hot(indices, num_classes=len(tokens)).to(tokens.dtype)
    return RouteResult(
        tokens[indices], coordinates[indices], assignment, score, indices
    )


def dense(tokens: torch.Tensor, **_: object) -> RouteResult:
    count = len(tokens)
    coordinates = grid_coordinates(count, infer_grid(count), tokens.device)
    identity = torch.eye(count, device=tokens.device, dtype=tokens.dtype)
    score = torch.ones(count, device=tokens.device, dtype=tokens.dtype)
    return RouteResult(
        tokens,
        coordinates,
        identity,
        score,
        torch.arange(count, device=tokens.device),
    )


def random_pruning(
    tokens: torch.Tensor,
    budget: int,
    seed: int = 0,
    grid: tuple[int, int] | None = None,
    **_: object,
) -> RouteResult:
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    score = torch.rand(len(tokens), device=tokens.device, generator=generator)
    indices = torch.topk(score, min(budget, len(tokens))).indices
    return _select(tokens, indices, score, grid)


def _balanced_spatial_groups(
    count: int,
    grid: tuple[int, int],
    budget: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = grid
    if height * width != count:
        raise ValueError(f'grid {grid} does not contain {count} tokens')
    if not 1 <= budget <= count:
        raise ValueError('budget must be within the input token count')
    factors = [
        (rows, budget // rows)
        for rows in range(1, budget + 1)
        if budget % rows == 0
    ]
    target_ratio = width / max(height, 1)
    rows, columns = min(
        factors,
        key=lambda shape: abs(
            math.log((shape[1] / shape[0]) / target_ratio)
        ),
    )
    if rows <= height and columns <= width:
        row = torch.arange(height, device=device).repeat_interleave(width)
        column = torch.arange(width, device=device).repeat(height)
        row_group = torch.div(row * rows, height, rounding_mode='floor')
        column_group = torch.div(
            column * columns, width, rounding_mode='floor'
        )
        return row_group * columns + column_group, (rows, columns)
    groups = torch.div(
        torch.arange(count, device=device) * budget,
        count,
        rounding_mode='floor',
    )
    return groups, (1, budget)


def spatial_pooling(
    tokens: torch.Tensor,
    budget: int,
    grid: tuple[int, int] | None = None,
    **_: object,
) -> RouteResult:
    '''Training-free control that averages contiguous image regions.'''
    resolved = grid or infer_grid(len(tokens))
    requested = min(budget, len(tokens))
    groups, output_grid = _balanced_spatial_groups(
        len(tokens), resolved, requested, tokens.device
    )
    slots = int(groups.max().item()) + 1
    assignment = torch.zeros(
        (slots, len(tokens)), device=tokens.device, dtype=torch.float32
    )
    counts = torch.bincount(groups, minlength=slots).float()
    token_indices = torch.arange(len(tokens), device=tokens.device)
    assignment[groups, token_indices] = 1.0 / counts[groups]
    coordinates = grid_coordinates(len(tokens), resolved, tokens.device)
    merged = assignment.to(tokens.dtype) @ tokens
    merged_coordinates = assignment @ coordinates.float()
    return RouteResult(
        merged,
        merged_coordinates.to(tokens.dtype),
        assignment.to(tokens.dtype),
        torch.ones(len(tokens), device=tokens.device, dtype=tokens.dtype),
        diagnostics={
            'method_family': 'control',
            'transport_mode': 'uniform_spatial_pool',
            'output_grid': list(output_grid),
        },
    )
