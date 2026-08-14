from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normalize01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    lo = torch.quantile(x.float(), 0.05)
    hi = torch.quantile(x.float(), 0.95)
    return ((x - lo) / (hi - lo).clamp_min(eps)).clamp(0, 1).to(x.dtype)


def robust_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Featurewise median/MAD normalization with an RMS fallback."""
    xf = x.float()
    median = xf.median(dim=0).values
    mad = (xf - median).abs().median(dim=0).values * 1.4826
    rms = (xf - median).square().mean(dim=0).sqrt()
    scale = torch.where(mad > eps, mad, rms.clamp_min(eps))
    return ((xf - median) / scale).to(x.dtype)


def infer_grid(n: int, aspect_ratio: float = 1.0) -> tuple[int, int]:
    if n < 1:
        raise ValueError("token count must be positive")
    target_h = math.sqrt(n / max(aspect_ratio, 1e-6))
    factors = [(h, n // h) for h in range(1, int(math.sqrt(n)) + 1) if n % h == 0]
    if not factors:
        return 1, n
    return min(factors, key=lambda hw: abs(hw[0] - target_h))


def grid_coordinates(n: int, grid: tuple[int, int] | None, device: torch.device) -> torch.Tensor:
    h, w = grid or infer_grid(n)
    if h * w != n:
        raise ValueError(f"grid {h}x{w} does not match {n} tokens")
    ys = torch.linspace(0, 1, h, device=device)
    xs = torch.linspace(0, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)


def effective_rank(x: torch.Tensor, rank: int | None = None) -> torch.Tensor:
    s = torch.linalg.svdvals(x.float() - x.float().mean(0, keepdim=True))
    if rank is not None:
        s = s[:rank]
    p = s.square()
    p = p / p.sum().clamp_min(1e-8)
    return torch.exp(-(p * p.clamp_min(1e-8).log()).sum())


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).T

