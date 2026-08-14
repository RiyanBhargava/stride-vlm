from __future__ import annotations

import torch
import torch.nn.functional as F

from .math_utils import cosine_matrix, normalize01, robust_standardize


def modality_relevance_components(
    vision: torch.Tensor,
    text: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and robustly calibrated query-relevance scores.

    Moment matching is deliberately used only to score/assign tokens. Returned
    VLM embeddings remain untouched, avoiding an inference-time distribution shift.
    """
    if text is None or text.numel() == 0:
        neutral = torch.full(
            (vision.shape[0],), 0.5, device=vision.device, dtype=vision.dtype
        )
        return neutral, neutral
    v_raw, t_raw = vision.float(), text.float()
    raw = cosine_matrix(v_raw, t_raw)
    v_cal = robust_standardize(v_raw)
    t_cal = robust_standardize(t_raw) if len(t_raw) > 1 else F.layer_norm(t_raw, (t_raw.shape[-1],))
    calibrated = cosine_matrix(v_cal, t_cal)
    # Log-mean-exp is stable and less hostage to one accidental token match than max.
    raw_score = torch.logsumexp(raw / 0.20, dim=-1) * 0.20
    cal_score = torch.logsumexp(calibrated / 0.20, dim=-1) * 0.20
    return (
        normalize01(raw_score).to(vision.dtype),
        normalize01(cal_score).to(vision.dtype),
    )


def modality_calibrated_relevance(
    vision: torch.Tensor,
    text: torch.Tensor | None,
    blend: float = 0.65,
) -> torch.Tensor:
    raw_score, calibrated_score = modality_relevance_components(vision, text)
    return normalize01(
        (1 - blend) * raw_score.float() + blend * calibrated_score.float()
    ).to(vision.dtype)


def rank_agreement(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Spearman-style agreement in [0, 1], used as a label-free trust signal."""
    if first.shape != second.shape:
        raise ValueError("rank_agreement inputs must have the same shape")
    if first.numel() < 2:
        return torch.ones((), device=first.device)
    first_rank = torch.argsort(torch.argsort(first.float(), stable=True), stable=True).float()
    second_rank = torch.argsort(torch.argsort(second.float(), stable=True), stable=True).float()
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = first_rank.norm() * second_rank.norm()
    correlation = (first_rank * second_rank).sum() / denominator.clamp_min(1e-8)
    return correlation.clamp(0, 1)


def svd_leverage(vision: torch.Tensor, rank: int = 32) -> torch.Tensor:
    x = vision.float() - vision.float().mean(0, keepdim=True)
    k = min(rank, x.shape[0] - 1, x.shape[1])
    if k < 1:
        return torch.ones(x.shape[0], device=x.device, dtype=vision.dtype)
    u = torch.linalg.svd(x, full_matrices=False).U[:, :k]
    return normalize01(u.square().sum(-1)).to(vision.dtype)


def global_distinctiveness(vision: torch.Tensor) -> torch.Tensor:
    """Cheap global residual signal used by the default real-time router.

    Tokens unlike the robust global center receive higher scores. This costs
    O(Nd), avoiding an unnecessary SVD in the inference-time relevance path.
    """
    center = vision.float().median(dim=0).values.unsqueeze(0)
    score = 1 - F.cosine_similarity(vision.float(), center, dim=-1)
    return normalize01(score).to(vision.dtype)


def local_novelty(vision: torch.Tensor, grid: tuple[int, int], radius: int = 1) -> torch.Tensor:
    h, w = grid
    if h * w != vision.shape[0]:
        raise ValueError("grid does not match tokens")
    z = F.normalize(vision.float(), dim=-1).reshape(h, w, -1)
    scores = torch.zeros((h, w), device=vision.device)
    counts = torch.zeros((h, w), device=vision.device)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            y0, y1 = max(0, -dy), min(h, h - dy)
            x0, x1 = max(0, -dx), min(w, w - dx)
            sim = (z[y0:y1, x0:x1] * z[y0 + dy : y1 + dy, x0 + dx : x1 + dx]).sum(-1)
            scores[y0:y1, x0:x1] += 1 - sim
            counts[y0:y1, x0:x1] += 1
    return normalize01((scores / counts.clamp_min(1)).reshape(-1)).to(vision.dtype)


def position_basis(coordinates: torch.Tensor) -> torch.Tensor:
    y, x = coordinates.float().unbind(-1)
    raster = 0.5 * (x + y)
    radial = torch.sqrt((x - 0.5).square() + (y - 0.5).square())
    return torch.stack(
        [torch.ones_like(x), x, y, x * y, x.square(), y.square(), raster, radial], dim=-1
    )


def residualize_position(
    score: torch.Tensor,
    coordinates: torch.Tensor,
    strength: float = 0.5,
    ridge: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the low-order component predictable from patch coordinates.

    This is an unlabeled, per-example nuisance projection rather than a learned
    correction curve. The intercept is preserved and `strength` shrinks the
    correction to avoid erasing real spatial semantics.
    """
    if strength <= 0 or len(score) < 9:
        return score, torch.zeros_like(score)
    basis = position_basis(coordinates)
    penalty = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype) * ridge
    penalty[0, 0] = 0
    target = score.float()
    coef = torch.linalg.solve(basis.T @ basis + penalty, basis.T @ target)
    fitted = basis @ coef
    nuisance = fitted - fitted.mean()
    corrected = target - strength * nuisance
    return normalize01(corrected).to(score.dtype), nuisance.to(score.dtype)
