import torch

from stride.signals import (
    modality_calibrated_relevance,
    rank_agreement,
    residualize_position,
)


def test_position_residualization_reduces_coordinate_fit():
    y, x = torch.meshgrid(torch.linspace(0, 1, 8), torch.linspace(0, 1, 8), indexing="ij")
    coords = torch.stack((y.flatten(), x.flatten()), dim=-1)
    score = 0.2 + 0.8 * y.flatten()
    corrected, nuisance = residualize_position(score, coords, strength=1.0, ridge=1e-5)
    before = torch.corrcoef(torch.stack((score, y.flatten())))[0, 1].abs()
    after = torch.corrcoef(torch.stack((corrected, y.flatten())))[0, 1].abs()
    assert after < before
    assert nuisance.abs().mean() > 0


def test_modality_relevance_is_finite_for_single_text_token():
    vision = torch.randn(20, 16)
    text = torch.randn(1, 16)
    score = modality_calibrated_relevance(vision, text)
    assert score.shape == (20,)
    assert torch.isfinite(score).all()
    assert ((0 <= score) & (score <= 1)).all()


def test_rank_agreement_detects_matching_and_reversed_orders():
    score = torch.tensor([0.1, 0.4, 0.2, 0.8])
    assert torch.isclose(rank_agreement(score, score), torch.tensor(1.0))
    assert torch.isclose(rank_agreement(score, -score), torch.tensor(0.0))
