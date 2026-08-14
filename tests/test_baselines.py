import torch

from stride.baselines import dense, random_pruning, spatial_pooling


def test_dense_is_identity():
    tokens = torch.randn(36, 16)
    result = dense(tokens)
    torch.testing.assert_close(result.tokens, tokens)
    torch.testing.assert_close(result.assignment, torch.eye(36))


def test_random_seed_is_reproducible():
    tokens = torch.randn(36, 8)
    first = random_pruning(tokens, budget=5, seed=7, grid=(6, 6))
    second = random_pruning(tokens, budget=5, seed=7, grid=(6, 6))
    torch.testing.assert_close(first.tokens, second.tokens)


def test_spatial_pooling_preserves_cell_means():
    tokens = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    result = spatial_pooling(tokens, budget=4, grid=(4, 4))
    expected = torch.tensor([[2.5], [4.5], [10.5], [12.5]])
    torch.testing.assert_close(result.tokens, expected)
    torch.testing.assert_close(result.assignment.sum(1), torch.ones(4))


def test_spatial_pooling_honors_nonfactorable_budget():
    tokens = torch.randn(64, 8)
    result = spatial_pooling(tokens, budget=13, grid=(8, 8))
    assert result.tokens.shape == (13, 8)
