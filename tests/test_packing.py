import torch

from stride.models.packing import position_preserving_ids


def test_selected_visual_tokens_retain_dense_positions():
    ids = torch.tensor([10, 99, 99, 99, 99, 11, 12])
    selected = torch.tensor([0, 3])
    coordinates = torch.zeros(2, 2)
    positions = position_preserving_ids(ids, 99, selected, coordinates, (2, 2))
    assert positions.tolist() == [0, 1, 4, 5, 6]


def test_pooled_tokens_use_nearest_original_grid_positions():
    ids = torch.tensor([99, 99, 99, 99, 7])
    coordinates = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    positions = position_preserving_ids(ids, 99, None, coordinates, (2, 2))
    assert positions.tolist() == [0, 3, 4]
