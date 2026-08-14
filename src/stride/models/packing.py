from __future__ import annotations

import torch


def position_preserving_ids(
    input_ids: torch.Tensor,
    image_token_id: int,
    selected_indices: torch.Tensor | None,
    coordinates: torch.Tensor,
    grid: tuple[int, int],
) -> torch.Tensor:
    '''Keep routed patches and following text at their original dense positions.'''
    image_positions = torch.nonzero(
        input_ids == image_token_id, as_tuple=False
    ).flatten()
    if len(image_positions) == 0:
        raise ValueError('processor output contains no image placeholder tokens')
    first, last = int(image_positions[0]), int(image_positions[-1])
    if selected_indices is None:
        height, width = grid
        rows = (coordinates[:, 0].float() * max(height - 1, 0)).round().long()
        columns = (coordinates[:, 1].float() * max(width - 1, 0)).round().long()
        selected_indices = rows * width + columns
    visual = first + selected_indices.to(device=input_ids.device, dtype=torch.long)
    prefix = torch.arange(first, device=input_ids.device, dtype=torch.long)
    suffix = torch.arange(
        last + 1, len(input_ids), device=input_ids.device, dtype=torch.long
    )
    return torch.cat((prefix, visual, suffix))
