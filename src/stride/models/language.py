from __future__ import annotations

from typing import Any

import torch


def _resolve_decoder_layers(model: torch.nn.Module):
    paths = (
        ('language_model', 'model', 'layers'),
        ('model', 'language_model', 'model', 'layers'),
        ('language_model', 'layers'),
        ('model', 'layers'),
    )
    for path in paths:
        value: Any = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None and len(value):
            return value
    raise RuntimeError('could not locate frozen decoder layers')


@torch.inference_mode()
def first_layer_value_space(
    model: torch.nn.Module,
    visual_tokens: torch.Tensor,
    text_tokens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    '''Map image and text embeddings through the same frozen value projection.'''
    if text_tokens is None or text_tokens.numel() == 0:
        return visual_tokens.float(), None, 'missing_text'
    layer = _resolve_decoder_layers(model)[0]
    norm = getattr(layer, 'input_layernorm', None)
    attention = getattr(layer, 'self_attn', None)
    value_projection = getattr(attention, 'v_proj', None)
    if norm is None or value_projection is None:
        raise RuntimeError('first decoder layer does not expose norm and v_proj')
    values = torch.cat((visual_tokens, text_tokens.to(visual_tokens.dtype)), dim=0)
    values = value_projection(norm(values))
    split = len(visual_tokens)
    return (
        values[:split].float(),
        values[split:].float(),
        'frozen_decoder_layer0_value_space_no_rope',
    )
