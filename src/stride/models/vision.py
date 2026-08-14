from __future__ import annotations

from collections.abc import Callable

import torch


def hidden_from_attention_call(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> torch.Tensor:
    '''Support Transformers versions that pass hidden states by position or name.'''
    value = args[0] if args else kwargs.get('hidden_states')
    if not isinstance(value, torch.Tensor):
        raise RuntimeError('vision attention call did not expose hidden_states')
    return value


def attention_salience(
    attention_module: torch.nn.Module,
    hidden_states: torch.Tensor,
    has_cls_token: bool,
) -> torch.Tensor:
    '''Recompute frozen vision attention without changing the attention backend.'''
    query = attention_module.q_proj(hidden_states)
    key = attention_module.k_proj(hidden_states)
    heads = int(
        getattr(
            attention_module,
            'num_heads',
            getattr(attention_module, 'num_attention_heads', 1),
        )
    )
    head_dim = query.shape[-1] // heads
    query = query.view(query.shape[0], query.shape[1], heads, head_dim).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], heads, head_dim).transpose(1, 2)
    scale = float(getattr(attention_module, 'scale', head_dim**-0.5))
    probability = torch.softmax(
        (query.float() @ key.float().transpose(-1, -2)) * scale,
        dim=-1,
    )
    if has_cls_token:
        return probability[:, :, 0, 1:].mean(dim=1)
    return probability.mean(dim=(1, 2))


def vision_routing_tensors(
    vision_tower: torch.nn.Module,
    pixel_values: torch.Tensor,
    feature_layer: int,
    has_cls_token: bool,
    projector: Callable[[torch.Tensor], torch.Tensor],
    projector_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, str]:
    '''Return projected patches, pre-projector patches, CLS, and salience.'''
    layers = vision_tower.vision_model.encoder.layers
    layer_index = feature_layer % len(layers)
    captured: dict[str, torch.Tensor] = {}

    def save_attention_input(_module, args, kwargs) -> None:
        captured['hidden'] = hidden_from_attention_call(args, kwargs)

    handle = layers[layer_index].self_attn.register_forward_pre_hook(
        save_attention_input, with_kwargs=True
    )
    try:
        outputs = vision_tower(pixel_values, output_hidden_states=True)
    finally:
        handle.remove()
    selected = outputs.hidden_states[feature_layer]
    global_feature = selected[:, 0] if has_cls_token else None
    if has_cls_token:
        selected = selected[:, 1:]
        source = f'vision_cls_attention_layer_{layer_index}'
    else:
        source = f'vision_attention_centrality_layer_{layer_index}'
    salience = attention_salience(
        layers[layer_index].self_attn,
        captured['hidden'],
        has_cls_token,
    )
    projected = projector(selected) / projector_scale
    return projected[0], selected[0], salience[0], (
        global_feature[0] if global_feature is not None else None
    ), source
