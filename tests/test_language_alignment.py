from types import SimpleNamespace

import pytest
import torch

from stride.models.language import first_layer_value_space


class FakeAttention(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.v_proj = torch.nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.v_proj.weight.copy_(torch.eye(width))


class FakeBlock(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.self_attn = FakeAttention(width)


def fake_model(width: int = 4):
    layers = torch.nn.ModuleList([FakeBlock(width)])
    return SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))


def test_first_layer_value_space_maps_image_and_text_together():
    visual = torch.randn(6, 4)
    text = torch.randn(3, 4)
    mapped_visual, mapped_text, source = first_layer_value_space(
        fake_model(), visual, text
    )
    assert source == 'frozen_decoder_layer0_value_space_no_rope'
    assert torch.allclose(mapped_visual, visual.float())
    assert torch.allclose(mapped_text, text.float())


def test_first_layer_value_space_handles_missing_text_without_model_lookup():
    visual = torch.randn(6, 4)
    mapped_visual, mapped_text, source = first_layer_value_space(
        SimpleNamespace(), visual, None
    )
    assert source == 'missing_text'
    assert mapped_text is None
    assert torch.allclose(mapped_visual, visual.float())


def test_first_layer_value_space_rejects_incompatible_decoder():
    bad_layer = SimpleNamespace(input_layernorm=torch.nn.Identity(), self_attn=SimpleNamespace())
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[bad_layer]))
    )
    with pytest.raises(RuntimeError, match='does not expose'):
        first_layer_value_space(model, torch.randn(2, 4), torch.randn(1, 4))
