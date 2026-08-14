import pytest
import torch

from stride.models.vision import hidden_from_attention_call


def test_attention_hook_accepts_positional_hidden_states():
    hidden = torch.randn(1, 4, 8)
    assert hidden_from_attention_call((hidden,), {}) is hidden


def test_attention_hook_accepts_keyword_hidden_states():
    hidden = torch.randn(1, 4, 8)
    assert hidden_from_attention_call((), {'hidden_states': hidden}) is hidden


def test_attention_hook_rejects_missing_hidden_states():
    with pytest.raises(RuntimeError, match='did not expose'):
        hidden_from_attention_call((), {})
