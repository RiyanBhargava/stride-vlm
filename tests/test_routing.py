import torch

from stride import RouterConfig, RoutingContext, route
from stride.routing import (
    _calibrated_concept_scores,
    _intent_stride_indices,
    _question_intent,
    divprune_indices,
    otprune_indices,
    vispruner_indices,
)


def context(seed: int = 7, count: int = 64, width: int = 24):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(count, width, generator=generator)
    features = torch.randn(count, width, generator=generator)
    salience = torch.rand(count, generator=generator)
    text = torch.randn(8, width, generator=generator)
    return RoutingContext(
        tokens=tokens,
        vision_features=features,
        vision_salience=salience,
        text_tokens=text,
        semantic_visual_tokens=tokens,
        routing_prompt='What color is the bicycle?',
        grid=(8, 8),
        salience_source='test',
        alignment_source='input_embedding_test',
        global_vision_feature=torch.randn(width, generator=generator),
        projector=torch.nn.Identity(),
    )


def test_all_methods_return_requested_token_count():
    sample = context()
    config = RouterConfig(budget=12)
    for method in (
        'random', 'divprune', 'otprune',
        'vispruner', 'visionzip', 'stride',
    ):
        result = route(method, sample, config)
        assert result.tokens.shape == (12, 24)
        assert result.assignment.shape == (12, 64)


def test_spatial_pool_has_exact_budget():
    sample = context()
    result = route('pool', sample, RouterConfig(budget=13))
    assert result.tokens.shape == (13, 24)


def test_divprune_starts_with_most_isolated_token():
    features = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    selected = divprune_indices(features, 1)
    assert selected.tolist() == [2]


def test_vispruner_has_exact_budget_and_unique_indices():
    sample = context()
    selected = vispruner_indices(
        sample.vision_features, sample.vision_salience, 13, 0.5
    )
    assert len(selected) == 13
    assert len(torch.unique(selected)) == 13


def test_visionzip_has_global_dominant_and_contextual_slots():
    result = route('visionzip', context(), RouterConfig(budget=16))
    assert result.selected_indices[0].item() == -1
    assert result.diagnostics['global_tokens'] == 1
    assert result.diagnostics['dominant_tokens'] > 0
    assert result.diagnostics['contextual_tokens'] > 0
    assert (result.assignment.sum(dim=1) > 1).any()


def test_no_semantics_still_uses_intent_expert():
    sample = context()
    result = route(
        'stride', sample, RouterConfig(budget=16, stride_use_semantics=False)
    )
    assert len(torch.unique(result.selected_indices)) == 16
    assert not result.diagnostics['semantic_active']
    assert result.diagnostics['selected_expert'] == 'salience'
    assert result.diagnostics['emits_original_tokens']
    assert result.diagnostics['decoder_probe_layers'] == 0


def test_query_intents_and_experts_are_distinct():
    assert _question_intent('Is there a bus in the image?') == 'existence'
    assert _question_intent('What word is written on the sign?') == 'ocr'
    assert _question_intent('Which answer?\n(A) one\n(B) two') == 'choice'
    sample = context(count=64)
    scores = torch.rand(64, 3, generator=torch.Generator().manual_seed(11))
    weights = torch.tensor([0.5, 0.3, 0.2])
    _, _, existence = _intent_stride_indices(
        sample.tokens, sample.vision_features, sample.semantic_visual_tokens,
        sample.vision_salience,
        scores, weights,
        'Is there a bus?', RouterConfig(budget=8),
    )
    _, _, choice = _intent_stride_indices(
        sample.tokens, sample.vision_features, sample.semantic_visual_tokens,
        sample.vision_salience,
        scores, weights,
        'Which answer? (A) one (B) two', RouterConfig(budget=8),
    )
    assert existence['selected_expert'] == 'diversity'
    assert choice['selected_expert'] == 'salience'
    assert choice['residual_space_active']


def test_joint_kernel_preserves_unique_tokens():
    sample = context(seed=9)
    result = route(
        'stride',
        sample,
        RouterConfig(budget=16),
    )
    assert len(torch.unique(result.selected_indices)) == 16


def test_otprune_is_deterministic_and_exact_budget():
    sample = context(seed=21)
    ot = otprune_indices(sample.tokens, 13)
    assert len(torch.unique(ot)) == 13
    assert torch.equal(ot, otprune_indices(sample.tokens, 13))


def test_calibrated_concepts_are_finite_normalized_and_selective():
    sample = context()
    scores, weights, contrast = _calibrated_concept_scores(
        sample.tokens, sample.text_tokens, 1.0
    )
    assert scores.shape == (64, 8)
    assert torch.isfinite(scores).all()
    assert scores.min() >= 0
    assert scores.max() <= 1
    assert torch.isfinite(weights).all()
    assert weights.sum() <= 1.00001
    assert contrast >= 0
