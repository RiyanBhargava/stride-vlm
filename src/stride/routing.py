from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F

from .baselines import dense, random_pruning, spatial_pooling
from .config import RouterConfig
from .math_utils import grid_coordinates, infer_grid, normalize01
from .types import RouteResult, RoutingContext


ATTENTION_METHODS = frozenset({'vispruner', 'visionzip', 'stride'})
SEMANTIC_METHODS = frozenset({'stride'})
METHODS = (
    'dense',
    'random',
    'pool',
    'divprune',
    'otprune',
    'vispruner',
    'visionzip',
    'stride',
)


def needs_vision_attention(method: str) -> bool:
    return method in ATTENTION_METHODS


def needs_semantic_alignment(method: str) -> bool:
    return method in SEMANTIC_METHODS


def _selected_result(
    context: RoutingContext,
    indices: torch.Tensor,
    score: torch.Tensor,
    diagnostics: dict[str, object],
) -> RouteResult:
    indices = torch.sort(indices.long()).values
    n = len(context.tokens)
    assignment = F.one_hot(indices, num_classes=n).to(context.tokens.dtype)
    grid = context.grid or infer_grid(n)
    coords = grid_coordinates(n, grid, context.tokens.device)
    return RouteResult(
        context.tokens[indices],
        coords[indices],
        assignment,
        score.to(context.tokens.dtype),
        indices,
        diagnostics=diagnostics,
    )


def _compressed_result(
    context: RoutingContext,
    hidden: torch.Tensor,
    assignment: torch.Tensor,
    importance: torch.Tensor,
    anchors: torch.Tensor,
    diagnostics: dict[str, object],
) -> RouteResult:
    if context.projector is None:
        if hidden.shape[-1] != context.tokens.shape[-1]:
            raise RuntimeError('compressed routing requires the VLM projector')
        tokens = hidden
    else:
        tokens = context.projector(hidden)
    patch_coords = grid_coordinates(
        len(context.tokens),
        context.grid or infer_grid(len(context.tokens)),
        context.tokens.device,
    ).to(assignment.dtype)
    mass = assignment.clamp_min(0).sum(dim=1, keepdim=True)
    coordinates = assignment @ patch_coords
    coordinates = torch.where(
        mass > 0,
        coordinates / mass.clamp_min(1e-8),
        torch.full_like(coordinates, 0.5),
    )
    row_weight = torch.arange(
        1, len(assignment) + 1, device=assignment.device, dtype=assignment.dtype
    )[:, None]
    column_weight = torch.arange(
        1, assignment.shape[1] + 1,
        device=assignment.device,
        dtype=assignment.dtype,
    )[None, :]
    diagnostics = dict(diagnostics)
    diagnostics['route_token_sum'] = float(tokens.float().sum().item())
    diagnostics['route_assignment_index_sum'] = float(
        (assignment.float() * row_weight.float() * column_weight.float()).sum().item()
    )
    return RouteResult(
        tokens=tokens,
        coordinates=coordinates,
        assignment=assignment.to(tokens.dtype),
        importance=importance.to(tokens.dtype),
        selected_indices=anchors,
        diagnostics=diagnostics,
    )


def _cosine_distance(features: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features.float(), dim=-1)
    return (1 - normalized @ normalized.T).clamp_min(0)


def _normalized_salience(salience: torch.Tensor) -> torch.Tensor:
    value = salience.float().flatten().clamp_min(0)
    if value.numel() == 0:
        return value
    return (value + 1e-8) / (value.max() + 1e-8)


def divprune_indices(features: torch.Tensor, budget: int) -> torch.Tensor:
    """Paper-faithful DivPrune Algorithm 1 at decoder layer zero."""
    n = len(features)
    k = min(max(1, budget), n)
    if k == n:
        return torch.arange(n, device=features.device)
    distance = _cosine_distance(features)
    no_self = distance.clone()
    no_self.fill_diagonal_(float('inf'))
    first = int(no_self.min(dim=1).values.argmax().item())
    chosen = torch.empty(k, device=features.device, dtype=torch.long)
    chosen[0] = first
    minimum = distance[first].clone()
    minimum[first] = -1
    for position in range(1, k):
        index = int(minimum.argmax().item())
        chosen[position] = index
        minimum = torch.minimum(minimum, distance[index])
        minimum[chosen[: position + 1]] = -1
    return chosen


def _dpp_greedy(
    kernel: torch.Tensor,
    budget: int,
    preselected: torch.Tensor | None = None,
) -> torch.Tensor:
    '''Stable Cholesky-greedy determinant maximization.'''
    n = len(kernel)
    k = min(max(1, budget), n)
    cis = torch.zeros((k, n), device=kernel.device, dtype=torch.float32)
    residual = torch.diag(kernel.float()).clone()
    chosen = torch.empty(k, device=kernel.device, dtype=torch.long)
    selected = torch.zeros(n, device=kernel.device, dtype=torch.bool)
    protected = (
        preselected.long()
        if preselected is not None and preselected.numel()
        else torch.empty(0, device=kernel.device, dtype=torch.long)
    )
    protected = protected[:k]
    if len(torch.unique(protected)) != len(protected):
        raise ValueError('preselected DPP anchors must be unique')
    for step in range(k):
        residual[selected] = -float('inf')
        index = (
            int(protected[step].item())
            if step < len(protected)
            else int(residual.argmax().item())
        )
        chosen[step] = index
        selected[index] = True
        if step + 1 == k:
            break
        previous = cis[:step, index]
        pivot = residual[index].clamp_min(1e-12).sqrt()
        correction = (
            previous.unsqueeze(0) @ cis[:step]
        ).squeeze(0) if step else torch.zeros(n, device=kernel.device)
        row = (kernel[index].float() - correction) / pivot
        cis[step] = row
        residual = (residual - row.square()).clamp_min(0)
    return chosen


def _ot_kernel(features: torch.Tensor, gamma: float) -> torch.Tensor:
    '''Official OTPrune layer-zero kernel construction.'''
    values = features.float()
    values = values / values.norm(dim=0, keepdim=True).clamp_min(1e-8)
    similarity = values @ values.T
    n = len(values)
    return (
        torch.eye(n, device=values.device)
        + gamma * similarity @ similarity.T
    )


def otprune_indices(
    features: torch.Tensor, budget: int, gamma: float = 0.01
) -> torch.Tensor:
    return _dpp_greedy(_ot_kernel(features, gamma), budget)


def _calibrated_concept_scores(
    visual_tokens: torch.Tensor,
    text_tokens: torch.Tensor | None,
    contrast_min: float,
    calibrated: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return modality-calibrated patch-by-concept evidence.

    Raw cross-modal cosine values are not comparable across words because the
    frozen image projector and input embedding table have different nuisance
    offsets and scales.  We therefore robustly standardize every word over the
    patches of the current image.  A word becomes active only when its visual
    response is spatially selective; diffuse words receive zero weight.
    """
    n = len(visual_tokens)
    if text_tokens is None or text_tokens.numel() == 0:
        empty = torch.empty((n, 0), device=visual_tokens.device)
        return empty, torch.empty(0, device=visual_tokens.device), 0.0
    visual = F.normalize(visual_tokens.float(), dim=-1)
    text = F.normalize(text_tokens.float(), dim=-1)
    raw = visual @ text.T
    median = raw.median(dim=0).values
    if not calibrated:
        low = raw.min(dim=0).values
        high = raw.max(dim=0).values
        scores = ((raw - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
        weights = torch.ones(raw.shape[1], device=raw.device)
        weights = weights / weights.sum().clamp_min(1)
        return scores, weights, float((raw.max(dim=0).values - median).mean().item())
    absolute = (raw - median).abs()
    mad = absolute.median(dim=0).values
    robust_scale = (1.4826 * mad).clamp_min(1e-4)
    standardized = ((raw - median) / robust_scale).clamp(min=0, max=8)
    scale = standardized.max(dim=0).values.clamp_min(1e-4)
    scores = (standardized / scale).clamp(0, 1)
    contrast = (raw.max(dim=0).values - median) / robust_scale
    active = contrast >= contrast_min
    weights = contrast.clamp(min=0, max=8) * active
    if float(weights.sum().item()) > 0:
        weights = weights / weights.sum()
    else:
        weights = torch.zeros_like(weights)
    active_contrast = contrast[active]
    contrast_mean = (
        float(active_contrast.mean().item()) if active_contrast.numel() else 0.0
    )
    return scores, weights, contrast_mean


def _set_coverage(
    affinity: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    if selected.numel() == 0:
        return torch.zeros(len(affinity), device=affinity.device)
    return affinity[:, selected].max(dim=1).values


def _cross_modal_residual(
    visual: torch.Tensor,
    text: torch.Tensor | None,
    ridge: float,
) -> torch.Tensor:
    '''Remove the regularized text-subspace projection from visual values.'''
    values = visual.float()
    if text is None or text.numel() == 0:
        return values
    basis = text.float()
    gram = basis @ basis.T
    gram = gram + ridge * torch.eye(
        len(basis), device=basis.device, dtype=basis.dtype
    )
    coefficients = torch.linalg.solve(gram, basis @ values.T).T
    return values - coefficients @ basis


def _question_intent(question: str) -> str:
    '''Classify answer-independent query intent with transparent lexical rules.'''
    value = question.lower().strip()
    if re.search(
        r'\b(text|word|words|letter|letters|number|numbers|written|write|read|'
        r'reads|say|says|sign|label|title|brand|price|time|date)\b', value
    ):
        return 'ocr'
    if re.search(r'\([a-d]\)|\bchoices?\b|\boptions?\b', value):
        return 'choice'
    if re.search(
        r'^(is|are|was|were|do|does|did|can|could|has|have)\b|'
        r'\b(is|are) there\b', value
    ):
        return 'existence'
    if re.search(
        r'\b(left|right|above|below|behind|front|next|near|between|under|'
        r'over|where|position|relative)\b', value
    ):
        return 'relation'
    return 'local'


def _intent_stride_indices(
    features: torch.Tensor,
    vision_features: torch.Tensor,
    residual_features: torch.Tensor | None,
    salience: torch.Tensor,
    concept_scores: torch.Tensor,
    concept_weights: torch.Tensor,
    question: str,
    config: RouterConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    '''Intent-conditioned evidence geometry with bounded semantic refinement.'''
    n = len(features)
    k = min(max(1, config.budget), n)
    intent = _question_intent(question) if config.stride_use_intent_routing else 'local'
    severe = k / max(n, 1) <= 0.15
    expert = 'salience'
    if severe and intent == 'existence' and config.stride_use_diversity_expert:
        selected = divprune_indices(features, k)
        expert = 'diversity'
    else:
        use_residual = (
            residual_features is not None and severe and intent == 'choice'
        )
        use_vision = intent == 'ocr' and config.stride_use_vision_space
        default_geometry = (
            features if config.stride_use_projected_geometry else vision_features
        )
        geometry = (
            residual_features if use_residual
            else vision_features if use_vision
            else default_geometry
        )
        expert = 'vision_salience' if use_vision else expert
        selected = vispruner_indices(
            geometry, salience, k, config.vispruner_important_ratio
        )
    salience_score = _normalized_salience(salience)
    semantic_available = (
        config.stride_use_semantics
        and severe
        and intent in {'local', 'relation'}
        and concept_scores.shape[1] > 0
        and float(concept_weights.sum().item()) > 0
    )
    raw_semantic = (
        (concept_scores * concept_weights[None]).sum(dim=1)
        if semantic_available else torch.zeros(n, device=features.device)
    )
    semantic_anchors = 0
    semantic_gain_total = 0.0
    if semantic_available:
        protected_count = min(k, max(1, int(k * config.vispruner_important_ratio)))
        maximum = min(
            int((concept_weights > 0).sum().item()),
            max(1, round(k * config.stride_semantic_anchor_fraction)),
        )
        for concept in torch.argsort(concept_weights, descending=True).tolist():
            if semantic_anchors >= maximum or float(concept_weights[concept]) <= 0:
                break
            current = float(concept_scores[selected, concept].max().item())
            joint = concept_scores[:, concept] * (0.5 + 0.5 * salience_score)
            joint[selected] = -1
            candidate = int(joint.argmax().item())
            gain = float(concept_scores[candidate, concept].item()) - current
            if gain < config.stride_semantic_gain_min:
                continue
            removable = torch.arange(protected_count, k, device=features.device)
            if not removable.numel():
                break
            utility = (
                salience_score[selected[removable]]
                + 0.25 * raw_semantic[selected[removable]]
            )
            selected[int(removable[utility.argmin()].item())] = candidate
            semantic_anchors += 1
            semantic_gain_total += gain
    normalized = F.normalize(features.float(), dim=-1)
    distribution_coverage = _set_coverage(
        ((normalized @ normalized.T) + 1) / 2, selected
    ).mean()
    return selected, salience_score + raw_semantic, {
        'query_intent': intent,
        'selected_expert': expert,
        'severe_compression': severe,
        'semantic_anchor_tokens': semantic_anchors,
        'semantic_gain_total': semantic_gain_total,
        'distribution_coverage': float(distribution_coverage.item()),
        'semantic_active': semantic_available,
        'residual_space_active': bool(
            residual_features is not None and severe and intent == 'choice'
        ),
        'vision_space_active': bool(
            intent == 'ocr' and config.stride_use_vision_space
        ),
        'projected_geometry_active': config.stride_use_projected_geometry,
    }


def vispruner_indices(
    features: torch.Tensor,
    salience: torch.Tensor,
    budget: int,
    important_ratio: float = 0.5,
) -> torch.Tensor:
    """Official VisPruner rule: salient core plus iterative duplicate removal."""
    n = len(features)
    k = min(max(1, budget), n)
    important_count = min(k, max(0, int(k * important_ratio)))
    diverse_count = k - important_count
    order = torch.argsort(salience.float(), descending=True, stable=True)
    important = order[:important_count]
    residual = order[important_count:]
    normalized = F.normalize(features.float(), dim=-1)
    while diverse_count > 0 and len(residual) > diverse_count:
        remove_count = min(8, len(residual) - diverse_count)
        even = residual[::2]
        odd = residual[1::2]
        paired = min(len(even), len(odd))
        if paired == 0:
            break
        similarity = (
            normalized[even[:paired]] @ normalized[odd[:paired]].T
        ).max(dim=-1).values
        keep_even = torch.argsort(
            similarity, descending=True, stable=True
        )[remove_count:]
        pieces = [even[:paired][keep_even], odd[:paired]]
        if len(even) > paired:
            pieces.append(even[paired:])
        residual = torch.cat(pieces)
    return torch.cat((important, residual[:diverse_count]))


def _even_anchors(indices: torch.Tensor, count: int) -> torch.Tensor:
    if count >= len(indices):
        return indices
    positions = torch.linspace(
        0, len(indices) - 1, count, device=indices.device
    ).round().long()
    return indices[positions]


def _diverse_anchors(
    features: torch.Tensor,
    candidates: torch.Tensor,
    count: int,
    utility: torch.Tensor,
) -> torch.Tensor:
    if count >= len(candidates):
        return candidates
    local = features[candidates]
    distance = _cosine_distance(local)
    no_self = distance.clone()
    no_self.fill_diagonal_(float('inf'))
    isolation = normalize01(no_self.min(dim=1).values)
    local_utility = normalize01(utility[candidates].float())
    first = int((isolation * (0.5 + 0.5 * local_utility)).argmax().item())
    chosen = torch.empty(count, device=candidates.device, dtype=torch.long)
    chosen[0] = first
    minimum = distance[first].clone()
    minimum[first] = -1
    for position in range(1, count):
        objective = minimum * (0.5 + 0.5 * local_utility)
        objective[chosen[:position]] = -1
        index = int(objective.argmax().item())
        chosen[position] = index
        minimum = torch.minimum(minimum, distance[index])
        minimum[chosen[: position + 1]] = -1
    return candidates[chosen]


def _cluster_assignments(
    features: torch.Tensor,
    candidates: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    normalized = F.normalize(features.float(), dim=-1)
    return (
        normalized[candidates] @ normalized[anchors].T
    ).argmax(dim=1)


def visionzip_route(context: RoutingContext, config: RouterConfig) -> RouteResult:
    """Faithful interface-level port of the official VisionZip reducer.

    It retains a global CLS token, dominant patches from CLS attention, evenly
    spaced contextual anchors, and the official anchor-plus-residual-mean merge.
    """
    n = len(context.tokens)
    k = min(max(1, config.budget), n)
    if k >= n or context.global_vision_feature is None:
        if context.global_vision_feature is None and k < n:
            raise RuntimeError('VisionZip requires a vision CLS token')
        return dense(context.tokens)
    global_count = 1
    contextual_count = min(config.visionzip_context_tokens, max(1, k // 4))
    contextual_count = min(contextual_count, k - global_count)
    dominant_count = k - global_count - contextual_count
    order = torch.argsort(
        context.vision_salience.float(), descending=True, stable=True
    )
    dominant = order[:dominant_count]
    keep_mask = torch.ones(n, device=order.device, dtype=torch.bool)
    keep_mask[dominant] = False
    residual = torch.arange(n, device=order.device)[keep_mask]
    contextual = _even_anchors(residual, contextual_count)
    clusters = _cluster_assignments(
        context.vision_features, residual, contextual
    )

    rows = []
    hidden = [context.global_vision_feature]
    rows.append(torch.zeros(n, device=order.device, dtype=context.tokens.dtype))
    for index in dominant:
        row = F.one_hot(index, num_classes=n).to(context.tokens.dtype)
        rows.append(row)
        hidden.append(context.vision_features[index])
    for slot, anchor in enumerate(contextual):
        members = residual[clusters == slot]
        members = members[members != anchor]
        row = F.one_hot(anchor, num_classes=n).to(context.tokens.dtype)
        value = context.vision_features[anchor]
        if len(members):
            row = row + F.one_hot(members, num_classes=n).to(
                context.tokens.dtype
            ).mean(dim=0)
            value = value + context.vision_features[members].mean(dim=0)
        rows.append(row)
        hidden.append(value)
    anchors = torch.cat(
        (torch.tensor([-1], device=order.device), dominant, contextual)
    )
    return _compressed_result(
        context,
        torch.stack(hidden),
        torch.stack(rows),
        _normalized_salience(context.vision_salience),
        anchors,
        {
            'method_family': 'official_port',
            'source': 'VisionZip official repository',
            'salience_source': context.salience_source,
            'global_tokens': global_count,
            'dominant_tokens': dominant_count,
            'contextual_tokens': contextual_count,
            'merge_rule': 'anchor_plus_residual_mean',
        },
    )


def stride_route(context: RoutingContext, config: RouterConfig) -> RouteResult:
    """STRIDE: counterfactual semantic coverage over a protected core."""
    n = len(context.tokens)
    k = min(max(1, config.budget), n)
    if k >= n:
        return dense(context.tokens)
    concept_scores, concept_weights, semantic_contrast = (
        _calibrated_concept_scores(
            context.semantic_visual_tokens
            if context.semantic_visual_tokens is not None
            else context.tokens,
            context.text_tokens if config.stride_use_semantics else None,
            config.stride_concept_contrast_min,
            config.stride_use_modality_calibration,
        )
    )
    residual_features = (
        _cross_modal_residual(
            context.semantic_visual_tokens,
            context.text_tokens,
            config.stride_residual_ridge,
        )
        if config.stride_use_residual_space
        and context.semantic_visual_tokens is not None
        else None
    )
    anchors, selection_utility, routing_diagnostics = _intent_stride_indices(
        context.tokens,
        context.vision_features,
        residual_features,
        context.vision_salience,
        concept_scores,
        concept_weights,
        context.routing_prompt,
        config,
    )
    diagnostics = {
        'method_family': 'proposed',
        'source': 'STRIDE intent-conditioned evidence router',
        'text_source': context.text_source,
        'alignment_source': context.alignment_source,
        'anchor_tokens': k,
        'active_concepts': int((concept_weights > 0).sum().item()),
        'semantic_contrast': semantic_contrast,
        'retention_ratio': k / max(n, 1),
        'concept_contrast_min': config.stride_concept_contrast_min,
        'otprune_gamma': config.otprune_gamma,
        'semantic_anchor_fraction': config.stride_semantic_anchor_fraction,
        'semantic_gain_min': config.stride_semantic_gain_min,
        'uses_semantics': config.stride_use_semantics,
        'uses_intent_routing': config.stride_use_intent_routing,
        'uses_diversity_expert': config.stride_use_diversity_expert,
        'uses_modality_calibration': config.stride_use_modality_calibration,
        'uses_residual_space': config.stride_use_residual_space,
        'uses_projected_geometry': config.stride_use_projected_geometry,
        'uses_vision_space': config.stride_use_vision_space,
        'residual_ridge': config.stride_residual_ridge,
        'emits_original_tokens': True,
        'decoder_probe_layers': 0,
        **routing_diagnostics,
    }
    return _selected_result(
        context,
        anchors,
        selection_utility,
        diagnostics,
    )

@torch.inference_mode()
def route(
    method: str,
    context: RoutingContext,
    config: RouterConfig,
    seed: int = 0,
) -> RouteResult:
    if method not in METHODS:
        raise KeyError(f'Unknown method {method!r}; choose from {METHODS}')
    common = {
        'tokens': context.tokens,
        'budget': config.budget,
        'grid': context.grid,
    }
    if method == 'dense':
        result = dense(context.tokens)
        result.diagnostics.update({'method_family': 'control'})
        return result
    if method == 'random':
        result = random_pruning(**common, seed=seed)
        result.diagnostics.update({'method_family': 'control'})
        return result
    if method == 'pool':
        result = spatial_pooling(**common)
        result.diagnostics.update({'method_family': 'control'})
        return result
    if method == 'divprune':
        indices = divprune_indices(context.tokens, config.budget)
        score = torch.ones(len(context.tokens), device=context.tokens.device)
        return _selected_result(
            context,
            indices,
            score,
            {'method_family': 'official_port', 'source': 'DivPrune Algorithm 1'},
        )
    if method == 'otprune':
        indices = otprune_indices(
            context.tokens, config.budget, config.otprune_gamma
        )
        return _selected_result(
            context,
            indices,
            torch.diag(_ot_kernel(context.tokens, config.otprune_gamma)),
            {
                'method_family': 'official_port',
                'source': 'OTPrune official repository',
                'gamma': config.otprune_gamma,
            },
        )
    if method == 'vispruner':
        indices = vispruner_indices(
            context.vision_features,
            context.vision_salience,
            config.budget,
            config.vispruner_important_ratio,
        )
        return _selected_result(
            context,
            indices,
            context.vision_salience,
            {
                'method_family': 'official_port',
                'source': 'VisPruner official repository',
                'salience_source': context.salience_source,
                'important_ratio': config.vispruner_important_ratio,
            },
        )
    if method == 'visionzip':
        return visionzip_route(context, config)
    return stride_route(context, config)


def estimated_token_reduction(input_tokens: int, output_tokens: int) -> float:
    return 1 - output_tokens / max(input_tokens, 1)


def budget_from_reduction(input_tokens: int, reduction: float) -> int:
    return max(1, min(input_tokens, math.ceil(input_tokens * (1 - reduction))))
