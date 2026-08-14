from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RouteResult:
    """Output of a visual-token reducer.

    `tokens` always contains features in the VLM's original projected embedding
    space. Assignment is slot-by-input-patch. A zero row represents a global
    class token; a row whose sum exceeds one records VisionZip-style residual
    accumulation rather than a convex average.
    """

    tokens: torch.Tensor
    coordinates: torch.Tensor
    assignment: torch.Tensor
    importance: torch.Tensor
    selected_indices: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingContext:
    '''Frozen signals exposed at the vision-language interface.'''

    tokens: torch.Tensor
    vision_features: torch.Tensor
    vision_salience: torch.Tensor
    text_tokens: Any = None
    semantic_visual_tokens: Any = None
    text_source: str = 'not_requested'
    routing_prompt: str = ''
    grid: Any = None
    salience_source: str = 'unknown'
    language_relevance: Any = None
    alignment_source: str = 'not_requested'
    global_vision_feature: Any = None
    projector: Any = None
